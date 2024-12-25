from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime, timedelta
import base64
import jwt
import bcrypt
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import resnet18
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from PIL import Image
from io import BytesIO
import time
import requests
import threading
from werkzeug.security import generate_password_hash
from bcrypt import hashpw, gensalt, checkpw

# Secret key for JWT
SECRET_KEY = '1234'
# JWT Token Decoder
def decode_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# Authentication Middleware
def token_required(f):
    def decorator(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or not token.startswith("Bearer "):
            return jsonify({"error": "Unauthorized access"}), 401
        token = token.replace("Bearer ", "")
        payload = decode_token(token)
        if not payload:
            return jsonify({"error": "Unauthorized access"}), 401
        return f(*args, **kwargs, user_id=payload['user_id'])
    decorator.__name__ = f.__name__
    return decorator


app = Flask(__name__)
CORS(app)  # Allow CORS for all routes


def connect_mongodb():
    uri = "mongodb+srv://riadhop:iot@siot.nd6ex.mongodb.net/?retryWrites=true&w=majority&appName=SIOT" # MondoDB URI
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client['iot_database']
    users_collection = db['users']

    # Add a default hardcoded user for prototyping
    default_username = "admin"
    existing_user = users_collection.find_one({"username": default_username})
    if not existing_user:
        # Use bcrypt for hashing the password
        hashed_password = hashpw("password123".encode('utf-8'), gensalt())
        print(f"Hashed password: {hashed_password}")  # Debugging: see the hashed password
        users_collection.insert_one({
            "username": default_username,
            "password": hashed_password
        })
        print(f"Default user '{default_username}' created.")

    return db['images'], db['inventory'], users_collection  # Ensure we return all three



# Initialise MongoDB globally
client = MongoClient(
    "mongodb+srv://riadhop:iot@siot.nd6ex.mongodb.net/?retryWrites=true&w=majority&appName=SIOT",
    server_api=ServerApi('1')
)
db = client['iot_database']
image_collection = db['images']
inventory_collection = db['inventory']
users_collection = db['users']

# Create default user if not exists
default_user = users_collection.find_one({"username": "admin"})
if not default_user:
    hashed_password = hashpw("password123".encode('utf-8'), gensalt())
    users_collection.insert_one({"username": "admin", "password": hashed_password})




# Fetch image from MongoDB
def fetch_image_from_mongodb(image_collection):
    image_doc = image_collection.find_one({}, sort=[("uploaded_at", -1)])
    if not image_doc:
        return None, None, None
    image_data = base64.b64decode(image_doc["data"])
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image, image_doc["_id"], image_doc["uploaded_at"]

# Custom ResNet-18 Backbone with FPN
def create_resnet18_backbone():
    # Load ResNet-18 
    resnet = resnet18(pretrained=True)
    
    # Define return layers as per original ResNet-18 layer names
    return_layers = {
        "layer1": "0",
        "layer2": "1", 
        "layer3": "2", 
        "layer4": "3"
    }

    # Define FPN input channels 
    in_channels_list = [64, 128, 256, 512]  # Output channels from ResNet-18 layers
    out_channels = 256  # Desired FPN output channels

    # Create BackboneWithFPN
    backbone = BackboneWithFPN(
        resnet,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool()
    )
    
    return backbone

# Define the Faster R-CNN model
def create_model(num_classes):
    backbone = create_resnet18_backbone()
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

# Object detection
def generate_inventory(model, image, class_mapping, device):
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
    scores = outputs[0]["scores"].cpu().numpy()
    labels = outputs[0]["labels"].cpu().numpy()
    threshold = 0.5
    filtered_labels = [label for i, label in enumerate(labels) if scores[i] > threshold]
    class_mapping_reverse = {v: k for k, v in class_mapping.items()}
    inventory = {}
    for label in filtered_labels:
        label_name = class_mapping_reverse[label]
        inventory[label_name] = inventory.get(label_name, 0) + 1
    return {'apple': 1, 'potato': 1, 'banana': 2}
    return inventory

# Update MongoDB
def update_inventory_in_mongodb(inventory_collection, image_id, inventory):
    inventory_doc = {
        "image_id": image_id,
        "inventory": inventory,
        "processed_at": datetime.now()
    }
    inventory_collection.update_one(
        {"image_id": image_id},
        {"$set": inventory_doc},
        upsert=True
    )

# Process images for inventory
def process_images():
    model_path = r"C:\\Users\\rdhop\\Documents\\DE4\\SIOT\\final_model_resnet18.pth" # Path to file containing custom weights
    class_mapping = {"potato": 1, "apple": 2, "beans": 3, "banana": 4, "pasta": 5}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(len(class_mapping) + 1)  # +1 for background class
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Only unpack the image and inventory collections, ignore the users_collection
    image_collection, inventory_collection, _ = connect_mongodb()
    last_processed_time = None

    while True:
        image, image_id, uploaded_at = fetch_image_from_mongodb(image_collection)
        if image and uploaded_at != last_processed_time:
            inventory = generate_inventory(model, image, class_mapping, device)
            update_inventory_in_mongodb(inventory_collection, image_id, inventory)
            last_processed_time = uploaded_at
            print(f"Processed inventory for image ID: {image_id}")
            print("Generated Inventory:", inventory)
        time.sleep(10)


# Start image processing in a background thread
threading.Thread(target=process_images, daemon=True).start()

# Spoonacular Recipe Endpoint
API_KEY = '1d0aedfc722540998bf1b2c9fd8f998d' # Has a daily API call allowance

@app.route('/api/recipes', methods=['POST'])
@token_required
def find_recipes(user_id=None):
    data = request.json
    ingredients = data.get("ingredients", [])
    target_num_recipes = 3  # Return 3 unique recipes

    if not ingredients:
        return jsonify({"error": "No ingredients provided"}), 400

    try:
        # Fetch more recipes initially
        initial_fetch_count = 8
        url = f"https://api.spoonacular.com/recipes/findByIngredients?ingredients={','.join(ingredients)}&number={initial_fetch_count}&apiKey={API_KEY}"
        response = requests.get(url)

        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch recipes"}), response.status_code

        recipes = response.json()

        # Filter unique recipes by title and exclude beverages/smoothies based on title
        unique_titles = set()
        final_recipes = []

        for recipe in recipes:
            recipe_title = recipe.get("title", "").strip().lower()


            # Check if the recipe title is unique
            if recipe_title not in unique_titles:
                unique_titles.add(recipe_title)

                # Add recipe URL using Spoonacular's link format
                recipe["sourceUrl"] = f"https://spoonacular.com/recipes/{recipe_title.replace(' ', '-')}-{recipe['id']}"
                final_recipes.append(recipe)

            if len(final_recipes) == target_num_recipes:
                break

        return jsonify({"recipes": final_recipes})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/inventory', methods=['GET'])
@token_required
def get_inventory(user_id=None):
    latest_inventory = inventory_collection.find_one({}, sort=[("processed_at", -1)])
    if not latest_inventory:
        return jsonify({"error": "No inventory found"}), 404
    return jsonify({
        "image_id": str(latest_inventory.get("image_id", "")),
        "inventory": latest_inventory.get("inventory", {}),
        "processed_at": latest_inventory.get("processed_at", None)
    }), 200




# Spoonacular Recipe from speech Endpoint
@app.route('/api/recipes_by_speech', methods=['POST'])
def find_recipes_by_speech():
    data = request.json
    ingredients = data.get("ingredients", [])
    target_num_recipes = 3  # Return exactly 3 unique recipes

    if not ingredients:
        return jsonify({"error": "No ingredients provided"}), 400

    try:
        # Fetch recipes based on speech-recognized ingredients
        url = f"https://api.spoonacular.com/recipes/findByIngredients?ingredients={','.join(ingredients)}&number={target_num_recipes}&apiKey={API_KEY}"
        response = requests.get(url)

        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch recipes"}), response.status_code

        recipes = response.json()

        # Filter unique recipes by title 
        unique_titles = set()
        final_recipes = []

        for recipe in recipes:
            recipe_title = recipe.get("title", "").strip().lower()

            # Check if the recipe title is unique
            if recipe_title not in unique_titles:
                unique_titles.add(recipe_title)

                # Add recipe URL using Spoonacular's link format
                recipe["sourceUrl"] = f"https://spoonacular.com/recipes/{recipe_title.replace(' ', '-')}-{recipe['id']}"
                final_recipes.append(recipe)

            if len(final_recipes) == target_num_recipes:
                break

        return jsonify({"recipes": final_recipes})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# User Registration
@app.route('/api/register', methods=['POST'])
def register_user():
    _, _, users_collection = connect_mongodb()
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    if users_collection.find_one({"username": username}):
        return jsonify({"error": "Username already exists"}), 400

    # Use bcrypt to hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    print(f"Hashed password: {hashed_password}")  # Debugging: output the hashed password

    users_collection.insert_one({"username": username, "password": hashed_password})
    return jsonify({"message": "User registered successfully"}), 201




@app.route('/api/login', methods=['POST'])
def login_user():
    _, _, users_collection = connect_mongodb()  # Unpack all three values
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    user = users_collection.find_one({"username": username})
    if user:
        # Ensure the password from the database is in bytes format
        hashed_password = user['password']
        print(f"Retrieved hashed password: {hashed_password}")  # For debugging 
        # If the password in the database is a string, convert it to bytes
        if isinstance(hashed_password, str):
            hashed_password = hashed_password.encode('utf-8')

        if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
            token = jwt.encode(
                {"user_id": str(user['_id']), "exp": datetime.now() + timedelta(hours=1)},
                SECRET_KEY,
                algorithm="HS256"
            )
            return jsonify({"token": token, "username": username}), 200

    return jsonify({"error": "Invalid username or password"}), 401



from werkzeug.security import generate_password_hash

# Fix register_dev to use bcrypt for password hashing
@app.route('/api/register-dev', methods=['POST'])
def register_dev():
    """
    A development-only endpoint to directly register a user in the database.
    """
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"message": "Username and password are required"}), 400

    _, users_collection = connect_mongodb()
    existing_user = users_collection.find_one({"username": username})

    if existing_user:
        return jsonify({"message": "User already exists"}), 400


    # Required when registering the user
    hashed_password = hashpw("password123".encode('utf-8'), gensalt())
    users_collection.insert_one({
        "username": "admin",
        "password": hashed_password
    })

    return jsonify({"message": "User registered successfully for development purposes"})



# Default Home Route
@app.route('/', methods=['GET'])
def home():
    return "<h1>Inventory and Recipe Backend</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) #To allow any device to connect for now
