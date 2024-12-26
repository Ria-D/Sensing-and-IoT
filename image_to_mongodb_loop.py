import cv2
import base64
import time
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime

# MongoDB connection function
def connect_mongodb():
    uri = "mongodb+srv://riadhop:iot@siot.nd6ex.mongodb.net/?retryWrites=true&w=majority&appName=SIOT"
    try:
        client = MongoClient(uri, server_api=ServerApi('1'))
        db = client['iot_database']
        return db['images']
    except Exception as e:
        print("Failed to connect to MongoDB:", e)
        return None

# Capture image from Raspberry Pi camera
def capture_image(image_path):
    try:
        cam = cv2.VideoCapture(0)  # Open default webcam (use /dev/video0 for Raspberry Pi)
        if not cam.isOpened():
            print("Error: Cannot open webcam")
            return None

        ret, frame = cam.read()
        if not ret:
            print("Error: Failed to capture image")
            cam.release()
            return None

        # Save the image locally
        cv2.imwrite(image_path, frame)
        print(f"Image saved: {image_path}")
        cam.release()
        return image_path
    except Exception as e:
        print("Error capturing image:", e)
        return None

# Upload image to MongoDB
def upload_image_to_mongodb(collection, image_path):
    try:
        image_id = image_path.split('/')[-1]  # Use filename as the unique identifier
        # Check if image already exists in MongoDB
        existing_image = collection.find_one({"filename": image_id})
        
        if existing_image:
            # If the image exists, update it
            collection.update_one(
                {"filename": image_id},
                {"$set": {"data": base64.b64encode(open(image_path, "rb").read()).decode('utf-8'),
                          "uploaded_at": datetime.now()}}
            )
            print(f"Image with ID '{image_id}' successfully updated in MongoDB.")
        else:
            # If the image doesn't exist, insert it
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')

            # Create a document to insert
            image_document = {
                "filename": image_id,
                "data": image_data,
                "uploaded_at": datetime.now()
            }

            # Insert into MongoDB
            result = collection.insert_one(image_document)
            print(f"Image successfully uploaded to MongoDB. Document ID: {result.inserted_id}")

    except Exception as e:
        print("Failed to upload image to MongoDB:", e)

# Main function to automate capturing and uploading image
def main():
    save_path = "captured_image.jpg"
    
    # Connect to MongoDB
    collection = connect_mongodb()
    if collection is None:
        return

    # Loop to continuously capture images and upload them to MongoDB
    while True:
        image_path = capture_image(save_path)
        if image_path:
            upload_image_to_mongodb(collection, image_path)
        time.sleep(30)  # Capture and upload every 10 seconds (you can adjust the interval)

if __name__ == "__main__":
    main()
