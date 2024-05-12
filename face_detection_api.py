import io
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

def analyze_image(image_bytes):
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_bytes)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # If faces are detected, return successful scan, otherwise return unsuccessful scan
        if len(faces) > 0:
            return {"result": "Successful scan. Faces detected."}
        else:
            return {"result": "Unsuccessful scan. No faces detected."}
    
    except Exception as e:
        return {"error": str(e)}

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get image stream from request
        image_bytes = request.data
        
        # Analyze the image
        result = analyze_image(image_bytes)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
