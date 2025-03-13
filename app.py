from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained model
try:
    model = tf.keras.models.load_model("model.h5")
    input_shape = model.input_shape[1:3]  # Auto-detect required input size
    print(f"\u2705 Model loaded successfully! Expected input size: {input_shape}")
except Exception as e:
    print(f"\u274c Error loading model: {e}")
    model = None
# **Updated Class Labels & Explanations**
CLASS_INFO = {
    0: {"name": "Actinic keratosis", "explanation": "Suraj ki zyada roshni se hone wali chamdi ki samasya, jo future me cancer ban sakti hai.", "risk": "High Risk"},
    1: {"name": "Basal cell carcinoma", "explanation": "Sabse common skin cancer, jo dheere-dheere badhta hai aur zyada khatarnak nahi hota.", "risk": "Medium Risk"},
    2: {"name": "Benign keratosis", "explanation": "Chamdi ka ek nuksaan jo cancer nahi hai, par dikhta hai jaise ho sakta hai.", "risk": "Low Risk"},
    3: {"name": "Dermatofibroma", "explanation": "Ek chhoti, hard, aur be-nuksaan wali chamdi ki ganth, jo generally safe hoti hai.", "risk": "Low Risk"},
    4: {"name": "Melanoma", "explanation": "Sabse dangerous skin cancer, jo jaldi failta hai aur jaanleva ho sakta hai.", "risk": "High Risk"},
    5: {"name": "Nevus", "explanation": "Simple mole ya til jo zyada tar nuksaan nahi karta.", "risk": "Low Risk"},
    6: {"name": "Squamous cell carcinoma", "explanation": "Ek doosra common skin cancer jo basal cell se thoda zyada aggressive hota hai.", "risk": "High Risk"},
    7: {"name": "Vascular lesion", "explanation": "Khoon ki naaliyo se judi chamdi ki problem, jo zyada tar harmful nahi hoti.", "risk": "Low Risk"},
    8: {"name": "No Cancer", "explanation": "Koi bhi cancer nahi hai, aap safe ho! \u2705", "risk": "No Risk"}
}

# Image preprocessing function
def preprocess_image(image):
    try:
        image = image.convert("RGB")  # Ensure RGB format
        target_size = input_shape if input_shape else (150, 150)  # Default size if not detected
        image = image.resize(target_size)  # Resize based on model input
        image = np.array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        print(f"\u274c Error in image preprocessing: {e}")
        return None

# Define API route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return jsonify({"error": "Invalid image format"}), 400

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = int(np.argmax(predictions, axis=1)[0])  # Get class index
        confidence = float(np.max(predictions))  # Get confidence score
        
        # Get human-readable class name and explanation
        disease_info = CLASS_INFO.get(predicted_class, {"name": "Unknown Condition", "explanation": "No details available.", "risk": "Unknown"})
        
        # Recommendation based on risk level
        recommendation = ""
        if disease_info["risk"] == "High Risk":
            recommendation = "⚠️ Jaldi se dermatologist se consult karein!"
        elif disease_info["risk"] == "Medium Risk":
            recommendation = "🔎 Nazar rakhein, agar badlav aaye toh doctor se milna zaroori hai."
        else:
            recommendation = "✅ Aap safe hain, lekin regular checkup achha hota hai."
        
        return jsonify({
            "disease": disease_info["name"],
            "explanation": disease_info["explanation"],
            "confidence": f"{confidence * 100:.2f}%",  # Convert to percentage
            "risk_level": disease_info["risk"],
            "recommendation": recommendation
        })
    
    except Exception as e:
        print(f"\u274c Error in prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default port 5000
    app.run(host="0.0.0.0", port=port, debug=True)
