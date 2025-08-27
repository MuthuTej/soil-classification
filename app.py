import os
import base64
import io
import numpy as np
import joblib
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load model and classes
model = load_model("soil_mobilenet.h5")
class_indices = joblib.load("soil_class_indices.pkl")
classes = {v: k for k, v in class_indices.items()}

@app.route("/")
def home():
    return "Soil Classification API is running ✅"

@app.route("/predict_base64", methods=["POST"])
def predict_base64():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Decode base64 image
        img_bytes = base64.b64decode(data["image"])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Preprocess
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)
        class_id = np.argmax(preds)
        confidence = float(np.max(preds))

        return jsonify({
            "prediction": classes[class_id],
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)

