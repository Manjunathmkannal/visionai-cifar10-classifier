
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = os.path.join("..", "image_classifier_cnn.keras")
model = load_model(MODEL_PATH)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            extension = os.path.splitext(file.filename)[1].lower()

            if extension not in [".jpg", ".jpeg", ".png"]:
                return "Please upload a JPG or PNG image.", 400

            filename = str(uuid.uuid4()) + extension
            full_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(full_path)

            image = load_img(full_path, target_size=(32, 32))
            image_array = img_to_array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array, verbose=0)
            predicted_index = int(np.argmax(predictions[0]))

            prediction = CLASS_NAMES[predicted_index]
            image_path = "/" + full_path.replace("\\", "/")

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)
