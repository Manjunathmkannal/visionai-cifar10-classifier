# ✨ VisionAI — CIFAR-10 Image Classifier

An AI-powered image classification web application built with **Python, Flask, TensorFlow, and Convolutional Neural Networks (CNNs)**. Upload an image through a clean, responsive interface and let the trained deep learning model predict its category.

## 🚀 Project Overview

VisionAI is a web-based image classification system that demonstrates how deep learning can identify objects in images. It combines a trained CNN model with a Flask backend and a modern web interface to provide an interactive image prediction experience.

## ✨ Features

- 🖼️ **Image Upload:** Upload images in PNG or JPG format.
- 🤖 **AI-Powered Prediction:** Uses a trained CNN model for image classification.
- ⚡ **Deep Learning Integration:** Built with TensorFlow and Keras.
- 🌐 **Web Application:** Flask handles image uploads and prediction requests.
- 🎨 **Modern UI:** Responsive interface with image preview and prediction results.
- 📦 **Pre-trained Model:** Loads the saved `image_classifier_cnn.keras` model.

## 🧠 CIFAR-10 Dataset

The model is designed for the 10 classes in the CIFAR-10 dataset:

| Class ID | Category |
|---:|---|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

**Note:** CIFAR-10 consists of small, 32 × 32 pixel images. Predictions on arbitrary real-world images may be less accurate because their content and appearance can differ from the training data.

## 🛠️ Technologies Used

- **Python** — Application logic
- **Flask** — Web framework
- **TensorFlow / Keras** — Deep learning and model inference
- **NumPy** — Numerical processing
- **Pillow** — Image handling
- **HTML5, CSS3, JavaScript** — Frontend interface
- **CNN** — Image classification architecture

## 📁 Project Structure

```text
visionai-cifar10-classifier/
├── app.py
├── ai image classi.py
├── image_classifier_cnn.keras
├── templates/
│   └── index.html
├── static/
│   └── uploads/
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Manjunathmkannal/visionai-cifar10-classifier.git
cd visionai-cifar10-classifier
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install flask tensorflow pillow numpy
```

### 4. Run the application

```bash
python app.py
```

### 5. Open the web application

Visit:

```text
http://127.0.0.1:5000
```

Upload a JPG or PNG image and click **Analyze Image** to view the predicted class.

## 🔍 How It Works

1. The user uploads an image through the web interface.
2. Flask receives the uploaded file.
3. The backend preprocesses the image into the format expected by the trained model.
4. The CNN model generates predictions.
5. The predicted class is displayed on the results page.

*The preprocessing steps and class-label mapping must match the configuration used when training the model.*

## 💻 Example Use Cases

- Learning and demonstrating deep learning concepts
- Exploring CNN-based image classification
- Building interactive machine learning web applications
- Demonstrating Python and TensorFlow integration

## 🔮 Future Improvements

- [ ] Display prediction confidence scores.
- [ ] Show top-3 predicted categories.
- [ ] Add prediction history.
- [ ] Improve image preprocessing and validation.
- [ ] Deploy the application to a public hosting platform.
- [ ] Add automated tests and model evaluation metrics.

## 👨‍💻 Author

**Manjunath M Kannal**

GitHub: [@Manjunathmkannal](https://github.com/Manjunathmkannal)

## ⭐ Support

If you find this project useful for learning or experimentation, consider giving the repository a star.

---

*Built as a deep learning and web development project using Python, Flask, and TensorFlow.*
