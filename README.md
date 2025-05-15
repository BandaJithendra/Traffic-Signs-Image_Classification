# 🚦 Traffic Sign Classification Web App

This is a deep learning-based web application that classifies traffic signs from images using a Convolutional Neural Network (CNN). The model is trained using Keras in a Jupyter Notebook and deployed via a Flask web interface.

---

## 🧠 Overview

- **Model Training**: Performed using `app.ipynb` with TensorFlow/Keras.
- **Model Usage**: The trained model is saved and loaded in `app.py` to serve predictions.
- **Input**: Image of a traffic sign
- **Output**: Predicted class label (e.g., "Stop", "Speed limit 60km/h", etc.)

---

## 📁 Project Structure

├── [app.ipynb] # Notebook used to train the deep learning model <br/>
├── [app.py] # Flask application that uses the trained model <br/>
├── Model/ # Directory containing the saved Keras model <br/>
├── Data/ # Directory ccontaining training images
├── templates/ <br/>
│ ├── [Home.html] # Image upload UI <br/>
│ └── [output.html] # Prediction result UI <br/>
└── [requirements.txt] # Python dependencies <br/> 


---

## 🧪 How It Works

1. **Training**: Run `app.ipynb` to train the CNN model on traffic sign images (GTSRB dataset or similar).
2. **Saving**: The model is saved to the `Model/` directory.
3. **Serving**: `app.py` loads the saved model and uses it to classify uploaded traffic sign images.
4. **Prediction**: The app predicts one of 43 traffic sign categories using the uploaded image.

---

## 🛠️ Technologies Used

- Python 3
- TensorFlow / Keras
- Flask
- PIL (Pillow)
- Aspose.Words (used for image rendering)
- HTML/CSS with Jinja2 templates

---

## 🚀 Getting Started

```bash
### 1. Clone the Repository
git clone https://github.com/yourusername/traffic-sign-classification-app.git
cd traffic-sign-classification-app

# 2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Train the Model (Optional if already trained)
# Open and run the notebook to generate the model
jupyter notebook app.ipynb

# 5. Run the Web App
python app.py

# 6. Visit in Browser
# Open http://localhost:4001/ to use the app.

