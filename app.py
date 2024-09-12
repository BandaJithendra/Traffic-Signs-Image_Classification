from flask import Flask, render_template, request, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
import aspose.words as aw

doc = aw.Document()
bulider = aw.DocumentBuilder(doc)

categories = ["Speed limit (20km/h)","Speed limit (30km/h)","Speed limit (50km/h)","Speed limit (60km/h)",
              "Speed limit (70km/h)","Speed limit (80km/h)","End of speed limit (80km/h)","Speed limit (100km/h)",
              "Speed limit (120km/h)","No passing","No passing for vechiles over 3.5 metric tons",
              "Right-of-way at the next intersection","Priority road","Yield","Stop","No vechiles",
              "Vechiles over 3.5 metric tons prohibited","No Entry","General caution","Dangerous curve to the left",
              "Dangerous curve to the right","Double curve","Bumpy road","Slippery road","Road narrows on the right",
              "Road work","Traffic signals","Pedestrians","Children crossing","Bicycles crossing","Beware of ice/snow",
              "Wild animals crossing","End of all speed and passing limits","Turn right ahead","Turn left ahead",
              "Ahead only","Go straight or right","Go straight or left","Keep right","Keep left","Roundabout mandatory",
              "End of no passing","End of no passing by vechiles over 3.5 metric tons"]

model1 = load_model("Model")

app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('Home.html')

@app.route('/home',methods=['POST'])
def home():
    file = request.form['file']
    shape = bulider.insert_image(file)
    shape.get_shape_renderer().save(file,aw.saving.ImageSaveOptions(aw.SaveFormat.JPEG))
    img = Image.open(file)
    img = img.resize((32,32))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    preds=model1.predict(x)
    pred = np.argmax(preds, axis=-1)
    # return categories[pred[0]]
    return render_template("output.html",Output = categories[pred[0]])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4001,debug=True)
