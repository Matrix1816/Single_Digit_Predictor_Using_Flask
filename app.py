from flask import Flask,request,render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

app=Flask(__name__)
model=load_model('model.hdf5')

def p_img(img):
    img = ImageOps.grayscale(img)
    img = ImageOps.invert(img)
    img = img.resize((28,28))
    img = np.array(img)/255
    img = img.reshape(1,28,28,1)
    return img

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        file=request.files['file']
        if file:
            img = Image.open(file.stream)
            img = p_img(img)
            pred = model.predict(img)
            pred_class=np.argmax(pred)
            return render_template('result.html',pred_class=pred_class)
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True,use_reloader=False)

