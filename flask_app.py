# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 19:49:24 2020

@author: subham
"""

import  os
import  pickle
import  numpy  as     np
import  pandas as     pd
from    flask  import Flask, request, render_template
from    PIL    import Image,ImageOps

app=Flask(__name__)
model=pickle.load(open('MNIST.pkl', 'rb'))
classes=[0,1,2,3,4,5,6,7,8,9]
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

@app.route('/', methods=['POST'])
def predict():
    
    files = request.files['image_data']
    files.save(os.path.join(ROOT_PATH,files.filename))
    data = pd.read_csv(os.path.join(ROOT_PATH,files.filename))
    
    #getting the csv file and converting it to Image type
    image = Image.fromarray(np.uint8(np.array(data)))
    image = np.array((ImageOps.invert(image)).resize((28, 28), Image.ANTIALIAS))
    image = (image.reshape(1,28,28,1))/255
    
    #predicting the result
    pred=model.predict(image)
    prediction=classes[np.argmax(pred[0])]
    accuracy=round(pred[0][np.argmax(pred[0])]*100,3)
    
    #returning the prediction and accuracy
    return "The pridicted number is {} and accuracy is {}".format(prediction,accuracy)
        
if( __name__ == "__main__"):
    app.run(threaded=False, debug=False)