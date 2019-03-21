from flask import Flask,request
import pandas as pd
import numpy as np
from keras.models import load_model
from keras import backend
import json

app=Flask(__name__)

@app.route("/")
def funk():
    return "Server working"
    
@app.route("/predict",methods=["POST"])
def functi():
    if request.method=="POST":
        age=int(request.form.get('age'))
        height=int(request.form.get('height'))
        weight=int(request.form.get('weight'))
        totalcal=int(request.form.get('totalcal'))
        totalfat=int(request.form.get('totalfat'))
        totalprot=int(request.form.get('totalprot'))
        totalcarbs=int(request.form.get('totalcarbs'))
        cigarettes=int(request.form.get('cigarettes'))
        alcohol=int(request.form.get('alcohol'))
        caloburnt=int(request.form.get('caloburnt'))

        input_pred=np.array([age,height,weight,totalcal,totalfat,totalprot,totalcarbs,cigarettes,alcohol,caloburnt])
        input_pred=input_pred.reshape(-1,10)
        model = load_model('model.h5')
        y=model.predict(input_pred)
        y=y>0.5
        b=y.tolist()
        backend.clear_session()
        return json.dumps(b)

if __name__ == '__main__':
    app.run(debug=True)
