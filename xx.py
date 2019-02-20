from flask import Flask
import pandas as pd
import numpy as np
from keras.models import load_model
import json
import tensorflow as tf

app=Flask(__name__)

def load_custommodel():
	global model
	model = load_model("model.h5")
	global graph
	graph = tf.get_default_graph()

@app.route("/")
def functi():
    input_values=pd.Series([21,176,56,1500,54,45,245,0,0,1453])
    input_pred=np.array(input_values)
    input_pred=input_pred.reshape(-1,10)
    with graph.as_default():
        y=model.predict(input_pred)
        y=y>0.5
        b=y.tolist()
        return json.dumps(b)

if __name__ == '__main__':
    load_custommodel()
    app.run(debug=True)
