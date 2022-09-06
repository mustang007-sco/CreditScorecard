import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
##Load model
model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('transfer.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    #new_data = scaler.transform(data)
    output = model.predict(data)
    print(output)
    return jsonify(output)


@app.route('/predict' , methods = ['POST'])
def predict():
    data = request.form
    final_input = scaler.transform(data)
    print(data)
    print(final_input)
    output = model.predict(data)
    return render_template("home.html",prediction_text = "The Scorecard prediction score is {}".format(output))

if __name__ =="__main__":
    app.run(debug=True)




