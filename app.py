from flask import Flask,request, url_for, redirect, render_template
from flask import Flask, request, jsonify, render_template
from flask import Flask, render_template,request,redirect,url_for
import pickle
import numpy as np
import joblib
import sqlite3
import csv

app = Flask(__name__)

model = joblib.load("diabetes-predict.pkl")
with open('templates/Testing.csv', newline='') as f:
        reader = csv.reader(f)
        symptoms = next(reader)
        symptoms = symptoms[:len(symptoms)-1]

@app.route("/")
@app.route('/diabetes')
def diabetes():
    return render_template("diabetes.html")



@app.route('/predict',methods=['POST','GET'])
def predict():  
    int_features= [float(x) for x in request.form.values()]
    print(int_features,len(int_features))
    final=[np.array(int_features)]
    print(final)
    prediction=model.predict(final)
    output=round(prediction[0],2)
    print(output)
    if output==1:
        return render_template('diabetes.html',pred='You have diabetes',result="diabetes")
    else:
        return render_template('diabetes.html',pred='You dont have diabetes',result="not diabetes")



@app.route('/find_doctor', methods=['POST'])
def get_location():
    location = request.form['doctor']
    return render_template('find_doctor.html',location=location,symptoms=symptoms)

@app.route('/drug', methods=['POST'])
def drugs():
    medicine = request.form['medicine']
    return render_template('homepage.html',medicine=medicine,symptoms=symptoms)


if __name__ == '__main__':
    app.run(debug=True)
