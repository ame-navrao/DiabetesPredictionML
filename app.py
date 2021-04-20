# -*- coding: utf-8 -*-

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in_classifier = open('Classifier.pkl','rb')

pickle_in_standardScalar = open('StandardScalar.pkl', 'rb')

classifier = pickle.load(pickle_in_classifier)

sc = pickle.load(pickle_in_standardScalar)

@app.route('/')
def welcome():
    return "Diabetes Prediction Api"


@app.route('/predict')
def predict_note_authentication():
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: Pregnancies
        in: query
        type: number
        required: true
      - name: Glucose
        in: query
        type: number
        required: true
      - name: BloodPressure
        in: query
        type: number
        required: true
      - name: SkinThickness
        in: query
        type: number
        required: true
      - name: Insulin
        in: query
        type: number
        required: true
      - name: BMI
        in: query
        type: number
        required: true
      - name: DiabetesPedigreeFunction
        in: query
        type: number
        required: true
      - name: Age
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    Pregnancies = request.args.get('Pregnancies')
    Glucose = request.args.get('Glucose')
    BloodPressure = request.args.get('BloodPressure')
    SkinThickness  = request.args.get('SkinThickness')
    Insulin  = request.args.get('Insulin')
    BMI  = request.args.get('BMI')
    DiabetesPedigreeFunction  = request.args.get('DiabetesPedigreeFunction')
    Age  = request.args.get('Age')
    
    X = [
        [
            Pregnancies,
            Glucose,
            BloodPressure,
            SkinThickness,
            Insulin,
            BMI,
            DiabetesPedigreeFunction,
            Age,
        ]
    ]
    
    X = sc.transform(X)
    
    prediction = classifier.predict(X)
    if prediction == 0:
        return "You are not Diabetic! Have a Chocolate 😉"
    else:
        return "You are Diabetic! Please take care😥"
    return "Unexpected Response Something went wrong..."
   



    
if __name__ == '__main__':
    app.run()
    