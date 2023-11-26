from flask import Flask, render_template, request
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt 
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, classification_report)
from sklearn.feature_selection import f_classif
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])

def predict():
    try:
        # Ambil nilai dari form untuk dua fitur yang diperlukan
        feature1 = float(request.form['food'])
        feature2 = float(request.form['weight'])
        # feature3 = float(request.form['exercise'])

        # Lakukan prediksi menggunakan dua fitur tersebut
        prediction = model.predict([[feature1, feature2,]])

        return render_template("index.html", prediction_text="Kualitas fisik Anda: {}".format(prediction))
    except Exception as e:
        return render_template("index.html", prediction_text="Error: {}".format(str(e)))