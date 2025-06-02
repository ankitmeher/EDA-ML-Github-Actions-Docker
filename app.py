import pickle
from flask import Flask, request, jsonify, app, url_for, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('linear_regression_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')    

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    prediction = model.predict(final_input)[0]
    print(prediction)
    return render_template('index.html', prediction_text=f'Predicted Value: {prediction:.2f}')

if __name__ == "__main__":
    app.run(debug=True)