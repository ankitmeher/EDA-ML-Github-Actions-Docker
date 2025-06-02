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
def predict_api():
    data = request.json['data']
    print(data)
    reshaped_data = np.array(list(data.values())).reshape(1, -1)
    new_data = scaler.transform(reshaped_data)
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)