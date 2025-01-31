from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf #type: ignore
from tensorflow.keras import Sequential # type: ignore
import joblib # type: ignore
import requests # type: ignore
import os
app = Flask(__name__)

# Example: Create and save a dummy model
model = Sequential()
model.save("bdg_prediction_model.h5")
# Load the trained model and scaler
model = tf.keras.models.load_model("bdg_prediction_model.h5")
scaler = joblib.load("scaler.pkl")

# Function to fetch real-time data from BDG Client API
def fetch_real_time_data():
    api_url = "https://api.bigdaddygame.cc/api/webapi/GetGameIssue"
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    return None

# Function to preprocess input data
def preprocess_input(period_number, past_data):
    past_numbers = [entry["Number"] for entry in past_data[-10:]]  # Use last 10 results
    past_numbers.append(period_number)
    past_numbers = np.array(past_numbers).reshape(1, -1)
    return scaler.transform(past_numbers)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to get prediction
@app.route('/predict', methods=['POST'])
def predict():
    period_number = int(request.form['period_number'])
    real_time_data = fetch_real_time_data()
    
    if not real_time_data:
        return jsonify({"error": "Failed to fetch real-time data"})
    
    input_data = preprocess_input(period_number, real_time_data)
    prediction = model.predict(input_data)
    predicted_number = int(np.round(prediction[0][0]))
    
    return jsonify({"predicted_next_number": predicted_number})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Get the PORT from Render
    app.run(host='0.0.0.0', port=port)
