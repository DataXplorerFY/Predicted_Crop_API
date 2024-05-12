from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
loaded_model = pickle.load(open("model.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    new_weather_conditions = [[temperature, humidity]]
    predicted_crop = loaded_model.predict(new_weather_conditions)
    return jsonify({'predicted_crop': predicted_crop[0]})

if __name__ == '__main__':
    app.run(debug=True)

