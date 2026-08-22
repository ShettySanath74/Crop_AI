from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import os

app = Flask(__name__)

# Train the SVM model if it's not already saved
def train_svm_model():
    # Load the dataset
    data = pd.read_csv("Crop_recommendation.csv")  # Make sure the dataset is in the same folder as app.py

    # Define features and target variable
    X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]  # Features
    y = data['label']  # Target (Crop)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the SVM model
    svm_model = SVC()

    # Train the model
    svm_model.fit(X_train, y_train)

    # Save the model to a file
    with open('svm_crop_model.pkl', 'wb') as file:
        pickle.dump(svm_model, file)

    return svm_model  # Return the trained model

# Route to render the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to predict the crop based on input data
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Load or train the SVM model
        if os.path.exists('svm_crop_model.pkl'):
            with open('svm_crop_model.pkl', 'rb') as file:
                svm_model = pickle.load(file)
        else:
            svm_model = train_svm_model()

        # Prepare the input for prediction
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = svm_model.predict(input_data)

        # Render the result template with prediction
        return render_template('result.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)

 
   