from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open('PredictionModel.pkl', 'rb') as f:
    model = pickle.load(f)

# Load symptom description data
symptom_desc = pd.read_csv('symptom_Description.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get symptoms from the form
    symptoms = request.form['symptoms']
    symptoms_list = symptoms.split(",")
    symptoms_array = np.array(symptoms_list)

    # Make a prediction using the trained model
    prediction = model.predict(pd.DataFrame(columns=['symptom1', 'symptom2', 'symptom3'], data=symptoms_array.reshape(1, 3)))
    output = ' '.join(prediction).strip('[]')

    # Get the description of the predicted disease
    index = symptom_desc['Disease'].str.replace(" ", "").tolist().index(output)
    description = symptom_desc['Description'][index]

    return render_template('index.html', prediction_text='Predicted Disease: {}'.format(output), description_text='Description: {}'.format(description))

if __name__ == '__main__':
    app.run(debug=True)
