import pandas as pd
import numpy as np
import pickle
import os
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder





app = Flask(__name__, template_folder="templates")

@app.route('/', methods=['GET'])
@app.route('/home')
def index():
    return render_template('home.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('home.html')

@app.route('/pred', methods=['GET'])
def page():
    return render_template('upload.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Load the model
        print("[INFO] loading model...")
        model = pickle.loads(open('mlmodel.pkl', "rb").read())

        # Get the form values and convert them to numerical data types
        goal = float(request.form['goal'])
        
        staff_pick = bool(request.form['staff_pick'])
        backers_count = int(request.form['backers_count'])
        usd_pledged = float(request.form['usd_pledged'])
        category_text = str(request.form['category'])

        # Instantiate the LabelEncoder
        label_encoder = LabelEncoder()

        # Fit the LabelEncoder on the category_text values and transform them to integer labels
        category = label_encoder.fit_transform([category_text])[0]



        # Create a feature vector from the form values
        feature_vector = np.array([goal, staff_pick, backers_count, usd_pledged, category])

        # Make a prediction using the model
        prediction = model.predict([feature_vector])[0]

        if prediction == 0:
            output="Fail"
        else:
            output="Successful"
        return render_template('upload.html', prediction_text=output)



        return render_template('upload.html', prediction_text=output)
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)

