# Importing essential libraries
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler


# Load the Random Forest CLassifier model
filename1 = 'random_forest_classifier.pkl'
model = pickle.load(open(filename1, 'rb'))

filename2 = 'input.pkl'
input_data = pickle.load(open(filename2, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        Age = int(request.form['Age'])
        Sex = request.form.get('Sex')
        ChestPainType = request.form.get('ChestPainType')
        RestingBP = int(request.form['RestingBP'])
        Cholesterol = int(request.form['Cholesterol'])
        FastingBS = request.form.get('FastingBS')
        RestingECG = int(request.form['RestingECG'])
        MaxHR = int(request.form['MaxHR'])
        ExerciseAngina = request.form.get('ExerciseAngina')
        Oldpeak = float(request.form['Oldpeak'])
        ST_Slope = request.form.get('ST_Slope')
    
        
        data = pd.DataFrame(np.array([[Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope]]))
        input_data.append(data)
        standardizer = StandardScaler()
        X_scaled = standardizer.fit_transform(input_data)
        result = model.predict(X_scaled)
        array_length = len(result)
        last_element = result[array_length - 1]

        
        return render_template('result.html', prediction=last_element)
        
        

if __name__ == '__main__':
	app.run(debug=True)