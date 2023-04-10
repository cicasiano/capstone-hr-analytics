# Flask
from flask import Flask, render_template, request
# Data manipulation
import pandas as pd
# Matrices manipulation
import numpy as np
# Script logging
import logging
# ML model
import joblib
# JSON manipulation
import json
# Utilities
import sys
import os

# Current directory
current_dir = os.path.dirname(__file__)

# Flask app
app = Flask(__name__, static_folder = 'static', template_folder = 'template')

# Logging
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

# Function
def ValuePredictor(data = pd.DataFrame):
	# Model name
	model_name = 'models/rfc_model.pkl'
	# Directory where the model is stored
	model_dir = os.path.join(current_dir, model_name)
	# Load the model
	loaded_model = joblib.load(open(model_dir, 'rb'))
	# Predict the data
	result = loaded_model.predict(data)
	return result[0]

# Home page
@app.route('/')
def home():
	return render_template('index.html')

# Prediction page
@app.route('/prediction', methods = ['POST'])
def predict():
	if request.method == 'POST':
		# Get the data from form
		age = request.form['age']
		gender = request.form['gender']
		marital_status = request.form['marital_status']
		distancefromhome = request.form['distancefromhome']
		numcompaniesworked = request.form['numcompaniesworked']
		totalworkingyears = request.form['totalworkingyears']
		yearsatcompany = request.form['yearsatcompany']
		yearsincurrentrole = request.form['yearsincurrentrole']
		yearswithcurrentmanager = request.form['yearswithcurrentmanager']
		department = request.form['department']
		overtime = request.form['overtime']
		businesstravel = request.form['businesstravel']
		stockoption = request.form['stockoption']
		joblevel = request.form['joblevel']
		hourlyrate = request.form['hourlyrate']
		jobinvolvement = request.form['jobinvolvement']
		jobsatisfaction = request.form['jobsatisfaction']
		environmentalsatisfaction = request.form['environmentalsatisfaction']


		# Load template of JSON file containing columns name
		# Schema name
		schema_name = 'models/columns_set.json'
		# Directory where the schema is stored
		schema_dir = os.path.join(current_dir, schema_name)
		with open(schema_dir, 'r') as f:
			cols =  json.loads(f.read())
		schema_cols = cols['data_columns']

		# Parse the categorical columns
		# Column of department
		try:
			col = ('Department_' + str(department))
			if col in schema_cols.keys():
				schema_cols[col] = 1
			else:
				pass
		except:
			pass
		# Column of marital status
		try:
			col = ('MaritalStatus_' + str(marital_status))
			if col in schema_cols.keys():
				schema_cols[col] = 1
			else:
				pass
		except:
			pass
		# Column of Business Travel
		try:
			col = ('BusinessTravel_' + str(businesstravel))
			if col in schema_cols.keys():
				schema_cols[col] = 1
			else:
				pass
		except:
			pass

		# Parse the numerical columns
		schema_cols['Age'] = age
		schema_cols['DistanceFromHome'] = distancefromhome
		schema_cols['NumCompaniesWorked'] = numcompaniesworked
		schema_cols['TotalWorkingYears'] = totalworkingyears
		schema_cols['Gender_Female'] = gender
		schema_cols['YearsAtCompany'] = yearsatcompany
		schema_cols['YearsInCurrentRole'] = yearsincurrentrole
		schema_cols['YearsWithCurrManager'] = yearswithcurrentmanager
		schema_cols['OverTime_No'] = overtime
		schema_cols['StockOptionLevel'] = stockoption
		schema_cols['JobLevel'] = joblevel
		schema_cols['JobInvolvement'] = jobinvolvement
		schema_cols['HourlyRate'] = hourlyrate
		schema_cols['JobSatisfaction'] = jobsatisfaction
		schema_cols['EnvironmentSatisfaction'] = environmentalsatisfaction
		schema_cols['DailyRate'] = float(hourlyrate) * 8
		schema_cols['Education'] = 3
		schema_cols['MonthlyRate'] = (float(hourlyrate) * 2080)/12
		schema_cols['PercentSalaryHike'] = 15.18
		schema_cols['PerformanceRating'] = 3.16
		schema_cols['RelationshipSatisfaction'] = 2.69
		schema_cols['TrainingTimesLastYear'] = 2.9
		schema_cols['WorkLifeBalance'] = 2.76
		schema_cols['YearsSinceLastPromotion'] = 1.8
		schema_cols['EducationField_Human Resources'] = 0.017
		schema_cols['EducationField_Life Sciences'] = 0.411
		schema_cols['EducationField_Marketing'] = 0.099
		schema_cols['EducationField_Medical'] = 0.098
		schema_cols['EducationField_Other'] = 0.044
		schema_cols['EducationField_Technical Degree'] = 0.092
		schema_cols['JobRole_Healthcare Representative'] = 0.068
		schema_cols['JobRole_Laboratory Technician'] = 0.180
		schema_cols['JobRole_Manager'] = 0.065
		schema_cols['JobRole_Manufacturing Director'] = 0.092
		schema_cols['JobRole_Research Director'] = 0.051
		schema_cols['JobRole_Research Scientist'] = 0.245
		schema_cols['JobRole_Sales Representative'] = 0.051

		# Convert the JSON into data frame
		df = pd.DataFrame(
				data = {k: [v] for k, v in schema_cols.items()},
				dtype = float
			)
		
		df = df[['Age', 'DailyRate', 'DistanceFromHome', 'Education',
       'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
       'JobSatisfaction', 'MonthlyRate', 'NumCompaniesWorked',
       'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager',
       'BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently',
       'BusinessTravel_Travel_Rarely', 'Department_Human Resources',
       'Department_Research & Development', 'EducationField_Human Resources',
       'EducationField_Life Sciences', 'EducationField_Marketing',
       'EducationField_Medical', 'EducationField_Other',
       'EducationField_Technical Degree', 'Gender_Female',
       'JobRole_Healthcare Representative', 'JobRole_Laboratory Technician',
       'JobRole_Manager', 'JobRole_Manufacturing Director',
       'JobRole_Research Director', 'JobRole_Research Scientist',
       'JobRole_Sales Representative', 'MaritalStatus_Divorced',
       'MaritalStatus_Married', 'MaritalStatus_Single', 'OverTime_No']].fillna(0)


		# Create a prediction
		print(df.loc[0])
		result = ValuePredictor(data = df)

		# Determine the output
		if int(result) == 1:
			prediction = 'Based on the features provided, this employee is at risk for attrition'
		else:
			prediction = 'Based on the features provided, this employee is not at risk for attrition'

		# Return the prediction
		return render_template('prediction.html', prediction = prediction)
	
	# Something error
	else:
		# Return error
		return render_template('error.html', prediction = prediction)

if __name__ == '__main__':
    app.run(debug = True)