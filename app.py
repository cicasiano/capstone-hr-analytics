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
# SHAP
import shap
from shap.plots._force_matplotlib import draw_additive_plot
import matplotlib.pyplot as plt 

# Current directory
current_dir = os.path.dirname(__file__)

# Flask app
app = Flask(__name__, static_folder = 'static', template_folder = 'template')

# Logging
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

# Functions
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

def shap_explainer(df):
    model_name = 'models/rfc_model.pkl'
    # Directory where the model is stored
    model_dir = os.path.join(current_dir, model_name)
    rfc = loaded_model = joblib.load(open(model_dir, 'rb'))
    rfexplainer = shap.TreeExplainer(rfc)
    rfshap_values = rfexplainer.shap_values(df)
    return rfexplainer,rfshap_values

# Function to sort the list by second item of tuple
def Sort_Tuple(tup): 
  
    # reverse = None (Sorts in Ascending order) 
    # key is set to sort using second element of 
    # sublist lambda has been used 
    return(sorted(tup, key = lambda x: x[1]))  

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

		# create shap values and force_plot
		rfexplainer,rfshap_values = shap_explainer(df)

		# https://www.youtube.com/watch?v=Z2kfLs2Dwqw for user interpetability and less cluttered visualization using the top 10 features impacting the prediction
		shap_user = rfshap_values[1][0,:]
		shap_importance = np.argsort(shap_user)
		neg_indexes = [shap_importance[c] for c in range(10)]
		pos_indexes = [shap_importance[-(c+1)] for c in range(10)]

		main_feats = neg_indexes[:]
		main_feats.extend(pos_indexes[:])

		feature_names = [list(df.columns)[_] for _ in main_feats]
		neg_vals = [shap_user[shap_importance[c]] for c in range(10)]
		pos_vals = [shap_user[shap_importance[-(c+1)]] for c in range(10)]

		main_vals = neg_vals
		main_vals.extend(pos_vals)

		main_sorted = Sort_Tuple(list(zip(feature_names,main_vals)))
		main_sorted = [t[0] for t in main_sorted]

		# Determine the output
		if int(result) == 1:
			prediction = 'Based on the features provided, this employee is at risk of attrition.'
			features   = main_sorted[-1]+' and '+ main_sorted[-2] + ' are contributing towards this employee leaving.'
		else:
			prediction = 'Based on the features provided, this employee is not at risk of attrition.'
			features   = str(main_sorted[0])+' and '+ str(main_sorted[1] )+ ' are contributing towards this employee staying.'
		# https://towardsdatascience.com/tutorial-on-displaying-shap-force-plots-in-python-html-4883aeb0ee7c
		def _force_plot_html(rfexplainer, rfshap_values,main_feats,feature_names):
			force_plot = shap.force_plot(rfexplainer.expected_value[1],
			rfshap_values[1][0,main_feats], df[feature_names], feature_names = feature_names, matplotlib=False, link='logit')
			shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
			return shap_html
		shap_plots = {}
		shap_plots[1] = _force_plot_html(rfexplainer, rfshap_values,main_feats,feature_names)

		# Return the prediction
		return render_template('prediction.html', prediction = prediction, shap_plots = shap_plots, features = features)
	
	# Something error
	else:
		# Return error
		return render_template('error.html', prediction = prediction, features = features)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)