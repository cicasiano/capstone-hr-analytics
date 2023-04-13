# Attrition Predictor Web App

Capstone project for University of Michigan's Master of Applied Data Science program by Harry Lin and Carolann Decasiano. This project uses data released by IBM for HR Analytics Employee Attrition & performance.

# Getting Started
## Clone the repo
Clone this repository to get started.
```
git clone https://github.com/cicasiano/capstone-hr-analytics.git
```

## Prereqs
Get all of the dependencies needed.
```
pip install -r requirements.txt
```

## SHAP Installation
SHAP is a major part of this project as it is used to evaluate and relay to end users the forces of our model's prediction. Installation of SHAP should be taken care of if you used the `requirements.txt`. To have the shap functions and plots run locally you'll need the complete shap library. 

```
pip install shap
```
For more information on SHAP, you can read the [documentation] (https://shap.readthedocs.io/en/latest/index.html) for installation and [Interpretable Machine Learning] (https://christophm.github.io/interpretable-ml-book/shapley.html#shapley) by Christoph Molnar for a thorough explaination of shapley/shap values.

## Flask Installation
Also should be taken care of if you used the `requirements.txt`. Flask is a lightweight WSGI web application framework we utilized to build our predictor app.

```
pip install Flask
```
For a more installation details you can read the installation [document](https://flask.palletsprojects.com/en/2.2.x/installation/#install-flask).

## AWS Container Services
We utilized AWS services to deploy our application. Other services are available but if this is your first time using AWS services we recommend this git repo for deploying a Flask Docker Image to an AWS container service.

https://github.com/vastevenson/docker-flask-demo-aws-ecs-and-ecr.git

You will need to have [Docker](https://docs.docker.com/get-docker/) and [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) installed.

### Run locally
Once you have the repo cloned and requirements installed, navigate to the root directory and start it up locally.

```
flask --app app run
```
To run in debug mode:
```
flask --app app run --debugger
```

Then go to [http://localhost:5000/]([http://localhost:5000/]) to see the app! 

## Attrition Pedictor Web App
Feel free to take a look at our already host web app.

[Employee Attrition Predictor](http://54.196.189.236:5000/)

## End to End ML Model Deployment
Below is a helper visual to get an idea of the development and deployment of our ML model.
![end to end ML](https://github.com/cicasiano/capstone-hr-analytics/blob/3b20cb0b601359afa36920af3bebffb06049edbf/end-to-end-ml.png?raw=true)
