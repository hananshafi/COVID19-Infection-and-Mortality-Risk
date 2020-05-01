# COVID19-Infection-and-Mortality-Risk
This repository contains models and code for COVID-19 infection risk and Mortality. The data set is obtained from following link:

https://www.covid19survivalcalculator.com/download

The primary goal of this project is to predict the probability with which a person might contract COVID-19 along with his / her mortality probability. The infection risk score can give an idea of whether a person is taking proper precautions or not. The mortality score can give an idea to medical professionals as to which patient might need a ventillator.

The model is trained taking into account following factors:

       ['sex', 'age', 'height', 'weight', 'blood_type',
       'smoking', 'alcohol', 'cannabis', 'amphetamines', 'cocaine', 'lsd',
       'mdma', 'contacts_count', 'house_count',
       'rate_government_action', 'rate_reducing_risk_single',
       'rate_reducing_risk_house', 'rate_reducing_mask',
       'covid19_symptoms', 'covid19_contact', 'asthma', 'kidney_disease',
       'compromised_immune', 'heart_disease', 'lung_disease', 'diabetes',
       'hiv_positive', 'hypertension', 'other_chronic',
       'risk_infection', 'risk_mortality']

# Model and Usage

The model_severity.ipynb notebook contains the code for preprocessing and training. The models folder contains the saved models and scalers. The app.py file contains code for hosting.

Here is the model Workflow:
![Model Workflow](https://github.com/hananshafi/COVID19-Infection-and-Mortality-Risk/tree/master/assets/workflow.JPG)

