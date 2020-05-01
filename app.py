import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
from keras.models import load_model
import os
import tensorflow as tf

app = Flask(__name__,template_folder='templates',static_folder='templates/css')

model_regr = joblib.load('models_encoders/regr.pkl')
model_nn = tf.keras.models.load_model('D:/covid-chestxray-dataset-master/models_encoders/model_severity_4.h5')
graph = tf.get_default_graph()


standard_scaler_x = joblib.load('models_encoders/standard_scaler_x.pkl')
standard_scaler_y = joblib.load('models_encoders/standard_scaler_y.pkl')

minmax_x = joblib.load('models_encoders/minmax_x.pkl')
minmax_y = joblib.load('models_encoders/minmax_y.pkl')


#os.environ['CUDA_VISIBLE_DEVICES']=""


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_infection', methods = ['POST']) 
def predict_infection(): 
    try:
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(to_predict_dict.values()) 
        to_predict_list = list(map(int, to_predict_list))
        

        
        #risk_infection,risk_mortality = predict(to_predict_list)   
        risk = predict(to_predict_list)
        return render_template("result.html", Infection_Risk = "Risk of getting exposed to COVID-19= {:0.2f}%".format(risk[0][0]), Mortality_Risk = "Risk of mortality due to COVID-19= {:0.2f}%".format(risk[0][1]) ) 
    
    except:
        print("An exception occurred.")
    
def predict(input_list): 
    print(input_list)
    
    #predict via neural network model_nn (uncomment the below lines)
    
    #minmax_scaled_input = minmax_x.transform(np.array(input_list).reshape(1,29))
    #print(minmax_scaled_input)
    #print(model_nn.summary())
    #global graph
    #with graph.as_default():
    #    result_nn = model_nn.predict(minmax_scaled_input)
    #result_nn_ = minmax_y.inverse_transform(result_nn)
    
    #predict via Random Forest Regressor
    standard_scaled_input = standard_scaler_x.transform(np.array(input_list).reshape(1,29))
    result_regr = model_regr.predict(standard_scaled_input)
    result_regr_ = standard_scaler_y.inverse_transform(result_regr)
    print(result_regr_)
    
    #Find mean of predictions if using both models
    
    #risk_infection = np.mean(np.array([result_regr[0],result_nn[0]]))
    #risk_mortality = np.mean(np.array([result_regr[1],result_nn[1]]))
    #return risk_infection, risk_mortality
    
    return result_regr_


if __name__ == "__main__":
    app.run()
        
    