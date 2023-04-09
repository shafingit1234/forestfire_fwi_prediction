from flask import Flask , request, jsonify, render_template;
import numpy as np;
import pandas as pd;
import pickle;
from sklearn.preprocessing import StandardScaler;
# to import above listed libraries run following command
# type pip install -r requirements.txt
app = Flask(__name__)

# import ridge regressor model and standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl' , 'rb'));
standard_scaler = pickle.load(open('models/scaler.pkl' , 'rb'));

# router for home page
@app.route('/')
def index():
    return render_template('index.html');

@app.route('/predictdata' , methods = ['GET' , 'POST'])
def predict_datapoint():
    # read all the values in form in my html code,
    # pick the values and predict using the models.
    if(request.method == 'POST'):
        # pass
        # fetch the required features, that will help during prediction
        Temperature = float(request.form.get('Temperature'));
        RH = float(request.form.get('RH'));
        Ws = float(request.form.get('Ws'));
        Rain = float(request.form.get('Rain'));
        FFMC = float(request.form.get('FFMC'));
        DMC = float(request.form.get('DMC'));
        ISI = float(request.form.get('ISI'));
        Classes = float(request.form.get('Classes'));
        Region = float(request.form.get('Region'));
        # perform feature scaling.
        new_data_scaled = standard_scaler.transform([[Temperature , RH , Ws, Rain, FFMC, DMC, ISI, Classes, Region]]);
        result_predicted = ridge_model.predict(new_data_scaled); #return a list having one element
        # display result_predicted in html home page code
        return render_template('home.html' , result = result_predicted[0]);
    else:
        return render_template('home.html')
if __name__=="__main__":
    app.run(host="0.0.0.0")
    # 0.0.0.0 will map to our local ip address
