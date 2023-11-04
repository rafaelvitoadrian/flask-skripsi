from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import sklearn
app = Flask(__name__, template_folder='views')
model = joblib.load("model/jantung_model.pkl")

@app.route('/')
def hello_world():
    # data = (1,1,0,1,1,0.0,2,0,2)
    # data_array = np.asarray(data)
    # data_reshape = data_array.reshape(1, -1)
    # prediction = model.predict(data_reshape)
    # if(prediction[0] == 1):
    #     return 'Heart Disease'
    # else:
    #     return 'No Heart Disease'
    # # return prediction
    return render_template('index.html')

@app.route('/form',methods=['GET'])
def form():
    return render_template('form.html')

@app.route('/submit',methods=['POST'])
def submit_form():
    data = (
        float(request.form['sex']),
        float(request.form['cp']),
        float(request.form['fbs']),
        float(request.form['restecg']),
        float(request.form['exang']),
        float(request.form['oldpeak']),
        float(request.form['slope']),
        float(request.form['ca']),
        float(request.form['thal']),
    )
    data_array = np.asarray(data)
    data_reshape = data_array.reshape(1, -1)
    prediction = model.predict(data_reshape)
    if(prediction[0] == 1):
        return 'Heart Disease'
    else:
        return 'No Heart Disease'

@app.route('/versions')
def versions():
    numpy_version = np.__version__
    sklearn_version = sklearn.__version__
    joblib_version = joblib.__version__
    
    return f"NumPy version: {numpy_version}<br>scikit-learn version: {sklearn_version}<br>joblib version: {joblib_version}" 

if __name__ == '__main__':
    app.run()
