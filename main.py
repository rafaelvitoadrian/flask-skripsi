from flask import Flask, render_template, request, redirect, url_for,jsonify
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

@app.route('/submits', methods=['POST'])
def submit():
    if request.method == 'POST':
        input_value = request.form['inputValue']
        # Lakukan sesuatu dengan nilai yang diterima, misalnya simpan ke database
        return jsonify(result=f'Nilai yang dimasukkan: {input_value}')

@app.route('/about',methods=['GET'])
def about():
    return render_template('about.html')

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
    result = ""
    if(prediction[0] == 1):
        result = 1
    else:
        result = 0
    
    return jsonify(result=result)

# @app.route('/submit',methods=['POST'])
# def submit_form():
#     return jsonify(result=f"Nilai yang dimasukkan: {request.form['sex']}, {request.form['cp']},  {request.form['fbs']} ,{request.form['restecg']}, {request.form['exang']}, {request.form['slope']} ,{request.form['ca']}, {request.form['thal']}{request.form['oldpeak']}")

@app.route('/versions')
def versions():
    numpy_version = np.__version__
    sklearn_version = sklearn.__version__
    joblib_version = joblib.__version__
    
    return f"NumPy version: {numpy_version}<br>scikit-learn version: {sklearn_version}<br>joblib version: {joblib_version}" 

if __name__ == '__main__':
    app.run(debug=True)
