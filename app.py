import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
ca_knn = joblib.load('./static/ca_knn.joblib')
pso_knn = joblib.load('./static/pso_knn.joblib')
ca_etr = joblib.load('./static/ca_etr.joblib')
pso_etr = joblib.load('./static/pso_etr.joblib')
ca_br = joblib.load('./static/ca_br.joblib')
pso_br = joblib.load('./static/pso_br.joblib')
ca_ar = joblib.load('./static/ca_ar.joblib')
pso_ar = joblib.load('./static/pso_ar.joblib')
ca_xgb = joblib.load('./static/ca_xgb.joblib')
pso_xgb = joblib.load('./static/pso_xgb.joblib')
ca_lr = joblib.load('./static/ca_lr.joblib')
pso_lr = joblib.load('./static/pso_lr.joblib')
ca_rf = joblib.load('./static/ca_rf.joblib')
pso_rf = joblib.load('./static/pso_rf.joblib')
ca_gbr= joblib.load('./static/ca_gbr.joblib')
pso_gbr = joblib.load('./static/pso_gbr.joblib')
ca_dtr= joblib.load('./static/ca_dtr.joblib')
pso_dtr = joblib.load('./static/pso_dtr.joblib')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features=[]
    for x in request.form.values():
        try:
            int_features.append(int(float(x)))
        except ValueError:
            # Handle the case where conversion is not possible
            print(f"Skipping invalid value: {x}")

    print(int_features)
    mo= int_features[0]
    prediction=[]
    output=0
    if mo==1:
        int_features= int_features[1:]
        final_features = np.array(int_features)
        prediction = ca_knn.predict([final_features])

    elif mo==2:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = pso_knn.predict(final_features)
    elif mo==5:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = ca_etr.predict(final_features)
    elif mo==6:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = pso_etr.predict(final_features)
    elif mo==9:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = ca_br.predict(final_features)
    elif mo==10:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = pso_br.predict(final_features)
    elif mo==13:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = ca_ar.predict(final_features)
    elif mo==14:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = pso_ar.predict(final_features)
    elif mo==17:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = ca_xgb.predict(final_features)
    elif mo==18:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = pso_xgb.predict(final_features)
    elif mo==21:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = ca_lr.predict(final_features)
    elif mo==22:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = pso_lr.predict(final_features)
    elif mo==25:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = ca_rf.predict(final_features)
    elif mo==26:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = pso_rf.predict(final_features)
    elif mo==29:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = ca_gbr.predict(final_features)
    elif mo==30:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = pso_gbr.predict(final_features)
    elif mo==33:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = ca_dtr.predict(final_features)
    elif mo==34:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = pso_dtr.predict(final_features)
   
    output = round(prediction[0], 2)
    print(output)

    return render_template('temp.html', prediction_text='The predicted critical shear stress of bed channel is {}N/m2'.format(output))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)
