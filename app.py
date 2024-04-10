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
iwo_knn = joblib.load('./static/iwo_knn.joblib')
gbo_knn = joblib.load('./static/gba_knn.joblib')
ca_etr = joblib.load('./static/ca_etr.joblib')
pso_etr = joblib.load('./static/pso_etr.joblib')
iwo_etr = joblib.load('./static/iwo_etr.joblib')
gbo_etr = joblib.load('./static/gba_etr.joblib')
ca_br = joblib.load('./static/ca_br.joblib')
pso_br = joblib.load('./static/pso_br.joblib')
iwo_br = joblib.load('./static/iwo_br.joblib')
gbo_br = joblib.load('./static/gba_br.joblib')
ca_ar = joblib.load('./static/ca_ar.joblib')
pso_ar = joblib.load('./static/pso_ar.joblib')
iwo_ar = joblib.load('./static/iwo_ar.joblib')
gbo_ar = joblib.load('./static/gba_ar.joblib')
ca_xgb = joblib.load('./static/ca_xgb.joblib')
pso_xgb = joblib.load('./static/pso_xgb.joblib')
iwo_xgb = joblib.load('./static/iwo_xgb.joblib')
gbo_xgb = joblib.load('./static/gba_xgb.joblib')
ca_lr = joblib.load('./static/ca_lr.joblib')
pso_lr = joblib.load('./static/pso_lr.joblib')
iwo_lr = joblib.load('./static/iwo_lr.joblib')
gbo_lr = joblib.load('./static/gba_lr.joblib')
ca_rf = joblib.load('./static/ca_rf.joblib')
pso_rf = joblib.load('./static/pso_rf.joblib')
iwo_rf = joblib.load('./static/iwo_rf.joblib')
gbo_rf = joblib.load('./static/gba_rf.joblib')
ca_gbr= joblib.load('./static/ca_gbr.joblib')
pso_gbr = joblib.load('./static/pso_gbr.joblib')
iwo_gbr = joblib.load('./static/iwo_gbr.joblib')
gbo_gbr = joblib.load('./static/gba_gbr.joblib')
ca_dtr= joblib.load('./static/ca_dtr.joblib')
pso_dtr = joblib.load('./static/pso_dtr.joblib')
iwo_dtr = joblib.load('./static/iwo_dtr.joblib')
gbo_dtr = joblib.load('./static/gba_dtr.joblib')
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
    elif mo==3:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = iwo_knn.predict(final_features)
    elif mo==4:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = gbo_knn.predict(final_features)
    elif mo==5:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = ca_etr.predict(final_features)
    elif mo==6:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = pso_etr.predict(final_features)
    elif mo==7:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = iwo_etr.predict(final_features)
    elif mo==8:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = gbo_etr.predict(final_features)
    elif mo==9:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = ca_br.predict(final_features)
    elif mo==10:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = pso_br.predict(final_features)
    elif mo==11:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = iwo_br.predict(final_features)
    elif mo==12:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = gbo_br.predict(final_features)
    elif mo==13:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = ca_ar.predict(final_features)
    elif mo==14:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = pso_ar.predict(final_features)
    elif mo==15:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = iwo_ar.predict(final_features)
    elif mo==16:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = gbo_ar.predict(final_features)
    elif mo==17:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = ca_xgb.predict(final_features)
    elif mo==18:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = pso_xgb.predict(final_features)
    elif mo==19:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = iwo_xgb.predict(final_features)
    elif mo==20:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = gbo_xgb.predict(final_features)
    elif mo==21:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = ca_lr.predict(final_features)
    elif mo==22:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = pso_lr.predict(final_features)
    elif mo==23:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = iwo_lr.predict(final_features)
    elif mo==24:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = gbo_lr.predict(final_features)
    elif mo==25:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = ca_rf.predict(final_features)
    elif mo==26:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = pso_rf.predict(final_features)
    elif mo==27:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = iwo_rf.predict(final_features)
    elif mo==28:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = gbo_rf.predict(final_features)
    elif mo==29:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = ca_gbr.predict(final_features)
    elif mo==30:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = pso_gbr.predict(final_features)
    elif mo==31:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = iwo_gbr.predict(final_features)
    elif mo==32:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = gbo_gbr.predict(final_features)
    elif mo==33:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = ca_dtr.predict(final_features)
    elif mo==34:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = pso_dtr.predict(final_features)
    elif mo==35:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = iwo_dtr.predict(final_features)
    elif mo==36:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = gbo_dtr.predict(final_features)
   
    output = round(prediction[0], 2)
    print(output)

    return render_template('temp.html', prediction_text='The predicted critical shear stress of bed channel is {}N/m2'.format(output))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)
