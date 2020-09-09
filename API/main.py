from flask import Flask, jsonify
import datetime
import numpy as np
import pandas_datareader as pdr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from flask import Flask
app = Flask(__name__)

# Predict
@app.route('/<string:stock>/<float:test>/<int:days>')
def pred(stock,test,days):
    dt=pdr.DataReader(stock,'tiingo','2000-1-1',datetime.datetime.now(),api_key='911ee28d70118f9cea5a84d2b8f1436fa32d3116')
    dt.reset_index(inplace=True)
    dt.set_index("date",inplace=True)
    dt=dt[['adjClose','adjHigh','adjLow','adjOpen','adjVolume']]
    no_days=int(days)
    dt['new_close']=dt['adjClose'].shift(-no_days)
    x=dt.drop(['adjClose','new_close'],axis=1)
    y=dt['new_close'].dropna()
    x1=x[:-no_days]
    x2=x[-no_days:]
    scaler=StandardScaler()
    scaler.fit(x1)
    x1=scaler.transform(x1)
    x2=scaler.transform(x2)
    x_tr,x_ts,y_tr,y_ts=train_test_split(x1,y,test_size=0.25)
    algo=LinearRegression()
    algo.fit(x_tr,y_tr)
    acu=algo.score(x_ts,y_ts)
    prd=algo.predict(x2)
    result={'stock':stock,'test_size':test,'no_of_days':days,'accuracy':acu,'prediction':list(prd)}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
