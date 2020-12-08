import tkinter as tk
from tkinter import ttk
import datetime
import numpy as np
# import matplotlib as plt
import pandas_datareader as pdr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

win=tk.Tk()
win.title('Stock Prediction')

# Predict
def pred(stock,test,days,acu,result):
    global dt
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
    acu.config(text=str(algo.score(x_ts,y_ts)))
    prd=algo.predict(x2)
    result.config(text=str(prd))
    dt['forecast']=np.nan
    last_day=dt.iloc[-1].name
    for i in prd:
        last_day=last_day+datetime.timedelta(1)
        dt.loc[last_day]=[np.nan for _ in range(6)]+[i]

# Visualize
def vis():
    dt['adjClose'].plot()
    dt['forecast'].plot()

# Label
stock_label=ttk.Label(win,font=('times',15,'bold'),text='Select Stock:')
stock_label.grid(row=0,column=0,sticky=tk.W)

test_label=ttk.Label(win,font=('times',15,'bold'),text='Select Test Size:')
test_label.grid(row=1,column=0,sticky=tk.W)

days_label=ttk.Label(win,font=('times',15,'bold'),text='No. of Days:')
days_label.grid(row=2,column=0,sticky=tk.W)

acurracy_label=ttk.Label(win,font=('times',15,'bold'),text='Acurracy:')
acurracy_label.grid(row=4,column=0,sticky=tk.W)

result_label=ttk.Label(win,font=('times',15,'bold'),text='Predicion Result:')
result_label.grid(row=5,column=0,sticky=tk.W)

# Combobox
stock_var=tk.StringVar()
stock_options=ttk.Combobox(win,font=('times',15,'bold'),textvariable=stock_var,state='readonly')
stock_options['values']=('googl','msft')
stock_options.current(0)
stock_options.grid(row=0,column=1)

test_var=tk.DoubleVar()
test_options=ttk.Combobox(win,font=('times',15,'bold'),textvariable=test_var,state='readonly')
test_options['values']=(0.2,0.25,0.3)
test_options.current(0)
test_options.grid(row=1,column=1)

# Entry
days_var=tk.IntVar()
days_entry=ttk.Entry(win,font=('times',15,'bold'),width=21,textvariable=days_var)
days_var.set(1)
days_entry.grid(row=2,column=1)

# Button
eval_button=tk.Button(win,font=('times',10,'bold'),text='Evaluate',command=lambda:pred(stock_var.get(),test_var.get(),days_var.get(),acu,result))
eval_button.grid(row=3,column=0)

visual_button=tk.Button(win,font=('times',10,'bold'),text='Visualize',command=vis)
visual_button.grid(row=3,column=1)

# Message
acu=tk.Message(win,font=('times',15,'bold'))
acu.grid(row=4,column=1)

result=tk.Message(win,font=('times',15,'bold'))
result.grid(row=5,column=1)

win.mainloop()