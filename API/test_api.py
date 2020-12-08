import requests
import json

URL='http://mic.pythonanywhere.com/{}/{}/{}'

stock=input('Enter Stock Name:')
test_size=input('Enter Test Size:')
days=input('Enter No. of Days:')

URL=URL.format(stock,test_size,days)
r=requests.get(url=URL)

data=r.json()
print('Prediction: ',data)
input()
