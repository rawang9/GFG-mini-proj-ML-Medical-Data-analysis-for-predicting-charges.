import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
data= pd.read_csv("medical_cost_data.csv")
print(data.head(10))
gender={"male":1,"female":0}
smoke={"yes":1,"no":0}
data["sex"]=data["sex"].map(gender)
data["smoker"]=data["smoker"].map(smoke)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#region
le.fit(data.region.drop_duplicates())
data.region = le.transform(data.region)
print(data.head())
print(data.corr()["charges"].sort_values())
f,ax=plt.subplots(figsize=(10,8))
sea.heatmap(data.corr(),ax=ax,annot=True)
#Smoker analyses
sea.factorplot(x="smoker",kind="count",data=data,hue="sex")
#BMI Analysis
plt.figure(figsize=(12,5))
plt.title("BMI Distribution")
sea.distplot(data["bmi"])
#The average BMI in patients is 30.
#Children Count Analysis
sea.factorplot(x=data["children"],kind="count",data=data)
#plt.show()
x=data.drop(data[["charges","region"]],axis=1)
y=data.charges
from sklearn.model_selection import train_test_split
#spliting
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.30,random_state=0)
#scaliing data
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)
print(xtrain[0:2])
#traing model used
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
#variable
lr=LinearRegression()
rfr=RandomForestRegressor()
dtr=DecisionTreeRegressor()
svr=svm.SVR()
#training
lr.fit(xtrain,ytrain)
rfr.fit(xtrain,ytrain)
dtr.fit(xtrain,ytrain)
svr.fit(xtrain,ytrain)
#predicting
y_pred_linear = lr.predict(xtest)
y_pred_dt = dtr.predict(xtest)
y_pred_svr = svr.predict(xtest)
y_pred_rf = rfr.predict(xtest)
#checking score
from sklearn.metrics import mean_squared_error
import math
import math
error_linear = math.sqrt(mean_squared_error((y_pred_linear), ytest))
error_dt = math.sqrt(mean_squared_error(y_pred_dt, ytest))
error_svr = math.sqrt(mean_squared_error(y_pred_svr, ytest))
error_rf = math.sqrt(mean_squared_error(y_pred_rf, ytest))

print ("    Model           :     RMSE Error\n" )
print ("Linear Regression   : ", error_linear)
print ("Decision Tree       : ", error_dt)
print ("Support Vector      : ", error_svr)
print ("Random Forest       : ", error_rf)