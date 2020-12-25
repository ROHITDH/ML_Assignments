#assignment_2.2 ML 25/Aug/2020 RDH

#QN:2.Predict weight of person if his height is 1.61.  
#	find dat.csv file.in material section. Using this 
#	build a regression model and predict the  weight of
#	person if his height is 1.61

#importing packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Linear regression model
model = LinearRegression()

#Reading and printing data from a file
a = pd.read_csv("res/data.csv")
print(a)

#y is dependent and x is independent
y=a.Weight
x=a[['Height']] # 2 dimesional 

#plooting graph
plt.xlabel('Height')
plt.ylabel('Weight')
plt.scatter(x,y)

#fitting values to the model
model.fit(x,y)

#predicting weight corrected to 2 decimal points
print("Weight for Height = 1.61 is",'%.2f'% model.predict([[1.61]])[0])