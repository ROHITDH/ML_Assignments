#assignment_2.1 ML 25/Aug/2020 RDH

#QN:1.Predict canada's per capita income in year 2020. 
#	find "canada_per_capita_income.csv" file.in material section. 
#	Using this build a regression model and predict the per capital
# 	income of canadian citizens in year 2020.

#importing packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Linear regression model
model = LinearRegression()

#Reading and printing data from a file
a = pd.read_csv("res/canada_per_capita_income.csv")
print(a)

#y is dependent and x is independent
y=a.income
x=a[['year']] # 2 dimesional 

#plooting graph
plt.xlabel('income')
plt.ylabel('income')
plt.scatter(x,y)


#fitting values to the model
model.fit(x,y)

#predicting income corrected to 6 decimal points
print("Income for year = 2020 is",'%.6f'% model.predict([[2020]])[0])