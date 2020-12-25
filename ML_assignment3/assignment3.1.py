#assignment_3.1 ML 26/Aug/2020 RDH

#QN:1.In classroom there is hiring.csv. 
#	This file contains hiring statics for 
#	a firm such as experience of candidate,
#	his written test score and personal interview score. 
#	Based on these 3 factors, HR will decide the salary.
#	Given this data, you need to build a machine learning 
#	model for HR department that can help them decide 
#	salaries for future candidates. Using this predict
#	salaries for following candidates,

#2 yr experience, 9 test score, 6 interview score

#12 yr experience, 10 test score, 10 interview score

#using word2number package to convert numbers in letter formet to integer formet
#pip install word2number
from word2number import w2n

#importing packages related to ML
import pandas as pd
from sklearn.linear_model import LinearRegression

#to ignore the pandas copy warnings
import warnings
warnings.filterwarnings("ignore")

#Linear regression model
model = LinearRegression()

#Reading and printing data from a file
a = pd.read_csv("res/hiring.csv")
print(a)

#getting the total no.of rows 
rows = a.shape[0]

#changing all the available numbers in words in 1st column
#	to interger values. If not exist thats NaN
for i in range(rows):
	try:
		a.experience[i]=w2n.word_to_num(a['experience'][i])
	except:
		pass

#Data Preprocessing: Fill NA values with median value of a column
a['test_score(out of 10)'] = a['test_score(out of 10)'].fillna(a['test_score(out of 10)'].median())
a.experience = a.experience.fillna(a.experience.median())

#y is dependent and x is independent
y=a['salary($)']
del a['salary($)']
x=a

#fitting values to the model
model.fit(x,y)

#predicting the salaries
#1
print("salary of person with 2 yr experience, 9 test score, 6 interview score: ")
print('$',round(int(model.predict([[2, 9, 6]])[0])))
#2
print("salary of person with 12 yr experience, 10 test score, 10 interview score: ")
print('$',round((model.predict([[12, 10, 10]])[0])))
