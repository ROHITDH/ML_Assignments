#assignment_4 ML 27/Aug/2020 RDH

#QN:1.there is an Exercise folder that contains 
#	carprices.csv. This file has car sell prices for 
#	3 different models. First plot data points on a 
#	scatter plot chart to see if linear regression model
#	can be applied. If yes, then build a model that can 
#	answer following questions

#1) Predict price of a mercedez benz that is 4 yr old with mileage 45000
#2) Predict price of a BMW X5 that is 7 yr old with mileage 86000
#3) check accuracy of your model


#importing packages related to ML
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


#importing package pickle to save model
import pickle


#Linear regression model
model = LinearRegression()


#Reading and printing data from a file
df = pd.read_csv("res/carprices.csv")
print(df,"\n\n")


#Using pandas to create dummy variables
dummies = pd.get_dummies(df['Car Model'])


#adding dummies to data frame and ct=reating new dataframe
df_dummies= pd.concat([df,dummies],axis='columns')

#---------------------Linearity Check---------------------------------------
#Dummy variables meet the assumption of linearity by definition
#plotting the values of df
plt.scatter(df_dummies['Mileage'],df_dummies['Sell Price($)'])
plt.xlabel('Mileage', fontsize=18)
plt.ylabel('Sell Price($)', fontsize=18)
plt.show()


LinearityFlag = 1 #flag for linearoty track
#checking whether the given data are linear or not using corelation
for i in range(3):
    if(df.corr()['Sell Price($)'][i] < -1.8 or df.corr()['Sell Price($)'][i] > 1.8 ):
    	LinearityFlag = 0
    	print("The given data are not Linearly Scattered\n")
    	break;


#-------------------Creation of Model and Prediction-----------------------
if LinearityFlag :
	print("The given data are Linearly Scattered, It follows LinearRegression\n")


	#deleting character content row from dataframe(copy)
	#	one of the dummy column related to 'Car Model' to minimise size
	del df_dummies['Car Model']
	del df_dummies['Mercedez Benz C class']


	#Dummy Variable Trap
	#y is dependent and x is independent
	y = df_dummies['Sell Price($)']

	del df_dummies['Sell Price($)']
	x = df_dummies


	#fitting values to the model
	model.fit(x,y)

	#saving the model 
	with open('model_carprices','wb') as file:
		pickle.dump(model,file)


	#load the saved model
	with open('model_carprices','rb') as file:
		model = pickle.load(file)

	#1	
	print("price of a mercedez benz that is 4 yr old with mileage 45000: ")
	print(round(model.predict([[45000, 4, 0, 0]])[0]),'\n')
	#2
	print("price of a BMW X5 that is 7 yr old with mileage 86000")
	print(round(model.predict([[86000, 7, 0, 1]])[0]),'\n\n')


#----------------Test and train of model for accuracy--------------------
# returns coefficient of determination R^2 
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
	tts = LinearRegression()
	tts.fit(x_train, y_train)
	print("Accuracy of the model is : ",tts.score(x_test, y_test),'\n')
