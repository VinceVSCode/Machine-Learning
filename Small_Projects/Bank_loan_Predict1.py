import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import os 

""" 
Dataset:https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset
"""
#Importing the data from our .csv file

df = pd.read_csv(r'bank.csv')

#Data cleaning 
print ("OUR DATAFRAME'S INFO",df.isnull().sum())
#No null values so we procceed to select only our values columns of interest

cols = list(df.columns)

#We need arithmetic values for our Classifier so we will encode all our data
le =LabelEncoder()

for col in cols:
    df[col] = le.fit_transform(df[col])
df.info()

print ("Here df columns \n",cols)
#Drop column that are of no interest to us
x = df.drop (columns =['default', "contact","day","month","campaign","loan",
                       "duration","pdays","previous",])
x.info()

y = df["loan"]


#We decide to use a decision tree as the ML model.
model = DecisionTreeClassifier()

#We will use 70/30 ratio 

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

#Train model 
model.fit(x_train,y_train)
print ("Model is trained and ready to classify entries.")

person_data = dict()

for col in list(x.columns):
    print ("Give ",col)
    person_data[col]=input()
person_df = pd.DataFrame(person_data, index=[0])
print ("Your info is :",person_df)

for col in (person_df.columns):
    person_df[col] = le.fit_transform(person_df[col])
decision = model.predict(person_df)
if decision[0] == 0: 
    print ("Loan is not approved. ",decision)
else:
    print ("Loan is approved",decision," to the ammount :",0.15*int(person_df["deposit"]))    
