"""
Kaggle Dataset: https://www.kaggle.com/datasets/syuzai/perth-house-prices

"""

import os

# Part of code that installs dependencies
# os.system('pip3 freeze > requirements.txt')
# os.system('pip3 install -r requirements.txt')


#Importing all the necessary libraries
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import tree
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score


import warnings
warnings.filterwarnings('ignore')
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

df = pd.read_csv(r"all_perth_310121.csv")

#Look our data's form.
print (df.head)

#Get all column names. Before we clean the data.
print(df.columns.tolist())

#Drop all the columns we are not going to use for the excecise. 
df.drop(columns= ['ADDRESS','SUBURB','NEAREST_STN','DATE_SOLD','POSTCODE'
                  ,'LATITUDE','NEAREST_SCH','NEAREST_SCH_DIST','NEAREST_SCH_RANK'], inplace=True )

#Get all column names once again.
listofnames = df.columns.tolist()
print("LIST OF OUR COLUMNS:",type(listofnames),listofnames)

#Check the type of our data and the existence of null variables.
print(df.info())

#Get rid of null values.
df.dropna(inplace = True)
print(df.info())

#Convert all our data into int32.
df=df.astype('int32')
print(df.info())

#Time to build a model from our dataset.
x_var = df.drop(columns=['PRICE'])
y_var = df['PRICE']

#Check the correct assignment of our variables.
print ("Original\n",df.head(),"x_var:\n",x_var.head(),'y_var: \n',y_var.head())

#Train test split.
x_train, x_test, y_train, y_test = train_test_split( 
    x_var, y_var, test_size=0.25, random_state=777)

#Create the Multivariate Regression Model
multivarmodel = LinearRegression()

#Train on our x_test, y_test
multivarmodel.fit(x_train,y_train)

#Predict on the test x_test data.
x_test_pred = multivarmodel.predict(x_test)

#Check the Accuracy.
m_sq_err = mean_squared_error(y_test,x_test_pred)
m_abs_err = mean_absolute_error(y_test,x_test_pred)

print(f'Mean squared error Tou Multivariable Montelou mas: {m_sq_err}')
print(f'Mean absolute error Tou Multivariable Montelou mas: {m_abs_err}')


df["ValuePerFLOOR_AREA"] =df["PRICE"]/df["FLOOR_AREA"]
df["ESTIMATION"] = "0"
print("Aksia ana tetragoniko: \n", df["ValuePerFLOOR_AREA"])

ValuePerFLOOR_AREA_STD = df["ValuePerFLOOR_AREA"].std()
ValuePerFLOOR_AREA_MEAN = df["ValuePerFLOOR_AREA"].mean()
df["ESTIMATION"].loc[df["ValuePerFLOOR_AREA"]>=(ValuePerFLOOR_AREA_MEAN+ValuePerFLOOR_AREA_STD)] ="EXPENSIVE"
df["ESTIMATION"].loc[(df["ValuePerFLOOR_AREA"]<=(ValuePerFLOOR_AREA_MEAN+ValuePerFLOOR_AREA_STD))& (df["ValuePerFLOOR_AREA"]>=(ValuePerFLOOR_AREA_MEAN-ValuePerFLOOR_AREA_STD))] ="GOOD PRICE"
df["ESTIMATION"].loc[df["ValuePerFLOOR_AREA"]<=(ValuePerFLOOR_AREA_MEAN-ValuePerFLOOR_AREA_STD)] ="CHANCE"
print("ESTIMATION:",df["ESTIMATION"])

x_var2 = df.drop(columns=["ESTIMATION","ValuePerFLOOR_AREA"])
y_var2 = df["ESTIMATION"]

#Train test split.
x_train2, x_test2, y_train2, y_test2 = train_test_split( 
    x_var2, y_var2, test_size=0.25, random_state=777)

#Train the Classifier
treeclf = tree.DecisionTreeClassifier()
treeclf = treeclf.fit(x_train2, y_train2)

#Predict on the test x_test data.
x_test_pred2 = treeclf.predict(x_test2)

#Check the Accuracy.
print("Accuracy tou TREE CLASSIFIER:",accuracy_score(y_test2, x_test_pred2))


def sim_print (ourdf):
    if ourdf.empty:
        print("Our Data is empty.")
    else:
        print(ourdf)


def newinput(listofcols):
    #A function that takes the name of the column returns a new input in dict form
    tempdict ={}
    for name in listofcols:
        print (f"Give an int value for {name}: ")
        userinput = int(input())
        tempdict[name] = userinput
    print ("The function will return this Dictionary:", tempdict,type(tempdict))
    return tempdict

#Statistic, Average, Median, Standard Deviation
def calcstats(curdf):
    #A function that takes the name of our df and finds Average, Media and Standard Deviation for 1 column or all and is safy for wrong input.
    print("Give the number of the column you wish to see the statistics. As follows:")
    i=0
    colnames = curdf.columns.tolist()
    for col in colnames:
        print(f"{i}){col}")
        i+=1
    print("Give -1 for statistics on all columns.")
    userinput= int(input())
    if 0 <= userinput <= i:
        ourdf_med = curdf[colnames[userinput]].median()
        ourdf_aver = curdf[colnames[userinput]].mean()
        ourdf_std = curdf[colnames[userinput]].std()
        print (f"The median : {ourdf_med}")
        print (f"The average : {ourdf_aver}")
        print (f"The standard deviation: {ourdf_std}") 

    elif userinput == -1:
        ourdf_med = curdf.median()
        ourdf_aver = curdf.mean()
        ourdf_std = curdf.std()
        print (f"The median: {ourdf_med}")
        print (f"The average: {ourdf_aver}")
        print (f"The standard deviation: {ourdf_std}")
    else:
        print("Number is out of bounds! Try again maybe?")

user_df = pd.DataFrame()
pred_df =pd.DataFrame()
final_df = pd.DataFrame()
while True:
    user_dict = {}
    print("|~~~~MENU~~~~|")
    print("Press 1 for printing the current data.")
    print("Press 2 for adding a new House to the current data.")
    print("Press 3 for statistics on the current data.")
    print("Press 4 for prediction on the given data.")
    print("Press 5 for estimation on the appartment.")
    print("Press anythin else to exit.")
    userinput = int(input("Your input is: "))

    if userinput == 1:
        sim_print(user_df)

    elif userinput == 2:
        #Give the list of columns of x_var dataframe, to take new inputs.
        print("Okey new input-> \n")
        user_dict = newinput(listofnames)
        temp_df = pd.DataFrame([user_dict])
        user_df = pd.concat ([user_df,temp_df],ignore_index=True)
        user_df.reset_index(drop = True, inplace= True)
        print("Your current data is:\n",user_df)

    elif userinput == 3:
        calcstats(user_df)

    elif userinput == 4:
        
        pred_df = user_df.drop(columns=["PRICE"])
        pred_df.insert(loc = 0, column='PPRICE',value = multivarmodel.predict(user_df.drop(columns=["PRICE"])),allow_duplicates= True)
        
        pred_df.reset_index(drop = True, inplace= True)
        print("Your predicted prices are:\n",pred_df)
    elif userinput == 5:

        clf_pred = pred_df.drop(columns= ["PPRICE"])
        clf_pred["ESTIMATION"] = treeclf.predict(pred_df.rename(columns = {"PPRICE":"PRICE"}))
        print("The predicted appartment ESTIMATION:",clf_pred)
    else:
        break

print("Good Bye !!!")





