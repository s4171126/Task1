#This is my Python Code for Case Studies in Data Science - Individual Task 1 
#This code is for a decision tree classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

myData=pd.read_csv('student_performance_dataset.csv')
print(myData.head())