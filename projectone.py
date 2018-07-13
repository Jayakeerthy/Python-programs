#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[: , :-1].values
y = dataset.iloc[:, 3].values
from sklearn.preprocessing import Imputer  #imported the class imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0) #create an object of the class
imputer = imputer.fit(X[:,1:3])# fit the "imputer" object to matrix of feature X
#replacing the missing names with the mean of the columns
X[:,1:3] = imputer.transform(X[:,1:3])
#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #imported the class LabelEncoder and OneHotEncoder for transforming categoical values from 0,1,2 to 3 col binary code
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
#i think i learned git !!!