import numpy as np
import pandas as pd
from univariate_datascience.univariate import quanQual, univaiate,freqtable,check_outlier,replace_outlier

dataset = pd.read_csv("../data/Placement.csv")
# 1. Split columns
quan, qual = quanQual(dataset)

#Check empty values
print(dataset.isnull().sum())
#fill empty values with 0 based on requirement
df = dataset["salary"].fillna(0,inplace=True)
print(dataset.isnull().sum())

#eg
dataset["salary"].fillna(dataset["salary"].mean(),inplace=True)
#if multiple columns use imputer funtion
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#imp.fit(dataset[quan])


#noraml Distribution
dataset.isnull().sum()

cate=dataset[qual]
two = [df,cate]
preprocesssed = pd.concat(two,axis=1)
preprocesssed.isnull().sum()
preprocesssed.to_csv("Preplacement.csv",index=0)