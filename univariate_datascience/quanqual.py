import pandas as pd
import numpy as np

dataset = pd.read_csv("../data/Placement.csv")
#print(dataset.info)
#print(dataset.dtypes)
quan = []
qual = []

for columnName in dataset.columns:
    if dataset[columnName].dtype == 'object':
        qual.append(columnName)
    else:
        quan.append(columnName)
#Quan,Qual
def quanQual(dataset):
    quan = []
    qual = []

    for columnName in dataset.columns:
        if dataset[columnName].dtype == 'object':
            qual.append(columnName)
        else:
            quan.append(columnName)
    return quan,qual

#Central tendency(mean,median,mode)
print(dataset['ssc_p'].mean())
print(dataset['ssc_p'].median())
print(dataset['ssc_p'].mode()[0])

# it return mean,std,min,max details
dataset.describe()

#Create table
descriptive = pd.DataFrame(index = ["Mean","Median","Mode"],columns=quan)
for columnName in quan:
    descriptive[columnName]["Mean"]=dataset[columnName].mean()
    descriptive[columnName]["Median"]=dataset[columnName].median()
    descriptive[columnName]["Mode"]=dataset[columnName].mode()[0]

#Percentile
np.percentile(dataset['ssc_p'],70)

descriptive = pd.DataFrame(index = ["Mean","Median","Mode","Q1:25%","Q2:50%","Q3:75%","Q4:100%"],columns=quan)
for columnName in quan:
    descriptive[columnName]["Mean"]=dataset[columnName].mean()
    descriptive[columnName]["Median"]=dataset[columnName].median()
    descriptive[columnName]["Mode"]=dataset[columnName].mode()[0]
    descriptive[columnName]["Q1:25%"]=dataset.describe()[columnName]["25%"]
    descriptive[columnName]["Q2:50%"]=dataset.describe()[columnName]["50%"]
    descriptive[columnName]["Q3:75%"]=dataset.describe()[columnName]["75%"]
    descriptive[columnName]["Q4:100%"]=dataset.describe()[columnName]["max"]

    #print(descriptive)

#Inter quatile Range(IQR) - 1.5 rule
#IQR = Q3-Q1
#1.5 = 1.5*IQR
#lesser = Q1-1.5rule
#greater = Q3+1.5rule
descriptive = pd.DataFrame(index = ["Mean","Median","Mode","Q1:25%","Q2:50%","Q3:75%","Q4:100%","IQR","1.5Rule","Lesser","Greater","Min","Max"],columns=quan)
for columnName in quan:
    descriptive[columnName]["Mean"]=dataset[columnName].mean()
    descriptive[columnName]["Median"]=dataset[columnName].median()
    descriptive[columnName]["Mode"]=dataset[columnName].mode()[0]
    descriptive[columnName]["Q1:25%"]=dataset.describe()[columnName]["25%"]
    descriptive[columnName]["Q2:50%"]=dataset.describe()[columnName]["50%"]
    descriptive[columnName]["Q3:75%"]=dataset.describe()[columnName]["75%"]
    descriptive[columnName]["Q4:100%"]=dataset.describe()[columnName]["max"]
    descriptive[columnName]["IQR"]=descriptive[columnName]["Q3:75%"]-descriptive[columnName]["Q1:25%"]
    descriptive[columnName]["1.5Rule"]=1.5*descriptive[columnName]["IQR"]
    descriptive[columnName]["Lesser"]=descriptive[columnName]["Q1:25%"]-descriptive[columnName]["1.5Rule"]
    descriptive[columnName]["Greater"]=descriptive[columnName]["Q3:75%"]+descriptive[columnName]["1.5Rule"]
    descriptive[columnName]["Min"]=dataset[columnName].min()
    descriptive[columnName]["Max"]=dataset[columnName].max()

#print(descriptive)

#check outlier in the dataset
# lesser > Min
#Max > Greater
lesser = []
greater = []
for columnName in quan:
    if descriptive[columnName]["Lesser"] > descriptive[columnName]["Min"]:
        lesser.append(columnName)
    if descriptive[columnName]["Greater"] < descriptive[columnName]["Q4:100%"]:
        greater.append(columnName)

def check_outlier(quan,descriptive):
    lesser = []
    greater = []
    for columnName in quan:
        if descriptive[columnName]["Lesser"] > descriptive[columnName]["Min"]:
            lesser.append(columnName)
        if descriptive[columnName]["Greater"] < descriptive[columnName]["Q4:100%"]:
            greater.append(columnName)
    return(lesser,greater)

#print(lesser,greater)

#replace outlier in dataset
for columnName in lesser:
    dataset[columnName][dataset[columnName]<descriptive[columnName]["Lesser"]]=descriptive[columnName]["Lesser"]
    dataset[columnName][dataset[columnName]>descriptive[columnName]["Greater"]]=descriptive[columnName]["Greater"]

def replace_outlier(dataset, columns, descriptive):
    for columnName in columns:
        lower_bound = descriptive[columnName]["Lesser"]
        upper_bound = descriptive[columnName]["Greater"]

        dataset.loc[dataset[columnName] < lower_bound, columnName] = lower_bound
        dataset.loc[dataset[columnName] > upper_bound, columnName] = upper_bound

    return dataset



print(descriptive)
print(lesser,greater)

#Frequencr,Relative Frequency,Cumulative frequency
#dataset["ssc_p"].value_counts()

freqTable = pd.DataFrame(columns=["Unique_Values","Frequency","Relative_Frequency","cumsum"])
freqTable["Unique_Values"] = dataset["ssc_p"].value_counts().index
freqTable["Frequency"] = dataset["ssc_p"].value_counts().values
freqTable["Relative_Frequency"] = (freqTable["Frequency"]/103)
freqTable["cumsum"] = freqTable["Relative_Frequency"].cumsum()


def freqtable(columnName,dataset):
    freqTable = pd.DataFrame(columns=["Unique_Values","Frequency","Relative_Frequency","cumsum"])
    freqTable["Unique_Values"] = dataset[columnName].value_counts().index
    freqTable["Frequency"] = dataset[columnName].value_counts().values
    freqTable["Relative_Frequency"] = (freqTable["Frequency"]/103)
    freqTable["cumsum"] = freqTable["Relative_Frequency"].cumsum()
    return freqTable

#print(freqTable("ssc_p",dataset))

#Skewnesss,Kurtosis (add to univaiate funtion)
dataset["ssc_p"].skew()
dataset["ssc_p"].kurtosis()

#Stadard deviation and variation (add to this function)
dataset["ssc_p"].var()
dataset["ssc_p"].std()


#Normal Distribution
import seaborn as sns
sns.displot(dataset["ssc_p"])
#probability density function
