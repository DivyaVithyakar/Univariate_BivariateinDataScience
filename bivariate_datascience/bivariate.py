import  pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
import scipy.stats as stats

dataset = pd.read_csv("../data/PrePlacement.csv")
#print(dataset.isnull().sum())

# Select only numeric columns
numeric_data = dataset.select_dtypes(include=['number'])

# Check covariance
print("Covariance Matrix:")
print(numeric_data.cov())

# Check correlation
print("\nCorrelation Matrix:")
print(numeric_data.corr())

#variance inflation factor(VIF)
dataset.drop("sl_no",inplace=True,axis=1)
sns.pairplot(dataset)
plt.show()

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

#ttest - independent ttest
dataset=dataset.dropna()
male = dataset[dataset['gender']=='M']['salary']
female = dataset[dataset['gender']=='F']['salary']
#print(male)
ttest_ind(male, female)

#ttest - dependent
#dataset=dataset.dropna()
male = dataset[dataset['gender']=='M']['ssc_p']
male1 = dataset[dataset['gender']=='M']['hsc_p']
ttest_rel(male, male1)

#analysis of variance(ANAVO)
stats.f_oneway(dataset['ssc_p'],dataset['hsc_p'],dataset['degree_p'])