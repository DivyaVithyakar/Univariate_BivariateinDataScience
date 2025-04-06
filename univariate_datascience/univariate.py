import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF

def quanQual(dataset):
    """
    Splits dataset columns into quantitative (numerical) and qualitative (categorical) lists.
    """
    quan = []
    qual = []

    for columnName in dataset.columns:
        if dataset[columnName].dtype == 'object':
            qual.append(columnName)
        else:
            quan.append(columnName)
    return quan,qual

def univaiate(dataset,quan):
    """
    Generates descriptive statistics and outlier thresholds for quantitative columns.
    """
    descriptive = pd.DataFrame(index = ["Mean","Median","Mode","Q1:25%","Q2:50%","Q3:75%","Q4:100%","IQR","1.5Rule","Lesser","Greater","Min","Max","Skew","Kurtosis","Var","Std_Deviation"],columns=quan)
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
        descriptive[columnName]["Skew"] = dataset[columnName].skew()
        descriptive[columnName]["Kurtosis"] = dataset[columnName].kurtosis()
        descriptive[columnName]["Var"] = dataset[columnName].var()
        descriptive[columnName]["Std_Deviation"] = dataset[columnName].std()
    return descriptive

def freqtable(columnName,dataset):
    """
    Creates frequency table with counts, relative frequency, and cumulative sum for a column.
    """
    freqTable = pd.DataFrame(columns=["Unique_Values","Frequency","Relative_Frequency","cumsum"])
    freqTable["Unique_Values"] = dataset[columnName].value_counts().index
    freqTable["Frequency"] = dataset[columnName].value_counts().values
    freqTable["Relative_Frequency"] = (freqTable["Frequency"]/103)
    freqTable["cumsum"] = freqTable["Relative_Frequency"].cumsum()
    return freqTable

def check_outlier(quan,descriptive):
    """
   Identifies columns with outliers based on 1.5*IQR rule.
   """
    lesser = []
    greater = []
    for columnName in quan:
        if descriptive[columnName]["Lesser"] > descriptive[columnName]["Min"]:
            lesser.append(columnName)
        if descriptive[columnName]["Greater"] < descriptive[columnName]["Q4:100%"]:
            greater.append(columnName)
    return(lesser,greater)

def replace_outlier(dataset, columns, descriptive):
    """
    Replaces outliers in dataset with threshold values from descriptive statistics.
    """
    for columnName in columns:
        lower_bound = descriptive[columnName]["Lesser"]
        upper_bound = descriptive[columnName]["Greater"]

        dataset.loc[dataset[columnName] < lower_bound, columnName] = lower_bound
        dataset.loc[dataset[columnName] > upper_bound, columnName] = upper_bound

    return dataset

def get_pdf_probability(dataset,startrange,endrange):
    ax = sns.displot(dataset,kde=True,kde_kws={'color':'blue'},color='Green')
    pyplot.axvline(startrange,color='Red')
    pyplot.axvline(endrange,color='Red')
    # generate a sample
    sample = dataset
    # calculate parameters
    sample_mean =sample.mean()
    sample_std = sample.std()
    print('Mean=%.3f, Standard Deviation=%.3f' % (sample_mean, sample_std))
    # define the distribution
    dist = norm(sample_mean, sample_std)

    # sample probabilities for a range of outcomes
    values = [value for value in range(startrange, endrange)]
    probabilities = [dist.pdf(value) for value in values]
    prob=sum(probabilities)
    print("The area between range({},{}):{}".format(startrange,endrange,sum(probabilities)))
    return prob


def compute_ecdf(dataset, column_name, value=None):
    """
    Computes ECDF for a given column.
    If a value is provided, returns the ECDF value at that point.

    Parameters:
    - dataset: pd.DataFrame
    - column_name: str
    - value: float or int (optional)

    Returns:
    - ECDF function or ECDF value at a specific point
    """
    ecdf = ECDF(dataset[column_name])

    if value is not None:
        return ecdf(value)

    return ecdf

def stdNBgraph(dataset):
    # Coverted to standard Normal Distribution
    import seaborn as sns
    mean=dataset.mean()
    std=dataset.std()

    values=[i for i in dataset]

    z_score=[((j-mean)/std) for j in values]

    sns.distplot(z_score,kde=True)

    sum(z_score)/len(z_score)
    #z_score.std()
