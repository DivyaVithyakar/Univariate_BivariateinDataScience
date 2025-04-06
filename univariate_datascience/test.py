import pandas as pd
from univariate_datascience.univariate import quanQual, univaiate,freqtable,check_outlier,replace_outlier

dataset = pd.read_csv("../data/Placement.csv")
# 1. Split columns
quan, qual = quanQual(dataset)

# 2. Get descriptive stats
desc = univaiate(dataset, quan)

# 3. Check outliers
lesser, greater = check_outlier(quan, desc)
outlier_cols = list(set(lesser + greater))

# 4. Replace outliers
df_cleaned = replace_outlier(dataset, outlier_cols, desc)

# 5. Frequency table (for example, on 'ssc_p')
print(freqtable("ssc_p", df_cleaned))

