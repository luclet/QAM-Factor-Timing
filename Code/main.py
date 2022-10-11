'''
Factor Timing - Quantitative Asset Management, Fall 2022
------------------------------------------------------------

Authors: Lucas Letulé, Jonas Neller, Lorena Tassone
'''

### 1. Data cleaning
import Data_Cleaning as data 

return_df = data.ls_df_ma
bm_df = data.bm_df_ma

return_df_train = data.ls_df_ma_train
return_df_test = data.ls_df_ma_test
bm_df_train = data.bm_df_ma_train
bm_df_test = data.bm_df_ma_test


### 2. Dominant components of factors

###PCA Analysis
# Scaling data to have expectation of 0 and variance of 1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
return_df_train = sc.fit_transform(return_df_train)
return_df_test = sc.transform(return_df_test)

#Remove NaN values in columns 22 and 24 of return_df_train array
import numpy as np
return_df_train[np.isnan(return_df_train)] = 0

# Applying PCA function on training and testing set
from sklearn.decomposition import PCA
pca = PCA(n_components = 5)
return_df_train = pca.fit_transform(return_df_train)
return_df_test = pca.transform(return_df_test)

# Explained variance of 5 PCs
explained_variance_ret = pca.explained_variance_ratio_


###Predictive Regressions
#Remove NaN columns from market returns
market_returns_train_clean = data.market_returns_train.drop(data.market_returns_train.columns[[1, 2, 3]], axis=1)  # df.columns is zero-based pd.Index

#Add the market excess returns to the return PCs as 6th pricing factor
return_df_Reg = np.append(return_df_train, market_returns_train_clean, axis=1)

#Regress 1995-2017 data of PC returns on net BM ratios
from sklearn.linear_model import LinearRegression
bm_df_train[np.isnan(bm_df_train)] = 0

lm = LinearRegression()
model = lm.fit(return_df_Reg, bm_df_train)
r_squared = model.score(return_df_Reg, bm_df_train)

#import statsmodels.api as sm
#X2 = sm.add_constant(return_df_Reg)
#est = sm.OLS(bm_df_Reg, X2)
#est2 = est.fit()
#print(est2.summary)



    # Dominant components of factors 
    # ->  We have 55, paper has 50?

### 3. Prediciting the large PCs of anomaly returns

### 4. Prediciting individual factors

### 5. Optimal factor timing portfolio