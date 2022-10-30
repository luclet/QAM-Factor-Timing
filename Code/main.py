'''
Factor Timing - Quantitative Asset Management, Fall 2022
------------------------------------------------------------

Authors: Lucas Letulé, Jonas Neller, Lorena Tassone
'''
#%% 

import Data_Cleaning as data 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")



### 1. Data cleaning
# see Data_Cleaning.py

# reading the cleaned data form Data_Cleaning.py

#return_df = data.ls_df
#bm_df = data.bm_df_ma
market_returns = data.market_returns
market_bm = data.market_bm

return_df_train = data.ls_df_ma_train
return_df_test = data.ls_df_test
return_df_extra = data.ls_df_extra
frames = [return_df_train, return_df_test, return_df_extra]
return_df = pd.concat(frames)
bm_df_train = data.bm_df_ma_train
bm_df_test = data.bm_df_test
market_returns_train = data.market_returns_train
market_returns_test = data.market_returns_test
market_bm_train = data.market_bm_train
market_bm_test = data.market_bm_test


market_bm_aggr = data.market_bm_aggr
market_bm_aggr_train = data.market_bm_aggr_train
market_bm_aggr_test = data.market_bm_aggr_test


#%%
### 2. Dominant components of factors

## PCA Analysis
# Scaling data to have expectation of 0 and variance of 1
sc = StandardScaler()
return_df_train = sc.fit_transform(return_df_train)
return_df_test = sc.transform(return_df_test)


# Applying PCA function on training and testing set
n_pc = 5
pca = PCA(n_components = n_pc)
return_df_train = pca.fit_transform(return_df_train)
return_df_test = pca.transform(return_df_test)

# Explained variance of 5 PCs
explained_variance_ret = pca.explained_variance_ratio_.tolist()

# Percentage of variance explained by anomaly PCs (Table 1)
pc_list = ["PC"+str(i) for i in list(range(1, n_pc+1))]
cum_expl_var_df = np.cumsum(explained_variance_ret)
expl_var_df = pd.DataFrame([explained_variance_ret, cum_expl_var_df], columns=pc_list, index=['% var. explained', 'Cumulative'])
print('\nPercentage of variance explained by anomaly PCs:\n', expl_var_df)

'''
REMARK:
    First 5 PC explain over 60% of the variance, hence we use the first 5 as predicting dominant components!
'''
#return_df_train = pd.DataFrame(return_df_train)
pc_eigenv_df = pd.DataFrame(pca.components_,columns=return_df.columns,index = pc_list)
print(pc_eigenv_df)

# Including market pf as pricing factor
return_df_train = np.append(return_df_train, market_returns_train, axis=1)
return_df_test = np.append(return_df_test, market_returns_test, axis=1)

# return_df_t columns: PC1, PC2, PC3, PC4, PC5, MKT

#%%
### 3. Prediciting the large PCs of anomaly returns (Predictive Regression)
'''
- Construct a single predictor for each portfolio: we use its net book-to-market ratio
- For predicting PC_i,t+1, we construct its own log book-to-market ratio bm_i,t by combining 
  the anomaly log book-to-market ratios according to portfolio weights: bm_i,t = q'_i*bm^F_t. 
  We use the difference between this quantity for the long and short leg of our PCs, 
  thereby capturing potentially useful information about future expected returns
'''
'''
 - Eigenvectors: pca.components_ is the set of all eigenvectors (aka loadings) for your projection space (one eigenvector for each principal component).
 - Eigenvalues: https://stackoverflow.com/questions/31909945/obtain-eigen-values-and-vectors-from-sklearn-pca/31941631#31941631
'''
# Cals own bm_i,t for each PC_i,t+1
# --> for loow over all PC and MKT

## Market regression
X = sm.add_constant(np.log(market_bm_train[:-1]))
#X = sm.add_constant(np.log(market_bm_aggr_train[:-1]))
m1_est = sm.OLS(return_df_train[1:,-1], X).fit()
print(m1_est.summary())

## PC1 regression   --> regressor: bm_i,t = q'_i * bm^F_i

# lin. comb. of eigenvector loadings q with bm (q'_i * bm^F_i)
X_bm_pc1_train = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_train.transpose())[:-1])  # starting at 0 (t) and deleting last value
X_bm_pc1_test = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value

Y_ret_pc1_train = return_df_train[1:,0]                                                                 # starting at 1 (t+1)
Y_ret_pc1_test = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est_train = sm.OLS(Y_ret_pc1_train, X_bm_pc1_train).fit()
bm_pc1_est_test = sm.OLS(Y_ret_pc1_test, X_bm_pc1_test).fit()
print(bm_pc1_est_train.summary())


## PC2 regression   

X_bm_pc2_train = sm.add_constant(np.dot(pc_eigenv_df.iloc[1,:].values, bm_df_train.transpose())[:-1])  # starting at 0 (t) and deleting last value
X_bm_pc2_test = sm.add_constant(np.dot(pc_eigenv_df.iloc[1,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value

Y_ret_pc2_train = return_df_train[1:,1]                                                                 # starting at 1 (t+1)
Y_ret_pc2_test = return_df_test[1:,1]                                                                 # starting at 1 (t+1)

bm_pc2_est_train = sm.OLS(Y_ret_pc2_train, X_bm_pc2_train).fit()
bm_pc2_est_test = sm.OLS(Y_ret_pc2_test, X_bm_pc2_test).fit()
print(bm_pc2_est_train.summary())



## PC3 regression   

X_bm_pc3_train = sm.add_constant(np.dot(pc_eigenv_df.iloc[2,:].values, bm_df_train.transpose())[:-1])  # starting at 0 (t) and deleting last value
X_bm_pc3_test = sm.add_constant(np.dot(pc_eigenv_df.iloc[2,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value

Y_ret_pc3_train = return_df_train[1:,2]                                                                 # starting at 1 (t+1)
Y_ret_pc3_test = return_df_test[1:,2]                                                                 # starting at 1 (t+1)

bm_pc3_est_train = sm.OLS(Y_ret_pc3_train, X_bm_pc3_train).fit()
bm_pc3_est_test = sm.OLS(Y_ret_pc3_test, X_bm_pc3_test).fit()
print(bm_pc2_est_train.summary())



## PC4 regression   

X_bm_pc4_train = sm.add_constant(np.dot(pc_eigenv_df.iloc[3,:].values, bm_df_train.transpose())[:-1])  # starting at 0 (t) and deleting last value
X_bm_pc4_test = sm.add_constant(np.dot(pc_eigenv_df.iloc[3,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value

Y_ret_pc4_train = return_df_train[1:,3]                                                                 # starting at 1 (t+1)
Y_ret_pc4_test = return_df_test[1:,3]                                                                 # starting at 1 (t+1)

bm_pc4_est_train = sm.OLS(Y_ret_pc4_train, X_bm_pc4_train).fit()
bm_pc4_est_test = sm.OLS(Y_ret_pc4_test, X_bm_pc4_test).fit()
print(bm_pc4_est_train.summary())



## PC5 regression   

X_bm_pc5_train = sm.add_constant(np.dot(pc_eigenv_df.iloc[4,:].values, bm_df_train.transpose())[:-1])  # starting at 0 (t) and deleting last value
X_bm_pc5_test = sm.add_constant(np.dot(pc_eigenv_df.iloc[4,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value

Y_ret_pc5_train = return_df_train[1:,4]                                                                 # starting at 1 (t+1)
Y_ret_pc5_test = return_df_test[1:,4]                                                                 # starting at 1 (t+1)

bm_pc5_est_train = sm.OLS(Y_ret_pc5_train, X_bm_pc5_train).fit()
bm_pc5_est_test = sm.OLS(Y_ret_pc5_test, X_bm_pc5_test).fit()
print(bm_pc5_est_train.summary())






#%%
### 4. Prediciting individual factors

#To measure the conditional expected factor returns, apply these forecasts to factors using their loadings on the dominant components.
#Step 4 of our approach is to infer expected return forecasts for the individual factors from the forecasts of our dominant components.
#Factors are known linear combinations of the PC portfolios, so we can use the estimates in Table 2 to generate forecasts for each anomaly.
#Notably, each anomaly return is implicitly predicted by the whole cross-section of bm ratios.
#Table 4: Monthly predictive R2 of individual anomalies returns using implied fitted values based on PC forecasts. Column 1 (IS) provides estimates in the full sample. Column 2 (OOS) shows out-of-sample R2.

#My take:
#using implied fitted values of the PC forecasts of PC forecasts to run regression on individual factor returns? Then get R2??
#for x: maybe create a df with the fitted values of PC1-PC5, then regress that on each y (individual factor return) and get the R-squared of each regression
#loop over individual factor returns and append the R-squared to a mew df


# #IS Factor Returns
# IS_fac_ret = data.ls_df_ma
# #OOS Factor Returns
# OOS_fac_ret = data.ls_df_ma_test

#IS Factor Returns
IS_fac_ret = return_df
#OOS Factor Returns
OOS_fac_ret = return_df_test


X_fit_pc1_train = sm.add_constant(bm_pc1_est_train.fittedvalues)
#X_fit_pc1_OOS = 
#Y_ret_pc1 = IS_fac_ret

#bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
#print(bm_pc1_est.summary())








#%%
### 5. Optimal factor timing portfolio