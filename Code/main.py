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

#%%

### 1. Data cleaning
# see Data_Cleaning.py

# reading the cleaned data form Data_Cleaning.py
return_df = data.ls_df_ma
bm_df = data.bm_df_ma
market_returns = data.market_returns
market_bm = data.market_bm

return_df_train = data.ls_df_ma_train
return_df_test = data.ls_df_ma_test
bm_df_train = data.bm_df_ma_train
bm_df_test = data.bm_df_ma_test
market_returns_train = data.market_returns_train
market_returns_test = data.market_returns_test
market_bm_train = data.market_bm_train
market_bm_test = data.market_bm_test


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
#X = sm.add_constant(market_bm_train)
m1_est = sm.OLS(return_df_test[1:,-1], market_bm_test[:-1]).fit()
print(m1_est.summary())


## PC1 regression   --> regressor: bm_i,t = q'_i * bm^F_i

# lin. comb. of eigenvector loadings q with bm (q'_i * bm^F_i)
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                # sterting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())

## PC2 regression   

X_bm_pc2 = sm.add_constant(np.dot(pc_eigenv_df.iloc[1,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc2 = return_df_test[1:,1]                                                                # sterting at 1 (t+1)

bm_pc2_est = sm.OLS(Y_ret_pc2, X_bm_pc2).fit()
print(bm_pc2_est.summary())


lm = LinearRegression()
model = lm.fit(X_bm_pc1, Y_ret_pc1)
print(model.coef_)
r_squared = model.score(return_df_Reg, bm_df_train)




#%% Jonas

# Add the market excess returns to the return PCs as 6th pricing factor
return_df_Reg = np.append(return_df_train, market_returns_train, axis=1)

# Regress 1995-2017 data of PC returns on net BM ratios
lm = LinearRegression()
model = lm.fit(return_df_Reg, bm_df_train)
r_squared = model.score(return_df_Reg, bm_df_train)

#import statsmodels.api as sm
#X2 = sm.add_constant(return_df_Reg)
#est = sm.OLS(bm_df_Reg, X2)
#est2 = est.fit()
#print(est2.summary)


#%%


### 4. Prediciting individual factors

### 5. Optimal factor timing portfolio