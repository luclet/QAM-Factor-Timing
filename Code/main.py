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

market_bm_aggr_train = data.market_bm_aggr_train
market_bm_aggr_test = data.market_bm_aggr_test


#%%
### 2. Dominant components of factors

## PCA Analysis
# Scaling data to have expectation of 0 and variance of 1
sc = StandardScaler()
return_df_train_pca = sc.fit_transform(return_df_train)
return_df_test_pca = sc.transform(return_df_test)


# Applying PCA function on training and testing set
n_pc = 5
pca = PCA(n_components = n_pc)
return_df_train_pca = pca.fit_transform(return_df_train_pca)
return_df_test_pca = pca.transform(return_df_test_pca)


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


##### NEW ######## ACTUALLY WERTE ZIEMLICH ÄHNLICH OB SO ODER WIE OBEN ZEILE 59
return_df_train_pca = pd.DataFrame(np.dot(pc_eigenv_df, return_df_train.transpose()).transpose())
return_df_train_pca = return_df_train_pca.set_index(return_df_train.index)
return_df_train_pca.columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']

return_df_test_pca = pd.DataFrame(np.dot(pc_eigenv_df, return_df_test.transpose()).transpose())
return_df_test_pca = return_df_test_pca.set_index(return_df_test.index)
return_df_test_pca.columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
########################


# Including market pf as pricing factor
return_df_train_pca['MKT'] = market_returns_train
return_df_test_pca['MKT'] = market_returns_test



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

params = []

## Market regression
X = sm.add_constant(np.log(market_bm_train[:-1]))
#X = sm.add_constant(np.log(market_bm_aggr_train[:-1]))
#XX = market_bm_train.to_numpy()
#XX = np.log(market_bm_train)
m1_est = sm.OLS(return_df_train_pca.iloc[1:,-1].values, X).fit()
#print(m1_est.summary())


## PC1 regression   --> regressor: bm_i,t = q'_i * bm^F_i
# lin. comb. of eigenvector loadings q with bm (q'_i * bm^F_i)
X_bm_pc1_train = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_train.transpose())[:-1])  # starting at 0 (t) and deleting last value
X_bm_pc1_test = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value

Y_ret_pc1_train = return_df_train_pca.iloc[1:,0]                                                                 # starting at 1 (t+1)
Y_ret_pc1_test = return_df_test_pca.iloc[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est_train = sm.OLS(Y_ret_pc1_train, X_bm_pc1_train).fit()
bm_pc1_est_test = sm.OLS(Y_ret_pc1_test, X_bm_pc1_test).fit()
print(bm_pc1_est_train.summary())


## PC2 regression   
X_bm_pc2_train = sm.add_constant(np.dot(pc_eigenv_df.iloc[1,:].values, bm_df_train.transpose())[:-1])  # starting at 0 (t) and deleting last value
X_bm_pc2_test = sm.add_constant(np.dot(pc_eigenv_df.iloc[1,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value

Y_ret_pc2_train = return_df_train_pca.iloc[1:,1]                                                                 # starting at 1 (t+1)
Y_ret_pc2_test = return_df_test_pca.iloc[1:,1]                                                                 # starting at 1 (t+1)

bm_pc2_est_train = sm.OLS(Y_ret_pc2_train, X_bm_pc2_train).fit()
bm_pc2_est_test = sm.OLS(Y_ret_pc2_test, X_bm_pc2_test).fit()
print(bm_pc2_est_train.summary())


## PC3 regression   
X_bm_pc3_train = sm.add_constant(np.dot(pc_eigenv_df.iloc[2,:].values, bm_df_train.transpose())[:-1])  # starting at 0 (t) and deleting last value
X_bm_pc3_test = sm.add_constant(np.dot(pc_eigenv_df.iloc[2,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value

Y_ret_pc3_train = return_df_train_pca.iloc[1:,2]                                                                 # starting at 1 (t+1)
Y_ret_pc3_test = return_df_test_pca.iloc[1:,2]                                                                 # starting at 1 (t+1)

bm_pc3_est_train = sm.OLS(Y_ret_pc3_train, X_bm_pc3_train).fit()
bm_pc3_est_test = sm.OLS(Y_ret_pc3_test, X_bm_pc3_test).fit()
print(bm_pc3_est_train.summary())


## PC4 regression   
X_bm_pc4_train = sm.add_constant(np.dot(pc_eigenv_df.iloc[3,:].values, bm_df_train.transpose())[:-1])  # starting at 0 (t) and deleting last value
X_bm_pc4_test = sm.add_constant(np.dot(pc_eigenv_df.iloc[3,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value

Y_ret_pc4_train = return_df_train_pca.iloc[1:,3]                                                                 # starting at 1 (t+1)
Y_ret_pc4_test = return_df_test_pca.iloc[1:,3]                                                                 # starting at 1 (t+1)

bm_pc4_est_train = sm.OLS(Y_ret_pc4_train, X_bm_pc4_train).fit()
bm_pc4_est_test = sm.OLS(Y_ret_pc4_test, X_bm_pc4_test).fit()
print(bm_pc4_est_train.summary())


## PC5 regression   
X_bm_pc5_train = sm.add_constant(np.dot(pc_eigenv_df.iloc[4,:].values, bm_df_train.transpose())[:-1])  # starting at 0 (t) and deleting last value
X_bm_pc5_test = sm.add_constant(np.dot(pc_eigenv_df.iloc[4,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value

Y_ret_pc5_train = return_df_train_pca.iloc[1:,4]                                                                 # starting at 1 (t+1)
Y_ret_pc5_test = return_df_test_pca.iloc[1:,4]                                                                 # starting at 1 (t+1)

bm_pc5_est_train = sm.OLS(Y_ret_pc5_train, X_bm_pc5_train).fit()
bm_pc5_est_test = sm.OLS(Y_ret_pc5_test, X_bm_pc5_test).fit()
print(bm_pc5_est_train.summary())



# collect parameters
regressions = [m1_est, bm_pc1_est_train, bm_pc2_est_train, bm_pc3_est_train, bm_pc4_est_train, bm_pc5_est_train]
output_df  = pd.DataFrame(index =['Own bm','Std. dev.', 'p-value', 'R_squared'], columns = ['MKT', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

for idx, regr in enumerate(regressions):
    output_df.iloc[0, idx] = regr.params[1]
    output_df.iloc[1, idx] = regr.bse[1]
    output_df.iloc[2, idx] = regr.pvalues[1]
    output_df.iloc[3, idx] = regr.rsquared

output_df = output_df.astype(float).round(2)
print(output_df)

#%%
### 4. Prediciting individual factors

#To measure the conditional expected factor returns, apply these forecasts to factors using their loadings on the dominant components.
#Step 4 of our approach is to infer expected return forecasts for the individual factors from the forecasts of our dominant components.
#Factors are known linear combinations of the PC portfolios, so we can use the estimates in Table 2 to generate forecasts for each anomaly.
#Notably, each anomaly return is implicitly predicted by the whole cross-section of bm ratios.
#Table 4: Monthly predictive R2 of individual anomalies returns using implied fitted values based on PC forecasts. Column 1 (IS) provides estimates in the full sample. Column 2 (OOS) shows out-of-sample R2.


#Lucas take:
#using implied fitted values of the PC forecasts of PC forecasts to run regression on individual factor returns? Then get R2??
#for x: maybe create a df with the fitted values of PC1-PC5, then regress that on each y (individual factor return) and get the R-squared of each regression
#loop over individual factor returns and append the R-squared to a mew df

'''
#IS Factor Returns
IS_fac_ret = data.ls_df_ma_train  # using scaled or non-scaled ones for train set?

#OOS Factor Returns
OOS_fac_ret = data.ls_df_test

X_fit_pc_train = [bm_pc1_est_train.fittedvalues, bm_pc2_est_train.fittedvalues, bm_pc3_est_train.fittedvalues, bm_pc4_est_train.fittedvalues, bm_pc5_est_train.fittedvalues]
X_fit_pc_train = np.transpose(X_fit_pc_train)
X_fit_pc_train = sm.add_constant(X_fit_pc_train)

bm_pc1_est = sm.OLS(list(IS_fac_ret['accruals'].iloc[1:]), X_fit_pc_train).fit()
#X_fit_pc_train[:,0], X_fit_pc_train[:,1], X_fit_pc_train[:,2], X_fit_pc_train[:,3], X_fit_pc_train[:,4], X_fit_pc_train[:,5]
print(bm_pc1_est.summary())
'''



### Lorena's try 
# https://www.activestate.com/resources/quick-reads/how-to-make-predictions-with-scikit-learn/

# regression estimates * factor pc loadings
new_estimates = []

for anom in pc_eigenv_df:
    new_estim = np.dot(output_df.iloc[0,1:], pc_eigenv_df[anom])
    new_estimates.append(new_estim)


# x = % change in bm
delta_bm_train = bm_df_train / bm_df_train.shift(1) - 1
delta_bm_train = delta_bm_train.iloc[1:-1,:]                      # % veränderung bm
#delta_ret_train = np.multiply(delta_bm_train, new_estimates)      # % veränderung anomaly return
 
delta_bm_test = bm_df_test / bm_df_test.shift(1) - 1 
delta_bm_test = delta_bm_test.iloc[1:-1,:] 
#delta_ret_test = np.multiply(delta_bm_test, new_estimates) 


# y = % change factor returns
delta_ret_train = return_df_train / return_df_train.shift(1) - 1
delta_ret_train = delta_ret_train.iloc[2:,:] 

delta_ret_test = return_df_test / return_df_test.shift(1) - 1
delta_ret_test = delta_ret_test.iloc[2:,:] 

# regression model
# x_acc_train = sm.add_constant(x_train['accruals'])  
# #y_acc_train = list(y_train['accruals'])  
# y_acc_train = return_df_train[['accruals']]
# abc = y_acc_train.iloc[1: , :]
# abcd = y_acc_train.iloc[:-1 , :]                                                                                                        
# acc_train = sm.OLS(abcd, x_acc_train).fit()
# print(acc_train.summary())
# print('\nR_squared: ',acc_train.rsquared)



#List of in sample R2s for all anomalies
R2_List_IS = []
for anom in delta_bm_train:
    x_reg = sm.add_constant(delta_bm_train[anom].reset_index(drop=True))  
    y_reg = delta_ret_train[[anom]].reset_index(drop=True)                                                                                             
    reg_train = sm.OLS(y_reg, x_reg).fit()
    R2 = reg_train.rsquared
    R2_List_IS.append(R2)

#List of out of sample R2s for all anomalies
R2_List_OOS = []
for anom in delta_bm_test:
    x_reg = sm.add_constant(delta_bm_test[anom].reset_index(drop=True) )  
    y_reg = delta_ret_test[[anom]].reset_index(drop=True)                                                                                                    
    reg_test = sm.OLS(y_reg, x_reg).fit()
    R2 = reg_test.rsquared
    R2_List_OOS.append(R2)

anomalies = return_df.columns.tolist()
R2_Table = [R2_List_IS, R2_List_OOS, anomalies]
R2_Table = pd.DataFrame(R2_Table).T
R2_Table = R2_Table.set_index(2)
R2_Table.columns = ['IS', 'OOS']
R2_Table.index.name = 'Anomalies'
print(R2_Table)


''' TODO: CORRECT!
# aufsplitten nach faktoren
# forecast errors per pc
# matrix soll cov von pc returns und faktor returns haben => zeilen factors, spalten pcs und jeweils deren covs


#Constructing forecast errors
fc_err_train = delta_bm_train.subtract(return_df_train, axis='columns', level=None, fill_value=None)
fc_err_train = fc_err_train.iloc[1: , :]
fc_err_train = fc_err_train.iloc[:-1 , :]

fc_err_test = delta_bm_test.subtract(return_df_test, axis='columns', level=None, fill_value=None)
fc_err_test = fc_err_test.iloc[1: , :]
fc_err_test = fc_err_test.iloc[:-1 , :]

return_df_test_trimmed = return_df_test.iloc[1: , :]
return_df_test_trimmed = return_df_test_trimmed.iloc[:-1 , :] 

cov_df = [delta_bm_train, return_df_train, Anomalies]
cov_df = pd.DataFrame(cov_df).T
cov_df = cov_df.set_index(2)
cov_df.columns = ['Predicted', 'Actual']
cov_df.index.name = 'Anomalies'

var_fc_err = fc_err_test.var()
var_ret = return_df_test.var()
Almost = var_fc_err.div(var_ret)
AAlmost = Almost.sub(1)
#Constructing conditional covariance matrix of market and PC returns

'''






#%%
#### 5. Optimal factor timing portfolio

### IS

## Calculate component means E_t[Z_{t+1}]

# Calculate own bm ratio timeseries for each PCs and MKT and create df
PCs = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
step5_bm_df_train = {}

for idx, pc in enumerate(PCs):
    step5_bm_df_train[pc] = np.dot(pc_eigenv_df.iloc[idx,:].values, bm_df_train.transpose())[:-1]

step5_bm_df_train = pd.DataFrame(step5_bm_df_train).set_index(bm_df_train.index[:-1])

step5_bm_df_train['MKT'] = np.log(market_bm_train[:-1]) # correct?


# modifie PC and MKT return df from pca
step5_return_df_train_pca = pd.DataFrame(return_df_train_pca).iloc[1:,:].set_index(bm_df_train.index[1:])
step5_return_df_train_pca.columns = step5_bm_df_train.columns


# Calculate % change in bm
step5_bmc_df_train = step5_bm_df_train / step5_bm_df_train.shift(1) - 1
step5_bmc_df_train = step5_bmc_df_train.iloc[1:, :]


# move columns MKT from output_df at the end
output_df_new = output_df[['PC1','PC2', 'PC3', 'PC4', 'PC5', 'MKT']]


# Multiply % change in bm with regression beta => gives % change in PC and MKT return
step5_retc_df_train = np.multiply(step5_bmc_df_train, output_df_new.iloc[0,:])


# Calculate predicted return: ret*(1+%change)
step5_return_df_train_pca_adj = step5_return_df_train_pca.iloc[:-1,:]
step5_predicted_ret_df_train = np.multiply(step5_return_df_train_pca_adj, 1+step5_retc_df_train)

step5_predicted_ret_df_train.index = step5_return_df_train_pca.index[1:]


## Calculate forecast errors 
step5_forecast_errors_train = abs(step5_predicted_ret_df_train - step5_return_df_train_pca.iloc[1:,:])


## Calculate estimate of conditional Covariance matrix
step5_cov_matrix_train = step5_predicted_ret_df_train.cov()   # conditional?


## Calculate portfolio weights
step5_weights_train = np.dot(np.linalg.inv(step5_cov_matrix_train), step5_predicted_ret_df_train.transpose()).transpose()
step5_weights_train = pd.DataFrame(step5_weights_train, index=step5_retc_df_train.index, columns=step5_retc_df_train.columns)

sum_weights_train = step5_weights_train.sum(axis=1)


# rescale weights
step5_weights_train_scaled = step5_weights_train.apply(lambda x: x/x.sum(), axis=1)



### OOS

step5_bm_df_test = {}

for idx, pc in enumerate(PCs):
    step5_bm_df_test[pc] = np.dot(pc_eigenv_df.iloc[idx,:].values, bm_df_test.transpose())[:-1]

step5_bm_df_test = pd.DataFrame(step5_bm_df_test).set_index(bm_df_test.index[:-1])

step5_bm_df_test['MKT'] = np.log(market_bm_test[:-1]) # correct?


# modifie PC and MKT return df from pca
step5_return_df_test_pca = pd.DataFrame(return_df_test_pca).iloc[1:,:].set_index(bm_df_test.index[1:])
step5_return_df_test_pca.columns = step5_bm_df_test.columns


# Calculate % change in bm
step5_bmc_df_test = step5_bm_df_test / step5_bm_df_test.shift(1) - 1
step5_bmc_df_test = step5_bmc_df_test.iloc[1:, :]


# move columns MKT from output_df at the end
output_df_new = output_df[['PC1','PC2', 'PC3', 'PC4', 'PC5', 'MKT']]

# Multiply % change in bm with regression beta => gives % change in PC and MKT return
step5_retc_df_test = np.multiply(step5_bmc_df_test, output_df_new.iloc[0,:])

# Calculate predicted return: ret*(1+%change)
step5_return_df_test_pca_adj = step5_return_df_test_pca.iloc[:-1,:]
step5_predicted_ret_df_test= np.multiply(step5_return_df_test_pca_adj, 1+step5_retc_df_test)

step5_predicted_ret_df_test.index = step5_return_df_test_pca.index[1:]


## Calculate forecast errors 
step5_forecast_errors_test = abs(step5_predicted_ret_df_test - step5_return_df_test_pca.iloc[1:,:])

## Calculate estimate of conditional Covariance matrix
step5_cov_matrix_test = step5_predicted_ret_df_test.cov()   # conditional?

## Calculate portfolio weights
step5_weights_test = np.dot(np.linalg.inv(step5_cov_matrix_test), step5_predicted_ret_df_test.transpose()).transpose()
step5_weights_test = pd.DataFrame(step5_weights_test, index=step5_retc_df_test.index, columns=step5_retc_df_test.columns)

sum_weights_test = step5_weights_test.sum(axis=1)

# rescale weights
step5_weights_test_scaled = step5_weights_test.apply(lambda x: x/x.sum(), axis=1)






