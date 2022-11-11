'''
Factor Timing - Quantitative Asset Management, Fall 2022
------------------------------------------------------------

Authors: Lucas Letulé, Jonas Neller, Lorena Tassone
'''
#%% 

import Data_Cleaning_volatility_long_short as data 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

### RMINDER: bm_df = var_df !! ###


### 1. Data cleaning
# see Data_Cleaning.py

# reading the cleaned data from Data_Cleaning.py

return_df_train = data.ls_df_ma_train
return_df_test = data.ls_df_test
return_df_extra = data.ls_df_extra

frames = [return_df_train, return_df_test, return_df_extra]
return_df = pd.concat(frames)

bm_df_train = data.bm_df_train
bm_df_test = data.bm_df_test
bm_df = data.bm_df

market_returns = data.market_returns
market_returns_train = data.market_returns_train
market_returns_test = data.market_returns_test

market_bm = data.market_bm
market_bm_train = data.market_bm_train
market_bm_test = data.market_bm_test



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


# calculating pca returns weighted by eigenvector loading
return_df_train_pca = pd.DataFrame(np.dot(pc_eigenv_df, return_df_train.transpose()).transpose())
return_df_train_pca = return_df_train_pca.set_index(return_df_train.index)
return_df_train_pca.columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']

return_df_test_pca = pd.DataFrame(np.dot(pc_eigenv_df, return_df_test.transpose()).transpose())
return_df_test_pca = return_df_test_pca.set_index(return_df_test.index)
return_df_test_pca.columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']



# Including market pf as pricing factor
return_df_train_pca['MKT'] = market_returns_train
return_df_test_pca['MKT'] = market_returns_test



#%%
### 3. Prediciting the large PCs of anomaly returns (Predictive Regression)


params = []

## Market regression
X = sm.add_constant(np.log(market_bm_train[:-1]))
m1_est = sm.OLS(return_df_train_pca.iloc[1:,-1].values, X).fit()
#print(m1_est.summary())


## PC1 regression   
X_bm_pc1_train = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_train.transpose())[:-1])  # starting at 0 (t) and deleting last value
X_bm_pc1_test = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value

Y_ret_pc1_train = return_df_train_pca.iloc[1:,0]                                                                 # starting at 1 (t+1)
Y_ret_pc1_test = return_df_test_pca.iloc[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est_train = sm.OLS(Y_ret_pc1_train, X_bm_pc1_train).fit()
bm_pc1_est_test = sm.OLS(Y_ret_pc1_test, X_bm_pc1_test).fit()
#print(bm_pc1_est_train.summary())


## PC2 regression   
X_bm_pc2_train = sm.add_constant(np.dot(pc_eigenv_df.iloc[1,:].values, bm_df_train.transpose())[:-1])  # starting at 0 (t) and deleting last value
X_bm_pc2_test = sm.add_constant(np.dot(pc_eigenv_df.iloc[1,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value

Y_ret_pc2_train = return_df_train_pca.iloc[1:,1]                                                                 # starting at 1 (t+1)
Y_ret_pc2_test = return_df_test_pca.iloc[1:,1]                                                                 # starting at 1 (t+1)

bm_pc2_est_train = sm.OLS(Y_ret_pc2_train, X_bm_pc2_train).fit()
bm_pc2_est_test = sm.OLS(Y_ret_pc2_test, X_bm_pc2_test).fit()
#print(bm_pc2_est_train.summary())


## PC3 regression   
X_bm_pc3_train = sm.add_constant(np.dot(pc_eigenv_df.iloc[2,:].values, bm_df_train.transpose())[:-1])  # starting at 0 (t) and deleting last value
X_bm_pc3_test = sm.add_constant(np.dot(pc_eigenv_df.iloc[2,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value

Y_ret_pc3_train = return_df_train_pca.iloc[1:,2]                                                                 # starting at 1 (t+1)
Y_ret_pc3_test = return_df_test_pca.iloc[1:,2]                                                                 # starting at 1 (t+1)

bm_pc3_est_train = sm.OLS(Y_ret_pc3_train, X_bm_pc3_train).fit()
bm_pc3_est_test = sm.OLS(Y_ret_pc3_test, X_bm_pc3_test).fit()
#print(bm_pc3_est_train.summary())


## PC4 regression   
X_bm_pc4_train = sm.add_constant(np.dot(pc_eigenv_df.iloc[3,:].values, bm_df_train.transpose())[:-1])  # starting at 0 (t) and deleting last value
X_bm_pc4_test = sm.add_constant(np.dot(pc_eigenv_df.iloc[3,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value

Y_ret_pc4_train = return_df_train_pca.iloc[1:,3]                                                                 # starting at 1 (t+1)
Y_ret_pc4_test = return_df_test_pca.iloc[1:,3]                                                                 # starting at 1 (t+1)

bm_pc4_est_train = sm.OLS(Y_ret_pc4_train, X_bm_pc4_train).fit()
bm_pc4_est_test = sm.OLS(Y_ret_pc4_test, X_bm_pc4_test).fit()
#print(bm_pc4_est_train.summary())


## PC5 regression   
X_bm_pc5_train = sm.add_constant(np.dot(pc_eigenv_df.iloc[4,:].values, bm_df_train.transpose())[:-1])  # starting at 0 (t) and deleting last value
X_bm_pc5_test = sm.add_constant(np.dot(pc_eigenv_df.iloc[4,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value

Y_ret_pc5_train = return_df_train_pca.iloc[1:,4]                                                                 # starting at 1 (t+1)
Y_ret_pc5_test = return_df_test_pca.iloc[1:,4]                                                                 # starting at 1 (t+1)

bm_pc5_est_train = sm.OLS(Y_ret_pc5_train, X_bm_pc5_train).fit()
bm_pc5_est_test = sm.OLS(Y_ret_pc5_test, X_bm_pc5_test).fit()
#print(bm_pc5_est_train.summary())


# collect parameters
regressions = [m1_est, bm_pc1_est_train, bm_pc2_est_train, bm_pc3_est_train, bm_pc4_est_train, bm_pc5_est_train]
output_df  = pd.DataFrame(index =['12-m Vola.','Std. dev.', 'p-value', 'R_squared'], columns = ['MKT', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

for idx, regr in enumerate(regressions):
    output_df.iloc[0, idx] = regr.params[1]
    output_df.iloc[1, idx] = regr.bse[1]
    output_df.iloc[2, idx] = regr.pvalues[1]
    output_df.iloc[3, idx] = regr.rsquared

output_df = output_df.astype(float).round(2)
print(output_df)



#%%
### 4. Prediciting individual factors

# regression estimates * factor pc loadings
new_estimates = []

# Obtain betas for all anomalies by multiplying PC Betas with loadings across all anomalies
for anom in pc_eigenv_df:
    new_estim = np.dot(output_df.iloc[0,1:], pc_eigenv_df[anom])
    new_estimates.append(new_estim)


#Calculating % change in bm for train and test
delta_bm_train = bm_df_train / bm_df_train.shift(1) - 1
delta_bm_train = delta_bm_train.iloc[1:-1,:]                      # % veränderung bm

delta_bm_test = bm_df_test / bm_df_test.shift(1) - 1 
delta_bm_test = delta_bm_test.iloc[1:-1,:]


#Calculating predicted change in returns for train and test
delta_predict_ret_train = np.multiply(delta_bm_train, new_estimates)     # % veränderung anomaly return
delta_predict_ret_test = np.multiply(delta_bm_test, new_estimates) 
delta_predict_ret_train.index = bm_df_train[2:].index
delta_predict_ret_test.index = bm_df_test[2:].index


# Calculate predicted return: ret*(1+%change)
return_df_train_adj = return_df_train.head(-1)
return_df_train_adj = return_df_train_adj.tail(-1)
step4_predicted_ret_train = np.multiply(return_df_train_adj, 1+delta_predict_ret_train)
step4_predicted_ret_train.index = bm_df_train[2:].index

return_df_test_adj = return_df_test.head(-1)
return_df_test_adj = return_df_test_adj.tail(-1)
step4_predicted_ret_test = np.multiply(return_df_test_adj, 1+delta_predict_ret_test)
step4_predicted_ret_test.index = bm_df_test[2:].index


step4_return_df_train = return_df_train.tail(-2)
step4_return_df_test = return_df_test.tail(-2)
step4_return_df_train.index = step4_predicted_ret_train.index
step4_return_df_test.index = step4_predicted_ret_test.index


#List of in sample R2s for all anomalies
R2_List_IS = []
for anom in step4_predicted_ret_train:
    x_reg = sm.add_constant(step4_predicted_ret_train[anom].reset_index(drop=True))  
    y_reg = step4_return_df_train[[anom]].reset_index(drop=True)                                                                                             
    reg_train = sm.OLS(y_reg, x_reg).fit()
    R2 = reg_train.rsquared
    R2_List_IS.append(R2)

#List of out of sample R2s for all anomalies
R2_List_OOS = []
for anom in step4_predicted_ret_test:
    x_reg = sm.add_constant(step4_predicted_ret_test[anom].reset_index(drop=True) )  
    y_reg = step4_return_df_test[[anom]].reset_index(drop=True)                                                                                                    
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




#%%
#### 5. Optimal factor timing portfolio

### IS

## Calculate component means E_t[Z_{t+1}]

# Calculate own bm ratio timeseries for each PCs and MKT and create df
PCs = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
bm_df_step5 = bm_df['1974-01-01':'2017-12-01']
step5_bm_df = {}

for idx, pc in enumerate(PCs):
    step5_bm_df[pc] = np.dot(pc_eigenv_df.iloc[idx,:].values, bm_df_step5.transpose())[:-1]

step5_bm_df = pd.DataFrame(step5_bm_df).set_index(bm_df_step5.index[:-1])

step5_bm_df['MKT'] = np.log(market_bm[:-1]) # correct?
step5_return_df_pca = pd.concat([return_df_train_pca, return_df_test_pca], axis=0)


# modify PC and MKT return df from pca
step5_return_df_pca = pd.DataFrame(step5_return_df_pca).iloc[1:,:].set_index(bm_df_step5.index[1:])
step5_return_df_pca.columns = step5_bm_df.columns


# Calculate % change in bm
step5_bmc_df = step5_bm_df / step5_bm_df.shift(1) - 1
step5_bmc_df = step5_bmc_df.iloc[1:, :]


# move columns MKT from output_df at the end
output_df_new = output_df[['PC1','PC2', 'PC3', 'PC4', 'PC5', 'MKT']]


# Multiply % change in bm with regression beta => gives % change in PC and MKT return
step5_retc_df = np.multiply(step5_bmc_df, output_df_new.iloc[0,:])

# Calculate predicted return: ret*(1+%change)
step5_return_df_pca_adj = step5_return_df_pca.iloc[:-1,:] #cut off last line
step5_predicted_ret_df = np.multiply(step5_return_df_pca_adj, 1+step5_retc_df)
step5_predicted_ret_df.index = step5_return_df_pca.index[1:]


# Calculate forecast errors 
step5_forecast_errors = abs(step5_predicted_ret_df - step5_return_df_pca.iloc[1:,:])


# Calculate estimate of conditional Covariance matrix
step5_cov_matrix = step5_predicted_ret_df.cov()   # conditional?


# Calculate portfolio weights
step5_weights = np.dot(np.linalg.inv(step5_cov_matrix), step5_predicted_ret_df.transpose()).transpose()
step5_weights = pd.DataFrame(step5_weights, index=step5_retc_df.index, columns=step5_retc_df.columns)
sum_weights = step5_weights.sum(axis=1)


# rescale weights
step5_weights_scaled = step5_weights.apply(lambda x: x/x.sum(), axis=1)
#step5_weights_scaled = step5_weights.div(sum_weights, axis=0) #rescaling appears to be correct, same results both ways
step5_sum_weights_scaled = step5_weights_scaled.sum(axis=1)


#Cumulative market returns over entire period
step5_final_returns = np.multiply(step5_weights_scaled, step5_return_df_pca_adj)
step5_cumulative_market_returns = np.cumprod(1 + data.market_returns['ret'].values) - 1
step5_cumulative_market_returns = pd.DataFrame(step5_cumulative_market_returns)
step5_cumulative_market_returns = step5_cumulative_market_returns.head(-1)
step5_cumulative_market_returns = step5_cumulative_market_returns.tail(-1)
step5_cumulative_market_returns.index = step5_final_returns.index


# Actual strategy returns
step5_sum_final_returns = step5_final_returns.sum(axis=1)
step5_sum_final_returns = step5_sum_final_returns.to_frame('ret')
step5_cumulative_returns = np.cumprod(1 + step5_sum_final_returns.values) - 1
step5_cumulative_returns = pd.DataFrame(step5_cumulative_returns)
step5_cumulative_returns.index = step5_cumulative_market_returns.index


#Sharpe Ratio assuming a risk-free rate of 0
Sharpe_Ratio_prelim = np.dot(step5_final_returns.mean().transpose(), step5_cov_matrix)
return_df_pca = pd.concat([return_df_train_pca, return_df_test_pca], axis = 0)
return_df_pca_adj = return_df_pca.tail(-2)
Sharpe_Ratio = np.dot(Sharpe_Ratio_prelim, return_df_pca_adj.mean())

#Information Ratio
step5_excess_returns = step5_cumulative_returns.subtract(step5_cumulative_market_returns)

step5_excess_returns_cum = step5_cumulative_returns.subtract(step5_cumulative_market_returns)
step5_excess_returns_cum = step5_excess_returns.iloc[261]
step5_excess_returns_cum = step5_excess_returns_cum.to_frame('ret')

market_returns_adj = data.market_returns.tail(-2)
market_returns_adj.index = step5_final_returns.index

Forecast_errors = step5_sum_final_returns.subtract(market_returns_adj, axis = 1)
Information_Ratio = step5_excess_returns_cum.div(Forecast_errors.std())


############################################################################################################################
### OOS

# Calculate own bm ratio timeseries for each PCs and MKT and create df

step5_bm_df_test = {}

for idx, pc in enumerate(PCs):
    step5_bm_df_test[pc] = np.dot(pc_eigenv_df.iloc[idx,:].values, bm_df_test.transpose())[:-1]

step5_bm_df_test = pd.DataFrame(step5_bm_df_test).set_index(bm_df_test.index[:-1])

step5_bm_df_test['MKT'] = np.log(market_bm_test[:-1]) # correct?


# modify PC and MKT return df from pca
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


# Calculate forecast errors 
step5_forecast_errors_test = abs(step5_predicted_ret_df_test - step5_return_df_test_pca.iloc[1:,:])


# Calculate estimate of conditional Covariance matrix
step5_cov_matrix_test = step5_predicted_ret_df_test.cov()   # conditional?


# Calculate portfolio weights
step5_weights_test = np.dot(np.linalg.inv(step5_cov_matrix_test), step5_predicted_ret_df_test.transpose()).transpose()
step5_weights_test = pd.DataFrame(step5_weights_test, index=step5_retc_df_test.index, columns=step5_retc_df_test.columns)

sum_weights_test = step5_weights_test.sum(axis=1)


# rescale weights
step5_weights_test_scaled = step5_weights_test.apply(lambda x: x/x.sum(), axis=1)


#Return strategy and market
step5_final_returns_test = np.multiply(step5_weights_test_scaled, step5_return_df_test_pca_adj)
step5_sum_final_returns_test = step5_final_returns_test.sum(axis=1)
step5_sum_final_returns_test = step5_sum_final_returns_test.to_frame('ret')

step5_cumulative_market_returns_test = np.cumprod(1 + market_returns_test['ret'].values) - 1
step5_cumulative_market_returns_test = pd.DataFrame(step5_cumulative_market_returns_test)
step5_cumulative_market_returns_test = step5_cumulative_market_returns_test.head(-1)
step5_cumulative_market_returns_test = step5_cumulative_market_returns_test.tail(-1)
step5_cumulative_market_returns_test.index = step5_final_returns_test.index


step5_cumulative_returns_test = np.cumprod(1 + step5_sum_final_returns_test.values) - 1
step5_cumulative_returns_test = pd.DataFrame(step5_cumulative_returns_test)
step5_cumulative_returns_test.index = step5_cumulative_market_returns_test.index
    

#Sharpe Ratio assuming a risk-free rate of 0
return_df_test_pca_adj = return_df_test_pca.tail(-2)
Sharpe_Ratio_prelim_test = np.dot(step5_final_returns_test.mean().transpose(), step5_cov_matrix_test)
Sharpe_Ratio_test = np.dot(Sharpe_Ratio_prelim_test, return_df_test_pca_adj.mean())


#Information Ratio
step5_excess_returns_test = step5_cumulative_returns_test.subtract(step5_cumulative_market_returns_test)
step5_excess_returns_cum_test = step5_cumulative_returns_test.subtract(step5_cumulative_market_returns_test)
step5_excess_returns_cum_test = step5_excess_returns.iloc[261]
step5_excess_returns_cum_test = step5_excess_returns_cum_test.to_frame('ret')
market_returns_test_adj = market_returns_test.tail(-2)
market_returns_test_adj.index = step5_final_returns_test.index

Forecast_errors_test = step5_sum_final_returns_test.subtract(market_returns_test_adj, axis = 1)
Information_Ratio_test = step5_excess_returns_cum_test.div(Forecast_errors_test.std())



##################### Plots #####################


#Plotting predicted vs actual PC returns IS & OOS
PC_List = pd.DataFrame(columns = ['PC1','PC2', 'PC3', 'PC4', 'PC5'])
return_df_pca = pd.concat([return_df_train_pca, return_df_test_pca], axis = 0)

for anom in PC_List:
    
    return_df_pca_adj = return_df_pca.tail(-2)
    return_df_pca_adj_cum = np.cumprod(1 + return_df_pca_adj[anom].values) - 1
    return_df_pca_adj_cum = pd.DataFrame(return_df_pca_adj_cum)
    return_df_pca_adj_cum.index = step5_predicted_ret_df.index
    
    step5_predicted_ret_df_cum = np.cumprod(1 + step5_predicted_ret_df[anom].values) - 1
    step5_predicted_ret_df_cum = pd.DataFrame(step5_predicted_ret_df_cum)
    step5_predicted_ret_df_cum.index = step5_predicted_ret_df.index
    
    step5_predicted_ret_df_test_cum = np.cumprod(1 + step5_predicted_ret_df_test[anom].values) - 1
    step5_predicted_ret_df_test_cum = pd.DataFrame(step5_predicted_ret_df_test_cum)
    step5_predicted_ret_df_test_cum.index = step5_predicted_ret_df_test.index
    
    PC_Graph_list = [return_df_pca_adj_cum, step5_predicted_ret_df_cum, step5_predicted_ret_df_test_cum]
    PC_Graph = pd.concat(PC_Graph_list, axis=1)
    PC_Graph.columns = ['Market', 'Strategy IS', 'Strategy OOS']
    PC_Graph.index=step5_predicted_ret_df.index
    
    plt.rcParams["figure.autolayout"] = True
    PC_Graph.plot(figsize=(10,6),title= anom+' Returns Actual vs. Predicted', xlabel = 'Date', ylabel = 'Return in %')


#Plotting predicted vs market returns for whole strategy
Returns_Graph_list = [step5_cumulative_returns, step5_cumulative_market_returns, step5_cumulative_returns_test]
Returns_Graph = pd.concat(Returns_Graph_list, axis=1)
Returns_Graph.columns = ['Strategy IS', 'Market', 'Strategy OOS']
plt.rcParams["figure.autolayout"] = True
Returns_Graph.plot(figsize=(10,6),title='Strategy vs. Market Return', xlabel = 'Date', ylabel = 'Return in %')

#Plotting predicted vs market returns only OOS
Returns_Graph_list = [step5_cumulative_market_returns_test, step5_cumulative_returns_test]
Returns_Graph = pd.concat(Returns_Graph_list, axis=1)
Returns_Graph.columns = ['Market', 'Strategy OOS']
plt.rcParams["figure.autolayout"] = True
Returns_Graph.plot(figsize=(10,6),title='Strategy vs. Market Return OOS', xlabel = 'Date', ylabel = 'Return in %')

#Plotting monthly and cumulative market returns
step5_cumulative_market_returns.columns = ['ret']
step5_cumulative_market_returns.plot(figsize=(10,6),title='Cumulative Market Return in sample', xlabel = 'Date', ylabel = 'Return in %')
market_returns.plot(figsize=(10,6),title='Monthly Market Returns in sample', xlabel = 'Date', ylabel = 'Return in %')






