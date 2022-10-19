''' 1. Data Cleaning '''

# import necessary libraries
import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
from statistics import mean


### Read data in
# set different paths
pathn10 = '../Data/FT_portfolio_sorts-monthly-05FEB2020/monthlyn10'
pathbmc10 = '../Data/FT_portfolio_sorts-monthly-05FEB2020/monthlybmc10'
pathret10 = '../Data/FT_portfolio_sorts-monthly-05FEB2020/monthlyret10'
pathtotret10 = '../Data/FT_portfolio_sorts-monthly-05FEB2020/monthlytotret10'

# get all the direct data paths for the different .csv files, create a list for loops at the end
csv_filesn10 = glob.glob(os.path.join(pathn10, "*.csv"))
csv_filesbmc10 = glob.glob(os.path.join(pathbmc10, "*.csv"))
csv_filesret10 = glob.glob(os.path.join(pathret10, "*.csv"))
csv_filestotret10 = glob.glob(os.path.join(pathtotret10, "*.csv"))
csv_li = [csv_filesn10, csv_filesbmc10, csv_filesret10, csv_filestotret10]

bmc10_accruals = pd.read_csv(r'../Data/FT_portfolio_sorts-monthly-05FEB2020/monthlybmc10/bmc10_accruals.csv')
bmc10_accruals['date'] = pd.to_datetime(bmc10_accruals['date'])
bmc10_accruals = bmc10_accruals.set_index(bmc10_accruals['date'])
bmc10_accruals = bmc10_accruals.sort_index()
bmc10_accruals = bmc10_accruals.iloc[:, 1 :]

# loop through csv_li to loop through different .csv files, the results get appended to the list, where it can be pulled from (i.e. li_bmc10[0])
li_n10 = []
li_bmc10 = []
li_ret10 = []
li_totret10 = []
li_li = [li_n10, li_bmc10, li_ret10, li_totret10]

df_names = []                                               # list containing df names -> mapping index with name -> easier to access specific df from a li_ list

for file in csv_filesn10:
    datafile = pd.read_csv(file)                            #read csv
    datafile['date'] = pd.to_datetime(datafile['date'])     #get date
    datafile = datafile.set_index(datafile['date'])         #set date index
    datafile = datafile.sort_index()                        #sort ascending
    datafile = datafile.iloc[:, 1 :]                        #drop date column, since we now have indexed dates
    datafile = datafile.loc['1974-01-01':'2019-12-01']
    fname = Path(file).stem.split("_",1)[1]
    df_names.append(fname)                                  # mapping list index with name
    locals()[fname] = datafile                              # naming the df according to csv name
    li_n10.append(locals()[fname])                          # adding df to list

for file in csv_filesbmc10:
    datafile = pd.read_csv(file)                        
    datafile['date'] = pd.to_datetime(datafile['date'])     
    datafile = datafile.set_index(datafile['date'])        
    datafile = datafile.sort_index()                        
    datafile = datafile.iloc[:, 1 :] 
    datafile = datafile.loc['1974-01-01':'2019-12-01']                       
    fname = Path(file).stem
    locals()[fname] = datafile
    li_bmc10.append(locals()[fname])
    
for file in csv_filesret10:
    datafile = pd.read_csv(file)                        
    datafile['date'] = pd.to_datetime(datafile['date'])     
    datafile = datafile.set_index(datafile['date'])       
    datafile = datafile.sort_index()                      
    datafile = datafile.iloc[:, 1 :]
    datafile = datafile.loc['1974-01-01':'2019-12-01']                       
    fname = Path(file).stem
    locals()[fname] = datafile
    li_ret10.append(locals()[fname])
    
for file in csv_filestotret10:
    datafile = pd.read_csv(file)                        
    datafile['date'] = pd.to_datetime(datafile['date'])     
    datafile = datafile.set_index(datafile['date'])         
    datafile = datafile.sort_index()                       
    datafile = datafile.iloc[:, 1 :]    
    datafile = datafile.loc['1974-01-01':'2019-12-01']                   
    fname = Path(file).stem
    locals()[fname] = datafile
    li_totret10.append(locals()[fname])
    
        
### Constructing long-short anomalies: return p10 - p1
ls_df = {}

for idx, anom in enumerate(li_ret10):
    name = df_names[idx]
    diff = anom.p10 - anom.p1   
    ls_df[name] = diff
    
ls_df = pd.DataFrame(ls_df)
    

### Constructing measure of relative valuation based on book-to-market ratios
# bm = log book-to-market ratio pf10 - p1
bm_df = {}

for idx, anom in enumerate(li_bmc10):
    name = df_names[idx]
    diff = np.log(anom.p10) - np.log(anom.p1)   
    bm_df[name] = diff
    
bm_df = pd.DataFrame(bm_df)

#%%
### Removing anomalies with NaN values
print('Dropping the following anomalies is long-short portfolios: ', ls_df.columns[ls_df.isna().any()].tolist())
print('Dropping the following anomalies is b/m portfolios: ', bm_df.columns[bm_df.isna().any()].tolist(), '\n')

ls_df = ls_df.dropna(axis='columns')
bm_df = bm_df.dropna(axis='columns')

#%% COULD BE DELETED

### Market adjust data
# check which anomaly has max stocks in pf to calc market returns (siehe check_max.xlsx)
sums = []

for a in li_n10:
    s = a.sum(axis=1)
    ss = s.sum()
    sums.append(ss)
    
anom_maxsum = max(sums)
anom_maxsumidx = sums.index(anom_maxsum)  # -> index: 1 (and others but use that)

#%% NEW MARKET DATA
'''
market_returns = pd.read_excel(r'../Data/mkt_data.xlsx', sheet_name='mkt_ret', index_col=0)
market_bm = pd.read_excel(r'../Data/mkt_data.xlsx', sheet_name='mkt_bm', index_col=0)

# index to date

# choose subset by dates

# hi10 - lo10

'''
#%% OLD MARKET DATA
# Compute regression beta w.r.t. aggregate market returns
# split up the data: training set (first half of original data frame), normal reproduction OOS (till 12.17) und new OOS (12.19)
# each 264 data points
market_returns = pd.read_excel(r'../Data/market_calcs.xlsx', sheet_name='r_mkt', index_col=0)
market_bm = np.log(pd.read_excel(r'../Data/market_calcs.xlsx', sheet_name='bm_mkt', index_col=0))

ls_df_train = ls_df['1974-01-01':'1995-12-01']
ls_df_test  = ls_df['1996-01-01':'2017-12-01']
market_returns_train = market_returns['1974-01-01':'1995-12-01']
market_returns_test = market_returns['1996-01-01':'2017-12-01']
market_bm_train = market_bm['1974-01-01':'1995-12-01']
market_bm_test = market_bm['1996-01-01':'2017-12-01']

#%%
# betas and var estimated using train sample s.t. OOS statistics contain no look-ahead bias
betas = []      # for each anomaly
ls_df_ma = {}   # market-adjustet df

# betas
for idx, anom in enumerate(ls_df_train):
    beta = ls_df_train[anom].cov(market_returns_train.ret)/market_returns_train.ret.var()
    betas.append(beta)

# market-scaled ls_df
for idx, anom in enumerate(ls_df):
    ls_df_ma[ls_df.columns[idx]] = ls_df[anom] - betas[idx]*market_returns.ret

ls_df_ma = pd.DataFrame(ls_df_ma)


# market-scaled bm_df
bm_df_ma = {}   # market-adjustet df

for idx, anom in enumerate(bm_df):
    bm_df_ma[bm_df.columns[idx]] = bm_df[anom] - betas[idx]*market_bm.bm

bm_df_ma = pd.DataFrame(bm_df_ma)   


### Rescale data
# rescale market-adj returns and bm ratios s.t. they have equal variance across anomalies
# done by dividing returns and bm for each anomaly by its std.deviation s.t. all std.dev are 1

#only using test set for Var
ls_df_train_adj = ls_df_ma['1974-01-01':'1995-12-01']
for anom in ls_df_ma:
    ls_df_ma[anom] = ls_df_ma[anom] / ls_df_train_adj[anom].var()**(1/2)

bm_df_train_adj = bm_df_ma['1974-01-01':'1995-12-01']
for anom in bm_df_ma:
    bm_df_ma[anom] = bm_df_ma[anom] / bm_df_train_adj[anom].var()**(1/2)


### Train / Test data split (with new data: train until 1996-12-01!)
ls_df_ma_train = ls_df_ma['1974-01-01':'1995-12-01']
ls_df_ma_test  = ls_df_ma['1996-01-01':'2017-12-01']

bm_df_ma_train = bm_df_ma['1974-01-01':'1995-12-01']
bm_df_ma_test  = bm_df_ma['1996-01-01':'2017-12-01']

'''
#import statistics
z = ls_df_ma_train.std(axis=0)
zz = ls_df_ma_train.std(axis=1)
'''
