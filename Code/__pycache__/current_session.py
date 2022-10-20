# coding: utf-8
runcell(0, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py')
runcell('OLD MARKET DATA', 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py')
runcell(5, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py')
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
get_ipython().run_line_magic('reset', '')
get_ipython().run_line_magic('clear', '')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
runcell(2, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
m1_est = sm.OLS(return_df_test[1:,-1], market_bm_test[:-1]).fit()
print(m1_est.summary())
return_df_test[1:,-1]
market_bm_test[:-1]
X = sm.add_constant(market_bm_test[:-1])
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
np.log(market_bm_test[:-1])
market_bm_test[:-1]
get_ipython().run_line_magic('reset', '')
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
get_ipython().run_line_magic('reset', '')
get_ipython().run_line_magic('clear', '')
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
get_ipython().run_line_magic('reset', '')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
runcell(2, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
np.log(market_bm_test[:-1])
X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
m1_est = sm.OLS(return_df_test[1:,-1], np.log(market_bm_test[:-1])).fit()
print(m1_est.summary())
np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                # sterting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, np..log(bm_df_test.transpose()))[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                # sterting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, np.log(bm_df_test.transpose()))[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                # sterting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, (bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                # sterting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
return_df_Reg = np.append(return_df_train, market_returns_train, axis=1)

# Regress 1995-2017 data of PC returns on net BM ratios
lm = LinearRegression()
model = lm.fit(return_df_Reg, bm_df_train)
r_squared = model.score(return_df_Reg, bm_df_train)

lm = LinearRegression()
model = lm.fit(X_bm_pc1, Y_ret_pc1)
print(model.coef_)
r_squared = model.score(return_df_Reg, bm_df_train)
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
lm = LinearRegression()
model = lm.fit(X_bm_pc1, Y_ret_pc1)
print(model.coef_)
r_squared = model.score(return_df_Reg, bm_df_train)
lm = LinearRegression()
model = lm.fit(Y_ret_pc1, X_bm_pc1)
print(model.coef_)
r_squared = model.score(return_df_Reg, bm_df_train)
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
runcell(2, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
runcell(2, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X = sm.add_constant(market_bm_test[:-1])
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
m1_est = sm.OLS(return_df_test[1:,-1], market_bm_test[:-1]).fit()
print(m1_est.summary())
X = sm.add_constant(market_bm_test[:-1])
#X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
get_ipython().run_line_magic('clear', '')
get_ipython().run_line_magic('reset', '')
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
X = sm.add_constant(market_bm_aggr[:-1])
#X = sm.add_constant(market_bm_test[:-1])
#X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
rest
get_ipython().run_line_magic('reset', '')
get_ipython().run_line_magic('clear', '')
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
X = sm.add_constant(market_bm_aggr_test[:-1])
#X = sm.add_constant(market_bm_test[:-1])
#X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
X = sm.add_constant(market_bm_test[:-1])
#X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
runcell(2, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X = sm.add_constant(market_bm_test[:-1])
#X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
runcell(2, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
get_ipython().run_line_magic('reset', '')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
runcell(2, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
runcell(2, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
runcell(2, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X = sm.add_constant(market_bm_test[:-1])
#X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, np.log(bm_df_test.transpose()))[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
np.log(
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
X_bm_pc2 = sm.add_constant(np.dot(pc_eigenv_df.iloc[1,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc2 = return_df_test[1:,1]                                                                # sterting at 1 (t+1)

bm_pc2_est = sm.OLS(Y_ret_pc2, X_bm_pc2).fit()
print(bm_pc2_est.summary())
X = sm.add_constant(market_bm_aggr_test[:-1])
#X = sm.add_constant(market_bm_test[:-1])
#X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
X = sm.add_constant(market_bm_test[:-1])
#X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
runcell(2, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
runcell(2, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
runcell(2, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
runcell(2, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X = sm.add_constant(market_bm_test[:-1])
#X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X = sm.add_constant(market_bm_test[:-1])
#X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
X = sm.add_constant(market_bm_aggr_test[:-1])
#X = sm.add_constant(market_bm_test[:-1])
#X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary()
X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X = sm.add_constant(market_bm_test[:-1])
#X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X = sm.add_constant(market_bm_test[:-1])
#X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
get_ipython().run_line_magic('quickref', '')
_oh
get_ipython().run_line_magic('hist', '-g main.py')
_oh
_dh
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X = sm.add_constant(market_bm_aggr_test[:-1])
#X = sm.add_constant(market_bm_test[:-1])
#X = sm.add_constant(np.log(market_bm_test[:-1]))
m1_est = sm.OLS(return_df_test[1:,-1], X).fit()
print(m1_est.summary())
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, bm_df_test.transpose())[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
runfile('C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/Data_Cleaning.py', wdir='C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code')
runcell(1, 'C:/Users/Lorena Tassone/Desktop/UZH/Master/HS 22/Quantitative Asset Management/GitHub/QAM-Factor-Timing/Code/main.py')
X_bm_pc1 = sm.add_constant(np.dot(pc_eigenv_df.iloc[0,:].values, np.log(bm_df_test.transpose()))[:-1])  # starting at 0 (t) and deleting last value
Y_ret_pc1 = return_df_test[1:,0]                                                                 # starting at 1 (t+1)

bm_pc1_est = sm.OLS(Y_ret_pc1, X_bm_pc1).fit()
print(bm_pc1_est.summary())
get_ipython().run_line_magic('history', '> history_for_print.txt')
get_ipython().run_line_magic('history', '-g -f history_for_print.txt')
get_ipython().run_line_magic('save', 'main.py _oh')
get_ipython().run_line_magic('save', 'output.txt _oh')
get_ipython().run_line_magic('save', 'current_session ~0/')
