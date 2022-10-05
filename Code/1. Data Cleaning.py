#loading data into VS
# import necessary libraries
from statistics import mean
import pandas as pd
import os
import glob

#set different paths
pathn10 = '/Users/lasuc/Desktop/QAM-Factor-Timing/Data/FT_portfolio_sorts-monthly-05FEB2020/monthlyn10'
pathbmc10 = '/Users/lasuc/Desktop/QAM-Factor-Timing/Data/FT_portfolio_sorts-monthly-05FEB2020/monthlybmc10'
pathret10 = '/Users/lasuc/Desktop/QAM-Factor-Timing/Data/FT_portfolio_sorts-monthly-05FEB2020/monthlyret10'
pathtotret10 = '/Users/lasuc/Desktop/QAM-Factor-Timing/Data/FT_portfolio_sorts-monthly-05FEB2020/monthlytotret10'

#get all the direct data paths for the different .csv files, create a list for loops at the end
csv_filesn10 = glob.glob(os.path.join(pathn10, "*.csv"))
csv_filesbmc10 = glob.glob(os.path.join(pathbmc10, "*.csv"))
csv_filesret10 = glob.glob(os.path.join(pathret10, "*.csv"))
csv_filestotret10 = glob.glob(os.path.join(pathtotret10, "*.csv"))
csv_li = [csv_filesn10, csv_filesbmc10, csv_filesret10, csv_filestotret10]


bmc10_accruals = pd.read_csv(r'/Users/lasuc/Desktop/QAM-Factor-Timing/Data/FT_portfolio_sorts-monthly-05FEB2020/monthlybmc10/bmc10_accruals.csv')
bmc10_accruals['date'] = pd.to_datetime(bmc10_accruals['date'])
bmc10_accruals = bmc10_accruals.set_index(bmc10_accruals['date'])
bmc10_accruals = bmc10_accruals.sort_index()
bmc10_accruals = bmc10_accruals.iloc[:, 1 :]

#loop through csv_li to loop through different .csv files, the results get appended to the list, where it can be pulled from (i.e. li_bmc10[0])
li_n10 = []
li_bmc10 = []
li_ret10 = []
li_totret10 = []
li_li = [li_n10, li_bmc10, li_ret10, li_totret10]

for f in csv_li:
    for filename in f:
        filename = pd.read_csv(filename)    #read csv
        filename['date'] = pd.to_datetime(filename['date'])     #get date
        filename = filename.set_index(filename['date'])     #set date index
        filename = filename.sort_index()    #sort ascending
        filename = filename.iloc[:, 1 :]    #drop date column, since we now have indexed dates
        li_li[csv_li.index(f)].append(filename)

#next to do: fix loop, so li_ret10 and li_totret10 also get the according files (currently empty)
#afterwards: solit up the data: training set (first half of original data frame), normal reproduction OOS (till 12.17) und new OOS (12.19)



train = df['2015-01-10':'2016-12-20']
test  = df['2016-12-21':]
print('Train Dataset:',train.shape)
print('Test Dataset:',test.shape)


