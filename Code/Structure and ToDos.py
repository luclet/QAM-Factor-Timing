#ÜBERSICHT AN NOTES, TODOS (coding-bezogen)

#Notes:
#For each Anomaly strategy, we should have the relative valuation
#in the form of B/M ratio of unterlying stocks.
#bm is the difference of B/M between Port 10 and Port 1

###
#Proposed Code To Dos:
#1.
#-Get acustomed to data
#-load data correctly into python
#-divide data into first half (Jan 74 - Jun 95) / training set for OOS (Jul 95 - Dec 2017) and third section with second half+rest (Jul 95-Dec 19)
#-calculate market beta of portfolios with the market return portfolio
#-market adjust data returns and bm ratios? (if not done already, check if data comes adjusted)
#-rescale market-adjusted returns and bm ratios, so the variance is equal across anomalies
#2.
#-Create Q, q from cov(F_t+1)=Q􏰙Q′
#-Construct PCs of 50 anomaly Portfolios (should be that each PC is for all of the anomalies)
#-
#3.
#-prediction via bm_i,t = q_i′*bm_Ft
