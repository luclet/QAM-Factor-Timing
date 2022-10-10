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
'''
NEXT STEPS:
    - PC analysis
    - ...
'''

#from sklearn.decomposition import PCA
#pca_bm_df_ma_train = PCA(n_components=5)
#principalComponents_bm_df_ma_train = pca_bm_df_ma_train.fit_transform(x)


    # Dominant components of factors 
    # ->  We have 55, paper has 50?

### 3. Prediciting the large PCs of anomaly returns

### 4. Prediciting individual factors

### 5. Optimal factor timing portfolio