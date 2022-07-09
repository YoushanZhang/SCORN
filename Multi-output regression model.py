#!/usr/bin/env python
# coding: utf-8

# In[101]:


#Junhui Li, Liangdong Guo and Youshan Zhang
#SCORN: Sinter Composition Optimization with Regressive Convolutional Neural Network
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')

from sklearn import metrics
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.io import loadmat
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
from sklearn.metrics import mean_squared_error # MSE
from sklearn.metrics import mean_absolute_error # MAE
from sklearn.metrics import r2_score # R square
from sklearn.model_selection import KFold
from math import sqrt


# In[102]:


## load data
data = pd.read_csv("SCORNdata.csv",header=None)
print(data.shape)
## variables Preparation
#testY=pd.read_csv("3.csv",header=None)
#testY=testY.T
#print(testY.shape)
#testX = pd.read_csv("2.csv",header=None)
#print(testX.shape)


# In[103]:


## Five-fold validation
D=data.to_numpy();
k=5;
np.random.shuffle(D)
dataset = np.array_split(D, k)
for i in range(1):

        train_set = dataset.copy()
##
        test_set = train_set.pop(i)

        train_set = np.vstack(train_set)
        train_set=np.array(train_set)
        test_set=np.array(test_set)
        trainX=train_set[:,9:10]
        print(trainX.shape)
        trainY=train_set[:,0:9]
        print(trainY.shape)
        testX=test_set[:,9:10]
        print(testX.shape)
        testY=test_set[:,0:9]
        print(testY.shape)


# In[104]:


# RandomForest model
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
Rmse=np.empty(5)
mae=np.empty(5)
R2=np.empty(5)
for i in range(k):

        train_set = dataset.copy()

        test_set = train_set.pop(i)

        train_set = np.vstack(train_set)
        train_set=np.array(train_set)
        test_set=np.array(test_set)
#####---------------------------------------   
        trainX=train_set[:,9:10]
        trainY=train_set[:,0:9]
        validationX=test_set[:,9:10]
        validationY=test_set[:,0:9]
#####---------------------------------------         
        y_vali=validationY.T
        #print(y_test.shape)
        model.fit(trainX,trainY)
        y_random=model.predict(validationX)
        y_random=y_random.T
        #print(ytestt)
        Rmse[i-1]=sqrt(mean_squared_error(y_vali,y_random))
        mae[i-1]=mean_absolute_error(y_vali,y_random)
        R2[i-1]=r2_score(y_vali,y_random)
        
print(np.mean(Rmse))
print(np.mean(mae))
print(np.mean(R2))


# In[105]:


#KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

knn = KNeighborsRegressor()
regr = MultiOutputRegressor(knn)

Rmse=np.empty(5)
mae=np.empty(5)
R2=np.empty(5)
for i in range(k):

        train_set = dataset.copy()

        test_set = train_set.pop(i)

        train_set = np.vstack(train_set)
        train_set=np.array(train_set)
        test_set=np.array(test_set)
#####---------------------------------------   
        trainX=train_set[:,9:10]
        trainY=train_set[:,0:9]
        validationX=test_set[:,9:10]
        validationY=test_set[:,0:9]
#####---------------------------------------         
        y_vali=validationY.T
        #print(y_test.shape)
        regr.fit(trainX,trainY)
        y_knn=regr.predict(validationX)
        y_knn=y_knn.T
        #print(ytestt)
        Rmse[i-1]=sqrt(mean_squared_error(y_vali,y_knn))
        mae[i-1]=mean_absolute_error(y_vali,y_knn)
        R2[i-1]=r2_score(y_vali,y_knn)
        
print(np.mean(Rmse))
print(np.mean(mae))
print(np.mean(R2))


# In[106]:


Rmse=np.empty(5)
mae=np.empty(5)
R2=np.empty(5)
for i in range(k):

        train_set = dataset.copy()

        test_set = train_set.pop(i)

        train_set = np.vstack(train_set)
        train_set=np.array(train_set)
        test_set=np.array(test_set)
#####---------------------------------------   
        trainX=train_set[:,9:10]
        trainY=train_set[:,0:9]
        validationX=test_set[:,9:10]
        validationY=test_set[:,0:9]
        
        y_vali=validationY.T
#least squre
        train_xadd = sm.add_constant(trainX)  ## 添加常数项
        lm = sm.OLS(trainY,train_xadd).fit()
        validationxadd = sm.add_constant(validationX)  ## 添加常数项
        y_ols = lm.predict(validationxadd)
        y_ols=y_ols.T
        Rmse[i-1]=sqrt(mean_squared_error(y_vali,y_ols))
        mae[i-1]=mean_absolute_error(y_vali,y_ols)
        R2[i-1]=r2_score(y_vali,y_ols)
        
print(np.mean(Rmse))
print(np.mean(mae))
print(np.mean(R2))


# In[107]:


#DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
regr_1 = DecisionTreeRegressor()

Rmse=np.empty(5)
mae=np.empty(5)
R2=np.empty(5)
for i in range(k):

        train_set = dataset.copy()

        test_set = train_set.pop(i)

        train_set = np.vstack(train_set)
        train_set=np.array(train_set)
        test_set=np.array(test_set)
#####---------------------------------------   
        trainX=train_set[:,9:10]
        trainY=train_set[:,0:9]
        validationX=test_set[:,9:10]
        validationY=test_set[:,0:9]
#####---------------------------------------         
        y_vali=validationY.T
        #print(y_test.shape)
        regr_1.fit(trainX,trainY)
        y_De=regr_1.predict(validationX)
        y_De=y_De.T
        #print(ytestt)
        Rmse[i-1]=sqrt(mean_squared_error(y_vali,y_De))
        mae[i-1]=mean_absolute_error(y_vali,y_De)
        R2[i-1]=r2_score(y_vali,y_De)
        
print(np.mean(Rmse))
print(np.mean(mae))
print(np.mean(R2))        


# In[108]:


# Multi-output SVR
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.datasets import make_regression
from sklearn.multioutput import RegressorChain
from sklearn.svm import LinearSVR
modelSVR = LinearSVR()
wrapper2 = RegressorChain(modelSVR)

Rmse=np.empty(5)
mae=np.empty(5)
R2=np.empty(5)
for i in range(k):

        train_set = dataset.copy()

        test_set = train_set.pop(i)

        train_set = np.vstack(train_set)
        train_set=np.array(train_set)
        test_set=np.array(test_set)
#####---------------------------------------   
        trainX=train_set[:,9:10]
        trainY=train_set[:,0:9]
        validationX=test_set[:,9:10]
        validationY=test_set[:,0:9]
#####---------------------------------------         
        y_vali=validationY.T
        #print(y_test.shape)
        wrapper2.fit(trainX,trainY)
        y_SVR=wrapper2.predict(validationX)
        y_SVR=y_SVR.T
        #print(ytestt)
        Rmse[i-1]=sqrt(mean_squared_error(y_vali,y_SVR))
        mae[i-1]=mean_absolute_error(y_vali,y_SVR)
        R2[i-1]=r2_score(y_vali,y_SVR)
        
print(np.mean(Rmse))
print(np.mean(mae))
print(np.mean(R2)) 


# In[109]:


# MLP
mlpr = MLPRegressor(hidden_layer_sizes=(20,50,100), 
                    activation='tanh', 
                    solver='adam', 
                    alpha=0.0001,   
                    max_iter=300, 
                    random_state=123,
#                     early_stopping=True, ## 
#                     validation_fraction=0.2, ##
#                     tol=1e-8,
                   )
Rmse=np.empty(5)
mae=np.empty(5)
R2=np.empty(5)
for i in range(k):

        train_set = dataset.copy()

        test_set = train_set.pop(i)

        train_set = np.vstack(train_set)
        train_set=np.array(train_set)
        test_set=np.array(test_set)
#####---------------------------------------   
        trainX=train_set[:,9:10]
        trainY=train_set[:,0:9]
        validationX=test_set[:,9:10]
        validationY=test_set[:,0:9]
#####---------------------------------------         
        y_vali=validationY.T
        #print(y_test.shape)
        mlpr.fit(trainX,trainY)
        y_mlp=mlpr.predict(validationX)
        y_mlp=y_mlp.T
        #print(ytestt)
        Rmse[i-1]=sqrt(mean_squared_error(y_vali,y_mlp))
        mae[i-1]=mean_absolute_error(y_vali,y_mlp)
        R2[i-1]=r2_score(y_vali,y_mlp)
        
print(np.mean(Rmse))
print(np.mean(mae))
print(np.mean(R2))         


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




