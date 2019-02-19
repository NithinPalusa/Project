#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
import dask.dataframe as dd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score,roc_curve,recall_score,classification_report,mean_squared_error,confusion_matrix
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import lightgbm as lgb
from contextlib import contextmanager
import time
import threading
import random
from sklearn.model_selection import train_test_split as model_tts
print(os.listdir("data"))


# In[2]:


# preset the data types
dtyp = {'ip': np.int64, 'app': np.int16,'device': np.int16,'os': np.int16,'channel': np.int16,'is_attributed' : np.int16}


# In[86]:


print("LOADING DATA..........................")
# TRAINING DATA
print("TRAINING DATA")

data = pd.read_csv("data/train.csv", nrows=1500000)
#need to skip 0th row as it is the header
print("Loading Completed")


# In[89]:


#check_true_positive_ratio(dfTest)
#dfTest = cleaning_transforming(dfTest)


# In[4]:


print("original dataframe")
data.head()



def cleaning_transforming(dataframe):
    #this column is completely blank
    del dataframe['attributed_time']
    # Create new features out of time. Year and month are skipped as the data is only for 4 days
    dataframe['click_time'] =  dd.to_datetime(dataframe['click_time'])

    # the given data is of 4 days. So useful data is day and hours
    dataframe['day'] = dataframe['click_time'].dt.day
    dataframe['hour'] = dataframe['click_time'].dt.hour
    del dataframe['click_time']

    dataframe.columns = ['ip', 'app', 'device', 'os','channel','is_attributed','day','hour']

    print("dataset columns",dataframe.columns)

    dataframe.astype(dtyp)
    print("\n\n=============================================================")
    print(dataframe.info())
    return dataframe



#Cleaning and transforming both data sets
dfTrain = cleaning_transforming(data)



# In[2]:


data_1=dfTrain[dfTrain.is_attributed==1]
data_1.describe()


# In[3]:


data_0=dfTrain[dfTrain.is_attributed==0]
data_0.describe()


# In[4]:


data_0_keep=data_0.sample(n=20000)
data_0_keep.describe()


# In[5]:


Final=pd.concat([data_1,data_0_keep])
Final.describe()


# In[6]:


nrows = len(Final)
print("Number of rows in the dataframe: ", nrows)
npositive = Final.is_attributed.sum() #since is_attributed has either 0 or 1. 1 is for positive cases
print("Number of positive cases are " + str(npositive))
nnegative = nrows - npositive
print("Number of negative cases are " + str(nnegative))


# In[7]:


feature_col=['app', 'device', 'os', 'channel','day', 'hour']
x=Final[feature_col]
y=Final.is_attributed
X_train,X_test,y_train,y_test=model_tts(x,y,test_size=0.45,random_state=2)


# In[8]:


logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)


# In[9]:


from sklearn import metrics
cnf_metric= metrics.confusion_matrix(y_test,y_pred)


# In[10]:


y_pred[0:50]


# In[11]:


from sklearn import metrics
cnf_metric_log= metrics.confusion_matrix(y_test,y_pred)


# In[12]:


cnf_metric_log


# In[13]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[14]:


# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_metric), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')




# In[15]:


print(logreg.intercept_)


# In[16]:


print(logreg.coef_)


# In[17]:


y_test


# In[18]:


z=logreg.predict_proba(X_test)[:,1]


# In[19]:


z


# In[20]:


y_pred[0:55]


# In[21]:


fpr, tpr, thresholds =roc_curve(y_test, y_pred,drop_intermediate=False)


# In[22]:


plt.figure()
##Adding the ROC
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()


# In[23]:


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=300)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred1=clf.predict(X_test)


# In[24]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred1))


# In[25]:


cnf_metric= metrics.confusion_matrix(y_test,y_pred1)


# In[26]:


cnf_metric


# In[27]:


fpr, tpr, thresholds =roc_curve(y_test, y_pred1,drop_intermediate=False)


# In[28]:


plt.figure()
##Adding the ROC
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()


# In[29]:


# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_metric), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')




# In[30]:


a=clf.predict_proba(X_test)[:,1]


# In[31]:


a


# In[32]:


y_pred


# In[33]:


df=pd.DataFrame(y_pred)
df2 = pd.DataFrame(z)


# In[34]:


#bind = pd.concat(df,df2)
df2


# In[35]:


n=pd.DataFrame(y_pred)
m=pd.DataFrame(y_test)


# In[36]:


j=pd.concat([n,m])


# In[ ]:




