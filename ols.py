
# In[5]:

from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# In[6]:


dataset=pd.read_csv('C://Users//ADMIN//Desktop//Dataset//ML codefest//train_NIR5Yl1.csv')
dataset.head(5)
#X=dataframe[['Tag','Reputation','Answers','Views']]
#y=dataframe['Upvotes']

# In[]
#dataset = dataset.reindex(np.random.permutation(dataset.index))


# In[23]:



X = dataset.drop(['Tag','ID','Username'], axis=1)
y = X.Upvotes                    # vector of labels (dependent variable)
X=X.drop(['Upvotes'], axis=1)       # remove the dependent variable from the dataframe X

X.head(10)

# In[]
from sklearn.model_selection import train_test_split
X, x_test, y, y_test = train_test_split(X,y,test_size=.2, random_state=0)

# In[]
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
X=scalar.fit_transform(X)
x_test=scalar.fit_transform(x_test)

# In[]
import statsmodels.api as sm
#X=np.append(arr=np.ones([X.shape[0],1]),values=X,axis=1)



# In[]
#**************************************************************


from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
X=poly_reg.fit_transform(X)

poly_reg.fit(X,y)

x_test=poly_reg.fit_transform(x_test)
#classifier.fit(X_poly,y_train)

#y_pred=regressor2.predict(yy)
# In[]

model=sm.OLS(y,X).fit()
# In[]
y_pred1=model.predict(x_test)

'''**************************************************'''

# In[32]:

rms_ols=np.sqrt(mean_squared_error(y_test, y_pred1))



# In[32]:


test_dataset=pd.read_csv('C://Users//ADMIN//Desktop//Dataset//ML codefest//test_8i3B3FC.csv')
aa=test_dataset.describe()

# In[]

# In[]
X_test_new = test_dataset.drop(['Tag','ID','Username'], axis=1)


# In[]
import statsmodels.api as sm

X_test=scalar.fit_transform(X_test_new)

# In[]

#X_test=np.append(arr=np.ones([X_test.shape[0],1]),values=X_test,axis=1)

X_testing= poly_reg.fit_transform(X_test)

# In[]

y_pred_xtest=model.predict(X_testing)



# In[24]:
np.savetxt('sub5.csv',y_pred_xtest,delimiter='\t')

# In[]


# In[25]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)




# In[26]:


# In[27]:
max(accuracies)
np.mean(accuracies)
# In[29]:
