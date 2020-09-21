#!/usr/bin/env python
# coding: utf-8

# # Prediction of the percentage of marks that a student is expected to score based upon the number of hours they studied.

# In[1]:


#Importing libraries 
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Load data
data = pd.read_csv('http://bit.ly/w-data')


# In[3]:


data


# In[4]:


data.head(5)


# In[6]:


# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('% Score')  
plt.show()


# In[7]:


#preparing the data
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


# In[8]:


#Spliting data into training and testing data
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


# In[9]:


#Training
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[26]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y, color = '#db0d0d')
plt.plot(X, line);
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Regression plot between hours and scores")
plt.show()


# In[11]:


# Testing data - In Hours
print(X_test) 

 # Predicting the scores
y_pred = regressor.predict(X_test)


# In[12]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[23]:


graph = df.head()
graph.plot(kind='bar',figsize=(30,5))
plt.title('Actual vs Predicted')
plt.grid(which = 'both', color='red', linestyle='-.', linewidth=0.5)
plt.show()


# In[20]:


#testing with own data
hours = [[9.25]]
pred = regressor.predict(hours)
print ("Number of hours :{}".format(hours))
print ("Predicted Score :{}".format(pred))


# In[19]:


#Evaluating the model
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# ### Predicted score for 9.25 hrs is:

# In[21]:


print ("Predicted Score :{}".format(pred))


# In[ ]:




