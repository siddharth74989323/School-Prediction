#!/usr/bin/env python
# coding: utf-8

# # Machine learning do not understand Unit it understand only value
# # numpy is use for mathematical operations
# # pandas is use for data frame
# # matplotlib is use for graphysical representation
# # sklearn is use for split train and test data this is IMP
# # from sklearn.model_selection import train_test_split use to split data
# # from sklearn.linear_model import LinearRegression use to linear regression
# # joblib :- Save the linear Regression model

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


# In[2]:


path='dataset/Student_Marks.csv'
df=pd.read_csv(path)
df


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# # Discover and visulize the data to gain insights

# In[6]:


df.info() # give the info of dataframe, 


# In[7]:


df.describe() # gives the all details like mean, count, median, max, min, 25%, 50%, 75%, count, std etc.


# In[8]:


plt.scatter(x=df.student_marks,y=df.study_hours)
plt.xlabel="Student Marks"
plt.ylabel="Student Hours"
plt.title="Student Marks and Study Hours"
plt.show()


# # Data Cleaning

# In[9]:


df.isnull() # return the all the column with True and False if data if data is present then return True else false


# In[10]:


df.isnull().sum() # gives the null value from each table.


# In[11]:


df.mean() # gives then mean of each table


# In[12]:


df2=df.fillna(df.mean())


# In[13]:


df2.isnull().sum() # use to fill dataframe NaN value with mean value


# In[14]:


df2.head()


# # Split Dataframe

# In[15]:


X=df2.drop('student_marks',axis='columns') #In Machine learning X=Matrix
y=df2.drop('study_hours',axis='columns')#In Machine learning y=Vector
print("Shape of x= ",X.shape)
print("Shape of y= ",y.shape)


# # In machine learning 70% or 80% data using for Training and 30% or 20% data is using for testing

# In[16]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=51) #0.2 = 20%
print("Shape of x_train= ",X_train.shape)
print("Shape of y_train= ",y_train.shape)
print("Shape of x_test= ",X_test.shape)
print("Shape of y_test= ",y_test.shape)


# # Select the model and train it
# y= m * x + c

# In[17]:


lr=LinearRegression() # Created the object of the class LinearRegression()


# In[18]:


lr.fit(X_train,y_train) # Fit the train data of X_train and y_train to LinearRegression method


# # lr.coef_ :- Gives the value of 'm' of the formula y = m * x + c
# # lr.intercept_ :- Gives the value of 'c' of the formula y = m * x + c

# In[19]:


lr.coef_


# In[20]:


lr.intercept_


# # This is just sample mathematics operation for understand, we do not have to do it, LinearRegression have methode for this.

# In[21]:


m=3.93
c=50.44
y= m * 4 + c
y


# # This below line give the prediction by using our ML model

# In[22]:


lr.predict([[4]])[0][0].round(2)


# In[23]:


y_pred=lr.predict(X_test)


# In[24]:


pd.DataFrame(np.c_[X_test,y_test,y_pred], columns=["Study Hours","Student Marks Original","Students Marks Predicted"])


# # Fine-Tune your Model
# # lr.score(X_test,y_test) or lr.score give the accuracy of the Machine Learning model

# In[25]:


lr.score(X_test,y_test)


# In[26]:


plt.scatter(X_train,y_train)


# In[27]:


plt.scatter(X_test,y_test)
plt.plot(X_train,lr.predict(X_train),color="r")


# # Present your solution
# # Save ML model
# # Finally we create .pkl file 

# In[28]:


joblib.dump(lr,"Student_Mark_Predictor_Model.pkl")


# In[29]:


model=joblib.load("Student_Mark_Predictor_Model.pkl")


# In[30]:


model.predict([[5]])[0][0].round(2)


# # Launch Monitor and Maintain our system
