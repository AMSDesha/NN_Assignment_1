
# coding: utf-8

# In[109]:


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as TF

from sklearn.metrics import classification_report


# In[110]:


df = pd.read_csv("original.csv",names=['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16'])


# In[111]:


df.dtypes
df =df[df['A1']!='?']
df =df[df['A2']!='?']
df =df[df['A4']!='?']
df =df[df['A5']!='?']
df =df[df['A6']!='?']
df =df[df['A7']!='?']
df =df[df['A14']!='?']


# In[112]:


df.A16 = np.where(df.A16=='+',1,0)


# In[113]:


df.A16.value_counts()

df.A2 = pd.to_numeric(df.A2)
df.A3 = pd.to_numeric(df.A3)
df.A8 = pd.to_numeric(df.A8)
df.A11 = pd.to_numeric(df.A11)
df.A14 = pd.to_numeric(df.A14)
df.A15 = pd.to_numeric(df.A15)


# In[114]:


df.A1 = df.A1.astype('category').cat.codes
df.A4 = df.A4.astype('category').cat.codes
df.A5 = df.A5.astype('category').cat.codes
df.A6 = df.A6.astype('category').cat.codes
df.A7 = df.A7.astype('category').cat.codes
df.A9 = df.A9.astype('category').cat.codes
df.A10 = df.A10.astype('category').cat.codes
df.A12 = df.A12.astype('category').cat.codes
df.A13 = df.A13.astype('category').cat.codes

X = df.drop(['A16'],axis =1)
Y = df.A16


# In[115]:


Y = to_categorical(Y)


# In[116]:


def model1():
    #creat model
    model = Sequential()
    n_cols = X.shape[1]
    model.add(Dense(47,activation='relu',input_shape =(15,)))
    model.add(Dense(20,activation='sigmoid'))
    model.add(Dense(2,activation='softmax'))
    
    #compile Model
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model


# In[117]:


def model2():
    #creat model
    model = Sequential()
    n_cols = X.shape[1]
    model.add(Dense(47,activation='relu',input_shape =(15,)))
    model.add(Dense(40,activation='sigmoid'))
    model.add(Dense(2,activation='softmax'))
    
    #compile Model
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model


# In[118]:


def model3():
    #creat model
    model = Sequential()
    n_cols = X.shape[1]
    model.add(Dense(47,activation='relu',input_shape =(15,)))
    model.add(Dense(50,activation='sigmoid'))
    model.add(Dense(2,activation='softmax'))
    
    #compile Model
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model


# In[119]:


def model4():
    #creat model
    model = Sequential()
    n_cols = X.shape[1]
    model.add(Dense(47,activation='relu',input_shape =(15,)))
    model.add(Dense(60,activation='sigmoid'))
    model.add(Dense(2,activation='softmax'))
    
    #compile Model
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model


# In[148]:


def model5():
    #creat model
    model = Sequential()
    n_cols = X.shape[1]
    model.add(Dense(47,activation='relu',input_shape =(15,)))
    model.add(Dense(120,activation='sigmoid'))
    model.add(Dense(2,activation='softmax'))
    
    #compile Model
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model


# In[149]:


def model6():
    #creat model
    model = Sequential()
    n_cols = X.shape[1]
    model.add(Dense(47,activation='relu',input_shape =(15,)))
    model.add(Dense(50,activation='sigmoid'))
    model.add(Dense(30,activation='sigmoid'))
    model.add(Dense(2,activation='softmax'))
    
    #compile Model
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model


# In[150]:


def model7():
    #creat model
    model = Sequential()
    n_cols = X.shape[1]
    model.add(Dense(47,activation='relu',input_shape =(15,)))
    model.add(Dense(50,activation='sigmoid'))
    model.add(Dense(60,activation='sigmoid'))
    model.add(Dense(2,activation='softmax'))
    
    #compile Model
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model


# In[157]:


def model8():
    #creat model
    model = Sequential()
    n_cols = X.shape[1]
    model.add(Dense(47,activation='relu',input_shape =(15,)))
    model.add(Dense(50,activation='sigmoid'))
    model.add(Dense(30,activation='sigmoid'))
    model.add(Dense(10,activation='sigmoid'))
    model.add(Dense(2,activation='softmax'))
    
    #compile Model
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model


# In[158]:


def model9():
    #creat model
    model = Sequential()
    n_cols = X.shape[1]
    model.add(Dense(47,activation='relu',input_shape =(15,)))
    model.add(Dense(50,activation='sigmoid'))
    model.add(Dense(30,activation='sigmoid'))
    model.add(Dense(20,activation='sigmoid'))
    model.add(Dense(2,activation='softmax'))
    
    #compile Model
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model


# In[159]:


def model10():
    #creat model
    model = Sequential()
    n_cols = X.shape[1]
    model.add(Dense(47,activation='relu',input_shape =(15,)))
    model.add(Dense(50,activation='sigmoid'))
    model.add(Dense(30,activation='sigmoid'))
    model.add(Dense(25,activation='sigmoid'))
    model.add(Dense(2,activation='softmax'))
    
    #compile Model
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model


# In[ ]:


estimator = KerasClassifier(build_fn=model10,epochs=20, verbose=0)

kfold = KFold(n_splits =5,shuffle=True)

results = cross_val_score(estimator,X,Y, cv=kfold)
print("Accuracy : %.2f%%" % (results.mean()*100))


# In[ ]:


pred = baseline_model().predict(X)
print(classification_report(df.A16, pred.argmax(axis=1)))

cm = confusion_matrix(df.A16, pred.argmax(axis=1))
cm

