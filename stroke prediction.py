#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[2]:


df=pd.read_csv(r"C:/Users/Asus/Documents/Datasets/healthcare-dataset-stroke-data.csv")
df.head()


# # Checking the uniques values in categorical column, before encoding them

# In[3]:


df['work_type'].unique()


# In[4]:


df['Residence_type'].unique()


# In[5]:


df['smoking_status'].unique()


# # Encoding categorical variables for further computation

# In[6]:


label_encoder = LabelEncoder()
df['gender_encoded'] = label_encoder.fit_transform(df['gender'])
df.drop('gender', axis=1, inplace=True)

df['married']=label_encoder.fit_transform(df['ever_married'])
df.drop('ever_married', axis=1, inplace=True)

df.head()


# In[7]:


df['Residence'] = label_encoder.fit_transform(df['Residence_type'])
df.drop('Residence_type', axis=1, inplace=True)
df.head()


# In[8]:


df['smoking'] = label_encoder.fit_transform(df['smoking_status'])
df.drop('smoking_status', axis=1, inplace=True)
df.head()


# In[9]:


df['work']= label_encoder.fit_transform(df['work_type'])
df.drop('work_type', axis=1, inplace=True)
df.head()


# In[10]:


df.head()


# In[11]:


df.tail()


# # Filling missing values for BMI column

# since the 'bmi' column has only 201 missing values out of 4909 total entries, and missingness seems relatively random, imputing with the mean or median might be a reasonable option as it's simple and won't drastically alter the distribution of the data

# In[12]:


df['bmi'].describe()


# In[13]:


missing_values = df.isnull().sum()
missing_values


# **We are checking the distribution of the data to understand whether to use mean() or median()**

# In[14]:


warnings.filterwarnings("ignore")
# Plot histogram
plt.figure(figsize=(8, 6))
sns.histplot(df['bmi'], kde=True, color='skyblue')
plt.title('Distribution of BMI')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()

# Plot kernel density plot
plt.figure(figsize=(8, 6))
sns.kdeplot(df['bmi'], shade=True, color='skyblue')
plt.title('Kernel Density Estimation of BMI')
plt.xlabel('BMI')
plt.ylabel('Density')
plt.show()


# **Considering that BMI (Body Mass Index) values may not follow a perfectly normal distribution and could potentially have outliers, using the median for imputation could be more robust. It ensures that the imputed values are not heavily influenced by extreme BMI values, providing a more representative estimate of the central tendency for this variable.**

# In[15]:


median_bmi = df['bmi'].median()
df['bmi'].fillna(median_bmi, inplace=True)
df.head()


# # Heat map to check which variables are strongly correlated 

# In[16]:


correlation_matrix=df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()


# # Checking the correlation of stroke with other attributes

# In[17]:


stroke_correlation = df.corr()['stroke'].sort_values(ascending=False)

print("Correlation of 'stroke' with other variables:")
print(stroke_correlation)


# In[61]:


import plotly.graph_objects as go

# Create a bar plot with Plotly
fig = go.Figure(go.Bar(
    x=df['Correlation_with_stroke'],
    y=df['Variable'],
    orientation='h'
))

fig.update_layout(
    title='Correlation of Variables with Stroke',
    xaxis_title='Correlation with Stroke',
    yaxis_title='Variable',
    yaxis=dict(autorange="reversed")  # Invert y-axis to show highest correlation at the top
)

fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




