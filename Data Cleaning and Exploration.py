#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset
file_path = 'global_air_quality.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
data.head()


# In[2]:


# Identify missing values
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)

# Handle missing values (example: fill with mean for numerical columns)
data.fillna(data.mean(), inplace=True)

# Verify that missing values have been handled
print("Missing values after handling:\n", data.isnull().sum())


# In[3]:


from scipy import stats
import numpy as np

# Detect outliers using Z-score for numerical columns
z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))

# Remove rows with outliers
data = data[(z_scores < 3).all(axis=1)]

# Verify the shape of the data after removing outliers
print("Data shape after removing outliers:", data.shape)


# In[4]:


# Summarize key statistics
summary = data.describe()
print("Summary statistics:\n", summary)


# In[12]:


import pandas as pd
import matplotlib.pyplot as plt

# Load your data into a DataFrame (update the file path and loading method as needed)
# For example: data = pd.read_csv('your_dataset.csv')

# Verify the DataFrame is loaded correctly
print(data.head())

# Define the correct column names
date_column = 'Start_Date'
air_quality_column = 'Data Value'

# Check if the columns exist in the DataFrame
if date_column not in data.columns:
    raise KeyError(f"Column '{date_column}' not found in DataFrame.")
if air_quality_column not in data.columns:
    raise KeyError(f"Column '{air_quality_column}' not found in DataFrame.")

# Convert the date column to datetime
data[date_column] = pd.to_datetime(data[date_column])

# Time series visualization
plt.figure(figsize=(10, 6))
data.groupby(date_column)[air_quality_column].mean().plot()
plt.title('Air Quality Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Air Quality Index')
plt.show()


# In[17]:


print(data.head())


# In[18]:


# Example: Assuming 'Geo Join ID' represents latitude and 'Geo Place Name' represents longitude
data = data.rename(columns={'Geo Join ID': 'latitude', 'Geo Place Name': 'longitude', 'Data Value': 'air_quality'})


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt

# Define geographical column names
latitude_column = 'latitude'  # Replace with the correct latitude column name
longitude_column = 'longitude'  # Replace with the correct longitude column name
air_quality_column = 'air_quality'  # Replace with the correct air quality column name

# Geographical visualization
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data, x=longitude_column, y=latitude_column, hue=air_quality_column, palette='coolwarm')
plt.title('Air Quality Across Different Regions')
plt.show()


# In[21]:


# Identify missing values
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)

# Handle missing values (example: fill with mean for numerical columns)
data.fillna(data.mean(), inplace=True)

# Verify that missing values have been handled
print("Missing values after handling:\n", data.isnull().sum())


# In[22]:


from scipy import stats
import numpy as np

# Detect outliers using Z-score for numerical columns
z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))

# Remove rows with outliers
data_cleaned = data[(z_scores < 3).all(axis=1)]

# Verify the shape of the data after removing outliers
data_cleaned_shape = data_cleaned.shape

data_cleaned_shape


# In[23]:


# Use a less strict Z-score threshold, e.g., 4
z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))

# Remove rows with outliers using the adjusted threshold
data_cleaned = data[(z_scores < 4).all(axis=1)]

# Verify the shape of the data after removing outliers
data_cleaned_shape = data_cleaned.shape

data_cleaned_shape


# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt

# Investigate the distribution of numerical columns
numerical_columns = data.select_dtypes(include=[np.number]).columns

for col in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


# In[25]:


from scipy import stats
import numpy as np

# Use a less strict Z-score threshold, e.g., 4
z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))

# Remove rows with outliers using the adjusted threshold
data_cleaned = data[(z_scores < 4).all(axis=1)]

# Verify the shape of the data after removing outliers
data_cleaned_shape = data_cleaned.shape

data_cleaned_shape


# In[26]:


import seaborn as sns
import matplotlib.pyplot as plt

# Investigate the distribution of numerical columns
numerical_columns = data.select_dtypes(include=[np.number]).columns

for col in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


# In[ ]:




