import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

file_path = 'Sleep_health_and_lifestyle_dataset.csv'
health_data = pd.read_csv(file_path, index_col=0)

# See the first 5 rows
print(health_data.head())

# See the column names
print(health_data.columns)

# Count the number of Male and Female
print(health_data.groupby("Gender").size())

# Find the average age of people who have the following conditions:
condition = (health_data['Sleep Duration'] > 7) & ( health_data['Quality of Sleep'] >= 6 ) & (health_data['BMI Category'] == 'Normal') 
print(health_data[condition]['Age'].mean())

print(health_data.dtypes)


# Add another index "sleep score" = "Sleep Duration" * "Quality of Sleep"
health_data = health_data.assign(sleep_score = health_data['Sleep Duration'] * health_data['Quality of Sleep'])
print(health_data.tail(8))
print(health_data.columns)


# Find the unique values of Occupation
unique_occu = health_data['Occupation'].unique()
print(unique_occu)


# To see the property of Stress Level
print(health_data['Stress Level'].describe())


# Trying some machine learning to predict the stress level
# Using [ features: 'Heart Rate', 'Age', 'Sleep Duration', 'Daily Steps' ] to predict 'Stress Level'
y = health_data['Stress Level']
feature_columns = ['Heart Rate', 'Age', 'Sleep Duration', 'Daily Steps']
X = health_data[feature_columns]


Stress_model = DecisionTreeRegressor(random_state=1)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
Stress_model.fit(train_X, train_y)


train_predictions = Stress_model.predict(train_X)
train_mae = mean_absolute_error(train_y, train_predictions)
print('Training Absolute Error: ', train_mae)

val_predictions = Stress_model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_predictions)
print('Prediction Absolute Error: ', val_mae)

print('Training R2 score: ', Stress_model.score(train_X, train_y))
print('Prediction R2 score: ', Stress_model.score(val_X, val_y))

