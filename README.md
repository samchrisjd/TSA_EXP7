# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 07.10.2025



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM :
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('World_Population.csv')

data['Yearly Change'] = data['Yearly Change'].astype(str).str.replace('%', '', regex=False).astype(float)

data.reset_index(inplace=True)
data.rename(columns={'index': 'Time'}, inplace=True)

population_data = data[['Time', 'Yearly Change']]
population_data.set_index('Time', inplace=True)

population_data.dropna(inplace=True)

result = adfuller(population_data['Yearly Change'])
print('\n Augmented Dickey-Fuller Test Results:')
print('ADF Statistic:', result[0])
print('p-value:', result[1])

plt.figure(figsize=(10, 4))
plot_acf(population_data['Yearly Change'], lags=20)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plt.figure(figsize=(10, 4))
plot_pacf(population_data['Yearly Change'], lags=20)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

train_size = int(len(population_data) * 0.8)
train_data = population_data.iloc[:train_size]
test_data = population_data.iloc[train_size:]

model = AutoReg(train_data['Yearly Change'], lags=5)
ar_model_fit = model.fit()
print("\n AR Model Fitted Successfully.")
print(ar_model_fit.summary())

plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data, label='Training Data', color='green')
plt.plot(test_data.index, test_data, label='Actual Test Data', color='blue')
plt.plot(test_data.index, predictions, label='Predicted Data', color='red', linestyle='dashed')
plt.title('Actual vs Predicted Yearly Change (Index as Time)')
plt.xlabel('Time Index')
plt.ylabel('Yearly Change (%)')
plt.legend()
plt.grid(True)
plt.show()

mse = mean_squared_error(test_data, predictions)
print('\n Mean Squared Error (MSE):', mse)

plt.figure(figsize=(12, 6))
plt.plot(population_data.index, population_data['Yearly Change'], label='Historical Data', color='blue')
plt.plot(range(len(population_data), len(population_data) + forecast_steps),future_forecast, label='Future Forecast', color='orange', marker='o', linestyle='dashed')
plt.title('Final Forecast of Yearly Change in World Population')
plt.xlabel('Time Index')
plt.ylabel('Yearly Change (%)')
plt.legend()
plt.grid(True)
plt.show()
```
### OUTPUT:

GIVEN DATA

<img width="981" height="265" alt="image" src="https://github.com/user-attachments/assets/e05c1b87-d2bc-45db-ac6d-e1ae67e2d919" />

Augmented Dickey-Fuller test

<img width="894" height="68" alt="image" src="https://github.com/user-attachments/assets/985768fa-f007-4ed0-bfb9-92e5ff5e7ffb" />

PACF - ACF

<img width="568" height="433" alt="download" src="https://github.com/user-attachments/assets/541e23ff-6bce-452f-8864-a3000000c070" />

<img width="568" height="433" alt="download" src="https://github.com/user-attachments/assets/fae26643-15c5-4b78-9eeb-2a307eb42841" />


PREDICTION

<img width="999" height="545" alt="download" src="https://github.com/user-attachments/assets/6420269a-3bdb-4666-9a02-96998e906b91" />

FINIAL PREDICTION

<img width="1009" height="545" alt="download" src="https://github.com/user-attachments/assets/b28840f6-e3db-4613-8517-e88907a08afe" />

### RESULT:
Thus we have successfully implemented the auto regression function using python.
