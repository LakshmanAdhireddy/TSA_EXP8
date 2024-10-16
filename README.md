## Developed by:Lakshman

## Reg no:212222240001

# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
~~~python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df=pd.read_csv('supermarketsales.csv')


# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Display shape and first 20 rows (or the available data if fewer rows)
print("Dataset shape:", df.shape)
print("First rows of dataset:\n", df.head(20))

# Plot the original data (Total)
plt.figure(figsize=(10, 6))
plt.plot(df['Total'], label='Original Data', marker='o')
plt.title('Original Time Series Data (Total)')
plt.ylabel('Total')
plt.xlabel('Date')
plt.legend()
plt.show()

# Moving Average with window size 5 and 10
rolling_mean_5 = df['Total'].rolling(window=5).mean()
rolling_mean_10 = df['Total'].rolling(window=10).mean()

# Plot original data and rolling means (5 and 10)
plt.figure(figsize=(10, 6))
plt.plot(df['Total'], label='Original Data', marker='o')
plt.plot(rolling_mean_5, label='Rolling Mean (Window=5)', marker='x')
plt.plot(rolling_mean_10, label='Rolling Mean (Window=10)', marker='^')
plt.title('Original Data vs Rolling Means')
plt.ylabel('Total')
plt.xlabel('Date')
plt.legend()
plt.show()

# Perform Exponential Smoothing
exp_smoothing = SimpleExpSmoothing(df['Total']).fit(smoothing_level=0.2, optimized=False)
exp_smoothed = exp_smoothing.fittedvalues

# Plot Original Data and Exponential Smoothing
plt.figure(figsize=(10, 6))
plt.plot(df['Total'], label='Original Data', marker='o')
plt.plot(exp_smoothed, label='Exponential Smoothing', marker='s')
plt.title('Original Data vs Exponential Smoothing')
plt.ylabel('Total')
plt.xlabel('Date')
plt.legend()
plt.show()

# Plot ACF and PACF
plt.figure(figsize=(10, 6))
plt.subplot(121)
plot_acf(df['Total'], lags=10, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')
plt.subplot(122)
plot_pacf(df['Total'], lags=10, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# Generate Predictions using Exponential Smoothing (Predict next 3 values)
prediction_steps = 3
forecast = exp_smoothing.forecast(steps=prediction_steps)

# Plot original data and predictions
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Total'], label='Original Data', marker='o')
plt.plot(pd.date_range(start=df.index[-1], periods=prediction_steps + 1, freq='D')[1:], forecast, label='Predictions', marker='x')
plt.title('Original Data vs Predictions (Exponential Smoothing)')
plt.ylabel('Total')
plt.xlabel('Date')
plt.legend()
plt.show()
~~~

### OUTPUT:
![17290522648865201917614762743826](https://github.com/user-attachments/assets/22e5ef60-c0e8-45e6-b3e0-77d9dc8edb88)

![17290522892745432667809239531642](https://github.com/user-attachments/assets/09d88492-d352-4aa3-bd90-f52ac26777fc)

![17290523010911158142138089437013](https://github.com/user-attachments/assets/17628ceb-cda5-4480-a5dc-02535c49cbd3)

![17290523171388019278535730447694](https://github.com/user-attachments/assets/00c12b2c-6467-44df-ade1-c939553a21a8)


![17290523282766428833896245309373](https://github.com/user-attachments/assets/585567ac-3732-40a6-994f-2e0dbf347e7d)


### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
