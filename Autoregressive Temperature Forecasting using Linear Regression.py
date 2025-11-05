import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'

try:
    df = pd.read_csv(URL, header=0, index_col=0, parse_dates=True, squeeze=True)
except TypeError:
    # Handle squeeze deprecation in newer pandas versions
    df = pd.read_csv(URL, header=0, index_col=0, parse_dates=True)
    df = df.iloc[:, 0] # Select the single series column

df = df.astype(str).str.replace('?', '', regex=False)
df = pd.to_numeric(df, errors='coerce')

df.dropna(inplace=True)

series = df.rename('Actual_Temp')

print(f"Dataset loaded and cleaned. Total observations: {len(series)}")


df_ar = pd.DataFrame(series)
df_ar['Lag_1_Temp'] = df_ar['Actual_Temp'].shift(1)

df_ar.dropna(inplace=True)

X = df_ar[['Lag_1_Temp']]
y = df_ar['Actual_Temp']


split_point = int(len(df_ar) * 0.70)

X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

print(f"Training set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))


print(f"RMSE (Root Mean Squared Error) on Test Set: {rmse:.4f} degrees Celsius")

plot_df = pd.DataFrame({
    'Actual_Temp': series.values,
}, index=series.index)

plot_df['Forecast'] = np.nan

plot_df.loc[y_test.index, 'Forecast'] = y_pred

plt.figure(figsize=(12, 6))

plt.plot(plot_df.index[:split_point], plot_df['Actual_Temp'][:split_point], label='Train (Actual)', color='tab:blue')

plt.plot(plot_df.index[split_point:], plot_df['Actual_Temp'][split_point:], label='Test (Actual)', color='tab:orange')

plt.plot(plot_df.index[split_point:], plot_df['Forecast'][split_point:], label='Forecast (AR-LR)', color='tab:green', linestyle='--')

plt.title('Autoregressive Model Fit: Daily Minimum Temperatures')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()