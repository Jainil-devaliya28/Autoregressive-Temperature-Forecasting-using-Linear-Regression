# Working with Autoregressive Modeling
Consider the [Daily Temperatures dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv) from Australia. This is a dataset for a forecasting task. That is, given temperatures up to date (or period) T, design a forecasting (autoregressive) model to predict the temperature on date T+1. You can refer to [link 1](https://www.turing.com/kb/guide-to-autoregressive-models), [link 2](https://otexts.com/fpp2/AR.html) for more information on autoregressive models. Use linear regression as your autoregressive model. Plot the fit of your predictions vs the true values and report the RMSE obtained. A demonstration of the plot is given below.
<img width="650" height="400" alt="Autoregressive_Demo" src="https://github.com/user-attachments/assets/15e272b9-8961-4b8d-9ce4-89bfbcc45348" />

# Working of my model
Dataset loaded and cleaned. Total observations: 3650
Training set size: 2554 samples
Test set size: 1095 samples
RMSE (Root Mean Squared Error) on Test Set: 2.4494 degrees Celsius
<img width="1200" height="600" alt="Autoregressive Model Fit" src="https://github.com/user-attachments/assets/33a8542a-86d8-4f64-94c6-aea55676a1c5" />
