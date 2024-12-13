# ARIMA Time Series Forecasting Library

## Overview
This library implements an ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting, designed to work without additional external dependencies like statsmodels for model fitting. The implementation includes tools for model fitting, forecasting, residual analysis, and hyperparameter optimization.

This project was created for the course **PGIA7326 - SISTEMAS INTELIGENTES PARA ANÁLISE E PREVISÃO DE SISTEMAS COMPLEXOS - 2024.2**, part of the **Postgraduate Program in Applied Informatics** at the **Universidade Federal Rural de Pernambuco (UFRPE)**.

---

## Features
- **ARIMA Model:**
  - Custom implementation of ARIMA(p, d, q) models.
  - Iterative prediction for forecasting horizons.
  - Manual and automatic optimization of AR and MA coefficients.
  - Grid search for hyperparameter tuning (p, d, q).

- **Residual Analysis:**
  - Histogram, ACF, and Q-Q plots for diagnostic checking.
  - Statistical summaries of residuals.

- **Forecasting Tools:**
  - Visualization of forecasted values alongside original time series.
  - Confidence interval calculations for predictions.

---

## Requirements
This implementation leverages basic Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `sklearn`

Ensure these are installed before using the library.

---

## How to Use

### 1. Fit an ARIMA Model
```python
from arima_model import ARIMAModel

# Example data (replace with your time series)
data = pd.Series([...])  # Time series data here

# Instantiate the model
model = ARIMAModel()

# Fit the model with p, d, q parameters
model.fit(data, p=2, d=1, q=2)
```

### 2. Generate Forecasts
```python
# Forecast next 10 steps
forecast = model.predict(steps=10)
print("Forecast:", forecast)
```

### 3. Perform Residual Analysis
```python
from arima_tools import ARIMATools

# Analyze residuals
residual_stats = ARIMATools.analyze_residuals(model.residuals)
print("Residual Statistics:", residual_stats)
```

### 4. Hyperparameter Tuning (Grid Search)
```python
# Define parameter ranges
p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]

# Perform grid search
best_params = ARIMATools.grid_search(data, p_values, d_values, q_values)
print("Best Parameters:", best_params)
```

### 5. Visualize Forecasts
```python
# Plot forecasted values
ARIMATools.plot_forecast_differenced(original_series=data, forecast=forecast, steps=10)
```

---

## File Structure
- `arima_model.py`: Contains the `ARIMAModel` class for fitting and forecasting.
- `arima_tools.py`: Provides utility functions for residual analysis, grid search, and plotting.

---

## Key Methods and Classes

### `ARIMAModel`
- **`fit(data, p, d, q)`**: Fits the ARIMA model to the data.
- **`predict(steps)`**: Predicts future values based on the fitted model.
- **`iterative_predict(steps)`**: Predicts future values iteratively, appending forecasts to the series.
- **`calculate_confidence_intervals(steps)`**: Calculates 95% confidence intervals for predictions.

### `ARIMATools`
- **`analyze_residuals(residuals)`**: Performs residual analysis with diagnostic plots.
- **`grid_search(dataset, p_values, d_values, q_values)`**: Conducts hyperparameter tuning for ARIMA.
- **`plot_forecast_differenced(original_series, forecast, steps)`**: Visualizes forecasts alongside the original series.

---

## Example Dataset
You can use any time series data formatted as a Pandas Series with a datetime index. Here's an example:

```python
import pandas as pd

data = pd.Series(
    [1132.98, 1136.52, 1137.14, 1141.69, 1144.98],
    index=pd.to_datetime(['2010-01-04', '2010-01-05', '2010-01-06', '2010-01-07', '2010-01-08'])
)
```

---

## Contributions
Feel free to submit pull requests to improve the library or add new features. Suggestions and bug reports are welcome!

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

