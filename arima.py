import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class ARIMAModel:
    def __init__(self, series: pd.Series):
        self.series = series
        self.transformed_series = series
        self.is_stationary = None
        self.adf_result = None

    def check_stationarity(self, alpha=0.05, plot_acf_pacf=True):
        """
        Verifica estacionariedade com base no teste ADF e na análise da ACF/PACF.

        Args:
            alpha (float): Nível de significância para o teste ADF. Default é 0.05.
            plot_acf_pacf (bool): Se True, plota os gráficos de ACF e PACF.

        Returns:
            dict: Resultados do teste de estacionariedade e diagnósticos visuais.
        """
        # Teste de Dickey-Fuller Aumentado
        adf_result = adfuller(self.series, autolag="AIC")
        p_value = adf_result[1]
        test_stat = adf_result[0]
        critical_values = adf_result[4]

        # Atualiza informações internas
        self.is_stationary = p_value < alpha
        self.adf_result = {
            "ADF Statistic": test_stat,
            "p-value": p_value,
            "Critical Values": critical_values,
            "Is Stationary": self.is_stationary,
        }

        # Diagnóstico ACF e PACF
        if plot_acf_pacf:
            self._plot_acf_pacf(self.series)

        return self.adf_result

    def _plot_acf_pacf(self, series):
        """
        Plota os gráficos de ACF e PACF para análise de estacionariedade.

        Args:
            series (pd.Series): Série temporal a ser analisada.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_acf(series, ax=axes[0], lags=20, title="ACF")
        plot_pacf(series, ax=axes[1], lags=20, title="PACF")
        plt.tight_layout()
        plt.show()

    def transform_series(self, diff_order=1):
        """
        Aplica transformações para tornar a série estacionária.

        Args:
            diff_order (int): Ordem de diferenciação para remover tendência.
            seasonal_diff_order (int): Ordem de diferenciação sazonal.
            seasonal_period (int): Período sazonal (ex: 12 para dados mensais).

        Returns:
            pd.Series: Série transformada.
        """
        series = self.series
        # Diferenciação para remover tendência
        if diff_order > 0:
            series = series.diff(diff_order).dropna()

        self.transformed_series = series
        return series


    def difference(self, series, order=1):
        """
        Aplica diferenciação na série para torná-la estacionária.

        Args:
            series (array-like): Série temporal.
            order (int): Ordem da diferenciação.

        Returns:
            np.ndarray: Série diferenciada.
        """
        diff_series = series.copy()
        for _ in range(order):
            diff_series = np.diff(diff_series, n=1)
        return diff_series

    def inverse_difference(self, original_series, diff_series, order=1):
        """
        Reverte a diferenciação para retornar à escala original.

        Args:
            original_series (array-like): Série original (não diferenciada).
            diff_series (array-like): Série diferenciada.
            order (int): Ordem da diferenciação.

        Returns:
            np.ndarray: Série na escala original.
        """
        inverted_series = diff_series.copy()
        for _ in range(order):
            inverted_series = np.r_[original_series[:1], inverted_series].cumsum()
        return inverted_series

    def fit_arima(self, series, p, d, q):
        """
        Ajusta o modelo ARIMA manualmente.

        Args:
            series (array-like): Série temporal (já diferenciada).
            p (int): Ordem autorregressiva.
            d (int): Ordem de diferenciação (assumida já aplicada).
            q (int): Ordem da média móvel.

        Returns:
            dict: Coeficientes ajustados e resíduos.
        """
        n = len(series)
        X = np.zeros((n - max(p, q), p + q))

        for i in range(max(p, q), n):
            # Autorregressivo
            for j in range(p):
                X[i - max(p, q), j] = series[i - j - 1] if i - j - 1 >= 0 else 0

            # Média móvel
            for k in range(q):
                X[i - max(p, q), p + k] = series[i - k - 1] if i - k - 1 >= 0 else 0

        y = series[max(p, q):]

        # Regressão linear para ajustar os coeficientes
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        fitted_values = X @ coeffs
        residuals = y - fitted_values

        return {
            "coeffs": coeffs,
            "residuals": residuals
        }

    def forecast_arima(self, series, coeffs, p, q, steps):
        """
        Realiza previsões usando o modelo ARIMA ajustado.

        Args:
            series (array-like): Série temporal (já diferenciada).
            coeffs (array-like): Coeficientes do modelo ARIMA.
            p (int): Ordem autorregressiva.
            q (int): Ordem da média móvel.
            steps (int): Número de passos à frente para previsão.

        Returns:
            np.ndarray: Previsões futuras.
        """
        forecast = []
        history = list(series)

        for _ in range(steps):
            ar_term = sum(coeffs[:p] * np.array(history[-p:])) if p > 0 else 0
            ma_term = sum(coeffs[p:p+q] * np.array(history[-q:])) if q > 0 else 0
            next_value = ar_term + ma_term
            forecast.append(next_value)
            history.append(next_value)

        return np.array(forecast)

    def grid_search(self, max_p, max_q, train_split=0.8, max_d=5, threshold=0.05):
        """
        Realiza uma busca em grade para encontrar os melhores parâmetros (p, d, q) para o modelo ARIMA.

        Args:
            max_p (int): Valor máximo para o parâmetro p.
            max_q (int): Valor máximo para o parâmetro q.
            train_split (float): Proporção da série para treino (default: 0.8).
            max_d (int): Número máximo de diferenciações permitidas.
            threshold (float): Nível de significância para estacionariedade.

        Returns:
            dict: Melhor combinação de parâmetros e o erro correspondente.
        """
        warnings.filterwarnings("ignore")

        best_score = float("inf")
        best_params = None

        # Divisão da série em treino e validação
        split_index = int(len(self.series) * train_split)
        train, validation = self.series[:split_index], self.series[split_index:]

        # Determina d dinamicamente
        transformed_train = train.copy()
        d = 0
        while d < max_d:
            stationarity_result = self.check_stationarity(transformed_train, threshold=threshold)
            if stationarity_result["Is Stationary"]:
                break
            transformed_train = self.difference(transformed_train)
            d += 1

        # Iteração pelos valores de p e q
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    # Ajuste do modelo ARIMA
                    model = self.fit_arima(transformed_train, p, d, q)

                    # Previsões
                    forecast = self.forecast_arima(transformed_train, model["coeffs"], p, q, len(validation))

                    # Reversão da diferenciação
                    forecast_inverted = self.inverse_difference(train, forecast, order=d)

                    # Calcula o erro (MSE) entre previsões e valores reais
                    mse = mean_squared_error(validation, forecast_inverted)

                    # Atualiza os melhores parâmetros se o erro for menor
                    if mse < best_score:
                        best_score = mse
                        best_params = (p, d, q)
                except Exception as e:
                    # Ignora combinações que causam erros
                    continue

        # Armazena os melhores parâmetros e seu desempenho
        self.best_params = best_params

        return {
            "Best Params": best_params,
            "Best MSE": best_score
        }

        
        