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

    def check_stationarity(self, series, alpha=0.05, plot_acf_pacf=True):
        """
        Verifica estacionariedade com base no teste ADF e na análise da ACF/PACF.

        Args:
            alpha (float): Nível de significância para o teste ADF. Default é 0.05.
            plot_acf_pacf (bool): Se True, plota os gráficos de ACF e PACF.

        Returns:
            dict: Resultados do teste de estacionariedade e diagnósticos visuais.
        """
        # Teste de Dickey-Fuller Aumentado
        adf_result = adfuller(series, autolag="AIC")
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
            self._plot_acf_pacf(series)

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

    def difference(self, series, order=1):
        """
        Aplica diferenciação na série (DataFrame) para torná-la estacionária.

        Args:
            series (pandas.DataFrame): Série temporal no formato DataFrame.
            order (int): Ordem da diferenciação.

        Returns:
            pandas.DataFrame: Série diferenciada.
        """
        diff_series = series.copy()
        for _ in range(order):
            diff_series = diff_series.diff().dropna()  # Aplica diferenças e remove NaN
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
            # Usar os valores iniciais de original_series para reconstruir a escala
            inverted_series = np.r_[original_series[-len(inverted_series):][0], inverted_series].cumsum()
        return inverted_series
    
    def autoregressive_term(self, series, phi, p):
        """
        Calcula o termo autorregressivo com base nos parâmetros phi.

        Args:
            series (array-like): Série temporal.
            phi (array-like): Coeficientes autorregressivos.
            p (int): Ordem autorregressiva.

        Returns:
            float: Valor do termo autorregressivo.
        """
        return sum(phi[j] * series[-j - 1] for j in range(p))

    def moving_average_term(self, residuals, theta, q):
        """
        Calcula o termo de médias móveis com base nos parâmetros theta.

        Args:
            residuals (array-like): Resíduos do modelo.
            theta (array-like): Coeficientes de médias móveis.
            q (int): Ordem da média móvel.

        Returns:
            float: Valor do termo de médias móveis.
        """
        return sum(theta[k] * residuals[-k - 1] for k in range(q))


    def fit(self, series, p, d, q):
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
        # Garantir que a série seja um array NumPy
        if isinstance(series, pd.DataFrame):
            series = series.values.flatten()
        elif isinstance(series, pd.Series):
            series = series.values

        n = len(series)
        if n <= max(p, q):
            raise ValueError(f"Série muito curta para ajustar ARIMA({p},{d},{q}). Tamanho da série: {n}, necessário: {max(p, q) + 1}")

        X = np.zeros((n - max(p, q), p + q))

        phi = np.random.rand(p)  # Coeficientes autorregressivos iniciais
        theta = np.random.rand(q)  # Coeficientes de médias móveis iniciais
        residuals = np.zeros(n)  # Inicializar resíduos

        for i in range(max(p, q), n):
            # Termo autorregressivo
            ar_term = self.autoregressive_term(series[:i], phi, p)

            # Termo de médias móveis
            ma_term = self.moving_average_term(residuals[:i], theta, q)

            # Construção da matriz de preditores
            X[i - max(p, q), :p] = ar_term
            X[i - max(p, q), p:] = ma_term

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

    def grid_search(self, max_p, max_d, max_q, train_split=0.9):
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
            stationarity_result = self.check_stationarity(transformed_train.values.flatten(), plot_acf_pacf=False)
            if stationarity_result["Is Stationary"]:
                break

            transformed_train = self.difference(transformed_train, order=1)
            d += 1

        if transformed_train.empty or len(transformed_train) <= max_p + max_q:
            raise ValueError("A série transformada é muito curta para ajuste do modelo ARIMA.")


        # Iteração pelos valores de p e q
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    print(f"Tentando ARIMA({p},{d},{q})...")  # Log de tentativa
                    
                    # Ajuste do modelo ARIMA
                    model = self.fit(transformed_train.values.flatten(), p, d, q)

                    # Previsões
                    forecast = self.forecast_arima(transformed_train.values.flatten(), model["coeffs"], p, q, len(validation))

                    # Reversão da diferenciação
                    forecast_inverted = self.inverse_difference(train.values.flatten(), forecast, order=d)
                    # Ajuste os tamanhos para coincidirem
                    min_len = min(len(validation.values.flatten()), len(forecast_inverted))
                    mse = mean_squared_error(validation.values.flatten()[:min_len], forecast_inverted[:min_len])
                    print(f"ARIMA({p},{d},{q}) -> MSE: {mse}")  # Log do resultado

                    # Atualiza os melhores parâmetros se o erro for menor
                    if mse < best_score:
                        best_score = mse
                        best_params = (p, d, q)
                except Exception as e:
                    print(f"Erro com ARIMA({p},{d},{q}): {e}")  # Log de erro
                    continue

        # Armazena os melhores parâmetros e seu desempenho
        self.best_params = best_params

        return {
            "Best Params": best_params,
            "Best MSE": best_score
        }

    def plot_forecast(self, actual, forecast, steps):
        """
        Gera um gráfico comparando os valores reais com as previsões.

        Args:
            actual (array-like): Valores reais da série temporal.
            forecast (array-like): Valores previstos pelo modelo.
            steps (int): Número de passos de previsão.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(actual)), actual, label="Valores Reais", color="blue")
        plt.plot(range(len(actual) - steps, len(actual)), forecast, label="Previsões", color="orange")
        plt.title("Comparação entre valores reais e previsões")
        plt.xlabel("Tempo")
        plt.ylabel("Valor")
        plt.legend()
        plt.grid()
        plt.show()

        
        