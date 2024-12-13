import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, acf, pacf
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import mean_squared_error

class ARIMAModel:
    """
    Classe para construir e prever modelos ARIMA.
    """

    def __init__(self):
        """
        Inicializa o modelo ARIMA sem parâmetros fixos.
        """
        self.p = None
        self.d = None
        self.q = None
        self.params_ar = None
        self.params_ma = None
        self.data = None
        self.original_data = None
        self.residuals = None

    def fit(self, data, p, d, q):
        """
        Ajusta o modelo ARIMA aos dados fornecidos.

        Parâmetros:
        - data: Série temporal (pandas Series ou lista/array).
        - p: Ordem da parte autoregressiva (AR).
        - d: Ordem da diferenciação (I).
        - q: Ordem da média móvel (MA).
        """
        self.original_data = data if isinstance(data, pd.Series) else pd.Series(data)
        self.data = self.original_data.copy()
        self.p = p
        self.d = d
        self.q = q

        # Diferenciação da série
        if self.d > 0:
            self.data = ARIMATools.difference(self.data, order=self.d)

        # Estima os parâmetros AR e MA
        initial_params_ar, initial_params_ma = ARIMATools.estimate_ar_ma(self.data, self.p, self.q)

        # Otimiza os parâmetros AR e MA
        self.params_ar, self.params_ma = self.optimize_params(initial_params_ar, initial_params_ma)

        # Calcula os resíduos para os dados ajustados
        self.residuals = []
        for t in range(max(self.p, self.q), len(self.data)):
            ar_term = sum(self.params_ar[i] * self.data.iloc[t - i - 1] for i in range(self.p)) if self.p > 0 else 0
            ma_term = sum(self.params_ma[i] * self.residuals[-(i + 1)] if len(self.residuals) > i else 0 for i in range(self.q)) if self.q > 0 else 0
            predicted = ar_term + ma_term
            self.residuals.append(self.data.iloc[t] - predicted)


    def optimize_params(self, initial_params_ar, initial_params_ma):
        """
        Ajusta os parâmetros AR e MA e recalcula os resíduos com base nos parâmetros iniciais.

        Parâmetros:
        - initial_params_ar: Lista de parâmetros iniciais AR (phi).
        - initial_params_ma: Lista de parâmetros iniciais MA (theta).

        Retorna:
        - params_ar: Lista com os parâmetros otimizados AR (phi).
        - params_ma: Lista com os parâmetros otimizados MA (theta).
        """
        data = self.data.values

        # Inicializa os coeficientes para otimização
        phi = np.array(initial_params_ar) if initial_params_ar else np.array([])
        theta = np.array(initial_params_ma) if initial_params_ma else np.array([])

        # Função de erro para ajustar os coeficientes
        def objective_function(params):
            p_len = len(phi)
            q_len = len(theta)

            # Atualiza os coeficientes AR e MA
            phi_updated = params[:p_len] if p_len > 0 else []
            theta_updated = params[p_len:] if q_len > 0 else []

            residuals = [0] * max(self.q, 1)  # Inicializa os resíduos com tamanho suficiente
            predictions = []
            for t in range(max(self.p, self.q), len(data)):
                ar_term = sum(phi_updated[i] * data[t - i - 1] for i in range(self.p)) if self.p > 0 else 0
                ma_term = sum(theta_updated[i] * residuals[-(i + 1)] if len(residuals) > i else 0 for i in range(self.q)) if self.q > 0 else 0

                predicted = ar_term + ma_term
                predictions.append(predicted)
                residuals.append(data[t] - predicted)

            # Calcula o erro quadrático médio
            mse = np.mean((data[max(self.p, self.q):] - np.array(predictions))**2)
            return mse

        # Otimização
        from scipy.optimize import minimize
        initial_params = np.concatenate([phi, theta]) if phi.size > 0 or theta.size > 0 else np.array([])

        if len(initial_params) > 0:
            result = minimize(objective_function, initial_params, method='L-BFGS-B')
            optimized_params = result.x
        else:
            optimized_params = []

        params_ar = list(optimized_params[:len(phi)]) if phi.size > 0 else []
        params_ma = list(optimized_params[len(phi):]) if theta.size > 0 else []

        return params_ar, params_ma

    def predict(self, steps):
        """
        Realiza previsões para os próximos passos com base no modelo ajustado.

        Parâmetros:
        - steps: Número de passos para prever.

        Retorno:
        - Previsões como uma lista.
        """
        if self.params_ar is None or self.params_ma is None:
            raise ValueError("Modelo não ajustado. Execute 'fit' antes de prever.")

        forecast = []
        residuals = [0] * self.q
        last_values = list(self.data[-self.p:].values) if self.p > 0 else []

        for _ in range(steps):
            # Calcula a parte autoregressiva (AR)
            ar_term = sum(self.params_ar[i] * last_values[-(i+1)] for i in range(self.p)) if self.p > 0 else 0

            # Calcula a parte de média móvel (MA)
            ma_term = sum(self.params_ma[i] * residuals[-(i+1)] for i in range(self.q)) if self.q > 0 else 0

            # Gera o próximo valor
            next_value = ar_term + ma_term + np.random.normal()
            forecast.append(next_value)

            # Atualiza os últimos valores e os resíduos
            if self.p > 0:
                last_values.append(next_value)
                if len(last_values) > self.p:
                    last_values.pop(0)

            residuals.append(next_value - (ar_term + ma_term))
            if len(residuals) > self.q:
                residuals.pop(0)

        # Reintegra a série se d > 0
        if self.d > 0:
            forecast = ARIMATools.inverse_difference(self.original_data, forecast, self.d)

        return forecast

    def iterative_predict(model, steps):
        """
        Realiza previsões iterativas usando a série diferenciada, sem reintegrar os valores.

        Parâmetros:
        - model: Instância de um modelo ARIMAModel já ajustado.
        - steps: Número de passos para prever.

        Retorna:
        - Lista de previsões baseadas na série diferenciada.
        """
        forecast = []
        temp_series = model.data.copy()

        for _ in range(steps):
            model.fit(temp_series, model.p, model.d, model.q)
            next_value = model.predict(steps=1)[0]
            forecast.append(next_value)

            if isinstance(temp_series.index, pd.DatetimeIndex):
                next_index = temp_series.index[-1] + pd.tseries.frequencies.to_offset(temp_series.index.freq or 'D')
            else:
                next_index = temp_series.index[-1] + 1

            temp_series = pd.concat([temp_series, pd.Series([next_value], index=[next_index])])

        return forecast


class ARIMATools:
    """
    Classe utilitária para cálculos relacionados ao modelo ARIMA.
    """

    @staticmethod
    def difference(series, order=1):
        """
        Aplica diferenciação a uma série temporal.

        Parâmetros:
        - series: Série temporal (pandas Series ou lista/array).
        - order: Ordem da diferenciação.

        Retorno:
        - Série diferenciada como pandas Series.
        """
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        diff_series = series.copy()
        for _ in range(order):
            diff_series = diff_series.diff().dropna()

        return diff_series

    @staticmethod
    def inverse_difference(original_series, forecast, order):
        """
        Reverte a diferenciação para trazer os valores previstos de volta à escala original.

        Parâmetros:
        - original_series: Série temporal original (pandas Series).
        - forecast: Lista de previsões.
        - order: Ordem da diferenciação.

        Retorno:
        - Lista de previsões integradas.
        """
        if not isinstance(original_series, pd.Series):
            raise ValueError("A série original deve ser um pandas Series.")

        integrated_forecast = forecast.copy()
        last_known = original_series.iloc[-1]

        for i in range(len(forecast)):
            integrated_forecast[i] += last_known
            last_known = integrated_forecast[i]

        return integrated_forecast

    @staticmethod
    def estimate_ar_ma(series, p, q):
        """
        Estima os parâmetros AR (phi) e MA (theta) para a série fornecida.

        Parâmetros:
        - series: Série temporal (pandas Series).
        - p: Ordem da parte autoregressiva (AR).
        - q: Ordem da parte de média móvel (MA).

        Retorno:
        - Parâmetros AR (phi) e MA (theta) como listas.
        """
        # Estima os coeficientes AR usando Yule-Walker
        ar_params = []
        if p > 0:
            autocorr = acf(series, nlags=p)  # Calcula a autocorrelação
            R = np.array([autocorr[:p]])  # Matriz de autocorrelações
            R = np.vstack([np.roll(R, i, axis=1) for i in range(p)])  # Matriz de autocorrelações
            r = autocorr[1:p+1]  # Vetor de autocorrelação
            ar_params = np.linalg.solve(R, r)  # Resolve o sistema linear

        # Estima os coeficientes MA usando ACF
        ma_params = []
        if q > 0:
            ma_params = acf(series, nlags=q)[1:q+1]

        return list(ar_params), list(ma_params)

    @staticmethod
    def find_d(series, max_d=3):
        """
        Encontra o grau de diferenciação necessário para estacionarizar a série.

        Parâmetros:
        - series: Série temporal (pandas Series).
        - max_d: Número máximo de diferenciações a testar.

        Retorno:
        - d: Grau de diferenciação ideal.
        """
        for d in range(max_d + 1):
            diff_series = ARIMATools.difference(series, order=d)
            p_value = adfuller(diff_series.dropna())[1]  # Teste de Dickey-Fuller
            if p_value < 0.05:  # Estacionariedade alcançada
                return print("Grau de diferenciação ideal (d):", d)
        return max_d

    @staticmethod
    def calculate_rmse(y_true, y_pred):
        """
        Calcula o RMSE entre valores reais e previstos.
        """
        return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

    @staticmethod
    def calculate_mae(y_true, y_pred):
        """
        Calcula o MAE entre valores reais e previstos.
        """
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

    @staticmethod
    def calculate_confidence_intervals(forecast, residuals, confidence=0.95):
        """
        Calcula os intervalos de confiança para as previsões.

        Parâmetros:
        - forecast: Lista de previsões.
        - residuals: Lista de resíduos do modelo.
        - confidence: Nível de confiança (padrão 95%).

        Retorno:
        - lower_bound: Limite inferior do intervalo de confiança.
        - upper_bound: Limite superior do intervalo de confiança.
        """
        z_score = 1.96  # Para 95% de confiança
        std_error = np.std(residuals) if residuals else 1
        lower_bound = [f - z_score * std_error for f in forecast]
        upper_bound = [f + z_score * std_error for f in forecast]

        return lower_bound, upper_bound

    @staticmethod
    def plot_series(series, title="Série Temporal"):
        """
        Plota a série temporal.

        Parâmetros:
        - series: Série temporal (pandas Series ou lista/array).
        - title: Título do gráfico.
        """
        if not isinstance(series, pd.Series):
            raise ValueError("A entrada 'series' deve ser um pandas Series com um índice temporal válido.")

        plt.figure(figsize=(12, 6))
        plt.plot(series.index, series.values, color='blue', label='Série Temporal')
        plt.title(title, fontsize=16)
        plt.xlabel("Data", fontsize=14)
        plt.ylabel("Valores", fontsize=14)
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()

    @staticmethod
    def plot_acf_pacf(series, lags=40):
        """
        Plota os gráficos de ACF (Função de Autocorrelação) e PACF (Função de Autocorrelação Parcial).

        Parâmetros:
        - series: Série temporal (pandas Series ou lista/array).
        - lags: Número de defasagens para calcular.
        """
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot ACF
        plot_acf(series, lags=lags, ax=axes[0])
        axes[0].set_title("ACF (Autocorrelação)")

        # Plot PACF
        plot_pacf(series, lags=lags, ax=axes[1], method='ywm')
        axes[1].set_title("PACF (Autocorrelação Parcial)")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_forecast(original_series, forecast, steps, lower_bound=None, upper_bound=None):
        """
        Plota as previsões em comparação com a série original.

        Parâmetros:
        - original_series: Série temporal original (pandas Series).
        - forecast: Lista de previsões.
        - steps: Número de passos previstos.
        - lower_bound: Limites inferiores do intervalo de confiança.
        - upper_bound: Limites superiores do intervalo de confiança.
        """
        if len(forecast) != steps:
            forecast = forecast[:steps]  # Ajusta o comprimento se necessário

        original_length = len(original_series)
        forecast_range = range(original_length, original_length + steps)

        plt.figure(figsize=(12, 6))
        plt.plot(range(original_length), original_series, label="Série Original")
        plt.plot(forecast_range, forecast, label="Previsões", linestyle="--")
        plt.axvline(x=original_length - 1, color="gray", linestyle="--", label="Início das Previsões")

        if lower_bound and upper_bound:
            plt.fill_between(forecast_range, lower_bound, upper_bound, color='gray', alpha=0.2, label="Intervalo de Confiança 95%")

        plt.title("Previsões vs Série Original")
        plt.xlabel("Índice de Tempo")
        plt.ylabel("Valores")
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()

    @staticmethod
    def analyze_residuals(residuals):
        """
        Realiza a análise dos resíduos do modelo.

        Parâmetros:
        - residuals: Lista ou pandas Series de resíduos do modelo.

        Retorna:
        - Um dicionário contendo métricas estatísticas dos resíduos.
        - Plots de diagnóstico dos resíduos.
        """
        if not isinstance(residuals, pd.Series):
            residuals = pd.Series(residuals)

        # Estatísticas básicas
        stats = {
            'mean': residuals.mean(),
            'std_dev': residuals.std(),
            'min': residuals.min(),
            'max': residuals.max(),
            'skewness': residuals.skew(),
            'kurtosis': residuals.kurtosis()
        }

        # Plots de diagnóstico
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Histograma
        sns.histplot(residuals, kde=True, ax=axes[0], color='blue')
        axes[0].set_title("Histograma dos Resíduos")
        axes[0].set_xlabel("Resíduos")
        axes[0].set_ylabel("Frequência")

        # ACF dos resíduos
        plot_acf(residuals, ax=axes[1])
        axes[1].set_title("ACF dos Resíduos")

        # Q-Q plot
        norm_residuals = (residuals - residuals.mean()) / residuals.std()
        sorted_residuals = np.sort(norm_residuals)
        theoretical_quantiles = norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
        axes[2].scatter(theoretical_quantiles, sorted_residuals, color='blue', alpha=0.6)
        axes[2].plot([-3, 3], [-3, 3], color='red', linestyle='--')
        axes[2].set_title("Q-Q Plot dos Resíduos")
        axes[2].set_xlabel("Quantis Teóricos")
        axes[2].set_ylabel("Quantis Observados")

        plt.tight_layout()
        plt.show()

        return stats
    
    @staticmethod
    def grid_search(dataset, p_values, d_values, q_values):
        """
        Realiza uma busca em grade (grid search) para encontrar os melhores parâmetros (p, d, q) para um modelo ARIMA.

        Parâmetros:
        - dataset: Série temporal (pandas Series ou numpy array).
        - p_values: Lista de valores para p (autoregressivo).
        - d_values: Lista de valores para d (diferenciação).
        - q_values: Lista de valores para q (média móvel).

        Retorna:
        - Um dicionário com os melhores parâmetros e o menor erro quadrático médio (MSE).
        """
        dataset = dataset.astype('float32') if isinstance(dataset, np.ndarray) else dataset.astype(float)
        best_score, best_cfg = float("inf"), None

        for p in range(p_values+1):
            for d in range(d_values+1):
                for q in range(q_values+1):
                    try:
                        # Ajusta o modelo ARIMA para os parâmetros atuais
                        model = ARIMAModel()
                        model.fit(dataset, p, d, q)

                        # Faz previsões no conjunto de treino para calcular o erro
                        forecast = model.predict(steps=len(dataset))

                        # Verifica valores extremos no forecast para evitar overflow
                        if np.any(np.abs(forecast) > 1e6):
                            raise ValueError("Valores previstos muito altos, possível overflow.")

                        mse = mean_squared_error(dataset[-len(forecast):], forecast)

                        if mse < best_score:
                            best_score, best_cfg = mse, (p, d, q)
                        print(f'ARIMA{(p, d, q)} MSE={mse:.3f}')
                    except Exception as e:
                        print(f'Erro com ARIMA{(p, d, q)}: {e}')
                        continue

        print(f'Melhor configuração: ARIMA{best_cfg} com MSE={best_score:.3f}')
        return {'best_params': best_cfg, 'best_mse': best_score}
    
    @staticmethod
    def plot_forecast_differenced(original_series, forecast, steps):
        """
        Plota as previsões em comparação com a série diferenciada.

        Parâmetros:
        - original_series: Série temporal original diferenciada (pandas Series).
        - forecast: Lista de previsões.
        - steps: Número de passos previstos.
        """
        if len(forecast) != steps:
            forecast = forecast[:steps]  # Ajusta o comprimento se necessário

        original_length = len(original_series)
        forecast_range = range(original_length, original_length + steps)

        plt.figure(figsize=(12, 6))
        plt.plot(range(original_length), original_series, label="Série Diferenciada Original", marker="o")
        plt.plot(forecast_range, forecast, label="Previsões", marker="x", linestyle="--")
        plt.axvline(x=original_length - 1, color="gray", linestyle="--", label="Início das Previsões")
        plt.title("Previsões vs Série Diferenciada")
        plt.xlabel("Índice de Tempo")
        plt.ylabel("Valores")
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()
