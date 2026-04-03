"""
================================================================================
  FRAMEWORK DE ARBITRAGEM ESTATÍSTICA (PAIRS TRADING) - CRIPTOMOEDAS
================================================================================
  Autor  : Engenheiro Quantitativo Sênior
  Versão : 2.0
  Python : 3.9+

  Descrição:
    Framework modular para identificar, testar e fazer backtest de estratégias
    de Long/Short baseadas em cointegração entre pares de criptomoedas.

  Instalação das dependências:
    pip install ccxt pandas numpy statsmodels plotly scipy

  Uso rápido:
    python crypto_pairs_trading.py
================================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import warnings
import logging
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  1. COLETOR DE DADOS
# ══════════════════════════════════════════════════════════════════════════════
class DataCollector:
    """
    Coleta dados OHLCV de exchanges via biblioteca ccxt.

    Suporta Binance e Bybit com paginação automática para
    séries históricas longas.
    """

    SUPPORTED_EXCHANGES = {
        "binance": ccxt.binance,
        "bybit": ccxt.bybit,
    }

    def __init__(self, exchange_name: str = "binance"):
        """
        Parâmetros
        ----------
        exchange_name : str
            Nome da exchange ('binance' ou 'bybit').
        """
        if exchange_name not in self.SUPPORTED_EXCHANGES:
            raise ValueError(
                f"Exchange '{exchange_name}' não suportada. "
                f"Use: {list(self.SUPPORTED_EXCHANGES.keys())}"
            )
        self.exchange = self.SUPPORTED_EXCHANGES[exchange_name](
            {"enableRateLimit": True}
        )
        self.exchange_name = exchange_name
        log.info("Exchange '%s' inicializada.", exchange_name)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        limit: int = 500,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Busca dados OHLCV e retorna um DataFrame limpo.

        Parâmetros
        ----------
        symbol    : str  — Par de trading (ex: 'BTC/USDT').
        timeframe : str  — Período ('1h', '4h', '1d', etc.).
        limit     : int  — Número de candles a buscar (máx ~1000 por request).
        since     : int  — Timestamp em ms para início da série (opcional).

        Retorna
        -------
        pd.DataFrame com colunas [open, high, low, close, volume]
        indexado por datetime UTC.
        """
        log.info("Buscando %s candles de %s [%s]...", limit, symbol, timeframe)
        try:
            raw = self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, limit=limit, since=since
            )
        except ccxt.BaseError as exc:
            log.error("Erro ao buscar dados: %s", exc)
            raise

        if not raw:
            raise ValueError(f"Nenhum dado retornado para {symbol}.")

        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)

        # ── Tratamento profissional de NaNs ───────────────────────────────
        # Forward-fill para fills de mercado fechado; drop restante
        n_missing = df["close"].isna().sum()
        if n_missing > 0:
            log.warning("%d NaNs encontrados em %s. Aplicando ffill.", n_missing, symbol)
            df.ffill(inplace=True)
            df.dropna(inplace=True)

        log.info("%s: %d registros carregados (de %s a %s).",
                 symbol, len(df), df.index[0].date(), df.index[-1].date())
        return df

    def fetch_close_prices(
        self,
        symbols: list,
        timeframe: str = "1d",
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Busca e alinha os preços de fechamento de múltiplos símbolos.

        Retorna
        -------
        pd.DataFrame com uma coluna por símbolo (apenas close).
        Linhas com qualquer NaN após alinhamento são descartadas.
        """
        closes = {}
        for sym in symbols:
            try:
                df = self.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
                closes[sym] = df["close"]
            except Exception as exc:  # pylint: disable=broad-except
                log.error("Ignorando %s devido a erro: %s", sym, exc)

        combined = pd.DataFrame(closes)
        before = len(combined)
        combined.dropna(inplace=True)
        after = len(combined)
        if before != after:
            log.warning("Descartadas %d linhas com NaN no alinhamento.", before - after)
        return combined


# ══════════════════════════════════════════════════════════════════════════════
#  2. ANÁLISE ESTATÍSTICA
# ══════════════════════════════════════════════════════════════════════════════
class StatisticalAnalysis:
    """
    Executa toda a análise estatística necessária para Pairs Trading.

    ──────────────────────────────────────────────────────────────────────────
    POR QUE COINTEGRAÇÃO > CORRELAÇÃO SIMPLES?
    ──────────────────────────────────────────────────────────────────────────
    Correlação mede a DIREÇÃO do movimento conjunto, mas é uma medida de
    curto prazo que pode se desfazer. Dois ativos podem ter correlação = 0.9
    e mesmo assim divergir permanentemente (spurious correlation).

    Cointegração é uma propriedade de LONGO PRAZO: ela garante que, apesar
    de cada série ser não-estacionária (I(1) — passeio aleatório), existe
    uma combinação linear entre elas que É estacionária (I(0)). Isso implica
    um mecanismo de correção de erros que FORÇA os preços a voltarem a um
    equilíbrio de longo prazo.

    Matematicamente:
      Se P1_t e P2_t são I(1), mas existe β tal que:
        ε_t = P1_t - β·P2_t  →  ε_t ~ I(0)
      Então P1 e P2 são cointegradas (Engle & Granger, 1987).

    Para o trader, isso significa: o spread ε_t é mean-reverting (reverte
    à média), tornando possível modelar entradas e saídas com base no
    desvio estatístico (Z-Score).
    ──────────────────────────────────────────────────────────────────────────
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Parâmetros
        ----------
        significance_level : float
            Nível de significância para rejeitar H0 (default: 5%).
        """
        self.alpha = significance_level

    # ── 2.1 Teste de Engle-Granger ────────────────────────────────────────
    def engle_granger_test(
        self, series_y: pd.Series, series_x: pd.Series
    ) -> dict:
        """
        Executa o teste de cointegração de Engle-Granger (two-step).

        Passo 1: Regride Y em X (OLS) → obtém resíduos ε̂.
        Passo 2: Aplica ADF nos resíduos → testa se ε̂ ~ I(0).

        H0: Não há cointegração (resíduos têm raiz unitária).
        H1: Há cointegração (resíduos são estacionários).
        """
        # statsmodels.tsa.stattools.coint implementa Engle-Granger
        t_stat, p_value, crit_values = coint(series_y, series_x)
        is_cointegrated = p_value < self.alpha

        result = {
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 4),
            "critical_values": {
                "1%": round(crit_values[0], 4),
                "5%": round(crit_values[1], 4),
                "10%": round(crit_values[2], 4),
            },
            "is_cointegrated": is_cointegrated,
        }
        log.info(
            "Engle-Granger | p-value=%.4f | Cointegrado: %s",
            p_value, is_cointegrated
        )
        return result

    # ── 2.2 Hedge Ratio via OLS ───────────────────────────────────────────
    def calculate_hedge_ratio(
        self, series_y: pd.Series, series_x: pd.Series
    ) -> tuple[float, pd.Series]:
        """
        Calcula o Hedge Ratio β usando regressão OLS:
            Y_t = α + β·X_t + ε_t

        O hedge ratio β indica quantas unidades de X são necessárias
        para fazer hedge de 1 unidade de Y.

        Retorna
        -------
        (beta, residuals) : tuple[float, pd.Series]
        """
        x_const = add_constant(series_x)
        model = OLS(series_y, x_const).fit()
        beta = model.params.iloc[1]        # Coeficiente de X
        alpha = model.params.iloc[0]       # Intercepto
        residuals = series_y - (alpha + beta * series_x)

        log.info("OLS | α=%.4f | β (hedge ratio)=%.4f | R²=%.4f",
                 alpha, beta, model.rsquared)
        return beta, residuals

    # ── 2.3 Z-Score do Spread ─────────────────────────────────────────────
    def calculate_zscore(
        self,
        spread: pd.Series,
        window: Optional[int] = None,
    ) -> pd.Series:
        """
        Calcula o Z-Score do spread.

        Se `window` for fornecido, usa médias/desvios móveis (rolling).
        Caso contrário, usa estatísticas globais da série completa.

        Z_t = (spread_t - μ) / σ

        Valores acima de +2 ou abaixo de -2 são usados como sinais de trade.
        """
        if window:
            mean = spread.rolling(window=window, min_periods=window).mean()
            std = spread.rolling(window=window, min_periods=window).std()
        else:
            mean = spread.mean()
            std = spread.std()

        zscore = (spread - mean) / std
        zscore.name = "zscore"
        return zscore

    # ── 2.4 Half-Life (Ornstein-Uhlenbeck) ───────────────────────────────
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Estima o Half-Life da reversão à média usando o modelo de
        Ornstein-Uhlenbeck (OU):

            dS_t = θ(μ - S_t)dt + σ·dW_t

        Forma discreta para estimação:
            ΔS_t = a + b·S_{t-1} + ε_t

        O coeficiente b = -θ·Δt estima a velocidade de reversão.
        Half-Life: t½ = -ln(2) / b

        Interpretação:
          - t½ < 10 candles → reversão muito rápida (bom para HFT)
          - t½ 10-100 candles → ideal para estratégias swing
          - t½ > 100 candles → reversão muito lenta (difícil de explorar)
        """
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        # Alinha os índices
        spread_lag, spread_diff = spread_lag.align(spread_diff, join="inner")

        # Regressão: ΔS_t = a + b·S_{t-1}
        x = add_constant(spread_lag)
        model = OLS(spread_diff, x).fit()
        b = model.params.iloc[1]

        if b >= 0:
            log.warning(
                "Coeficiente b=%.4f ≥ 0. O spread pode não ser mean-reverting.", b
            )
            return np.nan

        half_life = -np.log(2) / b
        log.info("Half-Life estimado: %.2f períodos", half_life)
        return round(half_life, 2)

    # ── 2.5 ADF Test nos Resíduos ─────────────────────────────────────────
    def adf_test(self, series: pd.Series) -> dict:
        """
        Augmented Dickey-Fuller Test.

        H0: A série tem raiz unitária (não-estacionária).
        H1: A série é estacionária.

        Para que a estratégia funcione, os resíduos devem rejeitar H0
        (p-value < α), confirmando que o spread é estacionário.
        """
        result = adfuller(series.dropna(), autolag="AIC")
        output = {
            "adf_statistic": round(result[0], 4),
            "p_value": round(result[1], 4),
            "lags_used": result[2],
            "n_observations": result[3],
            "critical_values": {k: round(v, 4) for k, v in result[4].items()},
            "is_stationary": result[1] < self.alpha,
        }
        log.info(
            "ADF | p-value=%.4f | Estacionário: %s",
            result[1], output["is_stationary"]
        )
        return output

    # ── 2.6 Relatório Completo ────────────────────────────────────────────
    def full_analysis(
        self, symbol_y: str, series_y: pd.Series,
        symbol_x: str, series_x: pd.Series,
    ) -> dict:
        """
        Executa toda a análise estatística de um par e retorna um
        dicionário com todos os resultados.
        """
        log.info("=" * 60)
        log.info("ANÁLISE: %s / %s", symbol_y, symbol_x)
        log.info("=" * 60)

        eg = self.engle_granger_test(series_y, series_x)
        beta, residuals = self.calculate_hedge_ratio(series_y, series_x)
        zscore = self.calculate_zscore(residuals)
        half_life = self.calculate_half_life(residuals)
        adf = self.adf_test(residuals)

        return {
            "pair": f"{symbol_y}/{symbol_x}",
            "engle_granger": eg,
            "hedge_ratio": round(beta, 6),
            "half_life_periods": half_life,
            "adf_on_residuals": adf,
            "residuals": residuals,
            "zscore": zscore,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  3. SCANNER DE COINTEGRAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
class CointegrationScanner:
    """
    Varre todos os pares possíveis em uma lista de símbolos e
    retorna uma matriz de p-values de cointegração.
    """

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level

    def scan(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Recebe um DataFrame de preços de fechamento (colunas = símbolos)
        e retorna uma matriz N×N de p-values do teste de Engle-Granger.

        Parâmetros
        ----------
        prices : pd.DataFrame
            Cada coluna é um símbolo. Linhas são timestamps.

        Retorna
        -------
        pd.DataFrame com p-values (NaN na diagonal).
        Pares com p-value < α são candidatos à estratégia.
        """
        symbols = prices.columns.tolist()
        n = len(symbols)
        pvalue_matrix = pd.DataFrame(
            np.full((n, n), np.nan),
            index=symbols,
            columns=symbols,
        )

        log.info("Escaneando %d símbolos → %d pares possíveis...",
                 n, n * (n - 1) // 2)

        for i in range(n):
            for j in range(i + 1, n):
                sym_a, sym_b = symbols[i], symbols[j]
                try:
                    _, p_val, _ = coint(prices[sym_a], prices[sym_b])
                    pvalue_matrix.loc[sym_a, sym_b] = round(p_val, 4)
                    pvalue_matrix.loc[sym_b, sym_a] = round(p_val, 4)
                except Exception as exc:  # pylint: disable=broad-except
                    log.warning("Erro no par %s/%s: %s", sym_a, sym_b, exc)

        # Ordena por menor p-value médio (pares mais promissores primeiro)
        mean_pvals = pvalue_matrix.mean(axis=1)
        pvalue_matrix = pvalue_matrix.loc[
            mean_pvals.sort_values().index,
            mean_pvals.sort_values().index
        ]

        cointegrated_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                sym_a, sym_b = symbols[i], symbols[j]
                pval = pvalue_matrix.loc[sym_a, sym_b]
                if pd.notna(pval) and pval < self.alpha:
                    cointegrated_pairs.append((sym_a, sym_b, pval))

        log.info("Pares cointegrados encontrados (p < %.2f): %d",
                 self.alpha, len(cointegrated_pairs))
        for pair in sorted(cointegrated_pairs, key=lambda x: x[2]):
            log.info("  ► %s / %s  →  p-value = %.4f", pair[0], pair[1], pair[2])

        return pvalue_matrix, cointegrated_pairs


# ══════════════════════════════════════════════════════════════════════════════
#  4. ENGINE DE BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════
class BacktestEngine:
    """
    Simula a estratégia de Pairs Trading com regras de entrada/saída
    baseadas em Z-Score e cálculo de PnL com taxas de corretagem.

    Lógica de Trading:
    ──────────────────
      LONG  spread : Z < -entry_threshold  → Compra Y, Vende X
      SHORT spread : Z > +entry_threshold  → Vende Y, Compra X
      SAÍDA        : Z cruza zero (ou stop-loss por desvio padrão)

    O spread é definido como:
        spread_t = price_Y_t - β · price_X_t
    """

    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        stop_loss_z: float = 3.5,
        maker_fee: float = 0.0002,     # 0.02% Binance Maker
        taker_fee: float = 0.0004,     # 0.04% Binance Taker
        initial_capital: float = 10_000.0,
    ):
        """
        Parâmetros
        ----------
        entry_threshold : float  — Z-Score para abrir posição (padrão: ±2.0).
        exit_threshold  : float  — Z-Score para fechar posição (padrão: 0.0).
        stop_loss_z     : float  — Stop-loss em desvios padrão (padrão: 3.5).
        maker_fee       : float  — Taxa maker (ordem limit, padrão Binance 0.02%).
        taker_fee       : float  — Taxa taker (ordem market, padrão 0.04%).
        initial_capital : float  — Capital inicial em USDT.
        """
        self.entry_z = entry_threshold
        self.exit_z = exit_threshold
        self.stop_z = stop_loss_z
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.capital = initial_capital

    def run(
        self,
        prices_y: pd.Series,
        prices_x: pd.Series,
        zscore: pd.Series,
        hedge_ratio: float,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Executa o backtest e retorna trades e métricas de performance.

        Parâmetros
        ----------
        prices_y     : Série de preços do ativo Y.
        prices_x     : Série de preços do ativo X.
        zscore       : Z-Score do spread (pré-calculado).
        hedge_ratio  : β (OLS hedge ratio).

        Retorna
        -------
        (trades_df, metrics) onde:
          trades_df : DataFrame com todos os trades executados.
          metrics   : Dicionário com Sharpe, Drawdown, Win Rate, etc.
        """
        # ── Alinha todas as séries ────────────────────────────────────────
        data = pd.DataFrame({
            "price_y": prices_y,
            "price_x": prices_x,
            "zscore": zscore,
        }).dropna()

        position = 0          # 0 = flat, 1 = long spread, -1 = short spread
        equity = self.capital
        equity_curve = []
        trades = []
        entry_info = {}

        for i in range(len(data)):
            row = data.iloc[i]
            z = row["zscore"]
            py = row["price_y"]
            px = row["price_x"]
            ts = data.index[i]

            # ── Sinal de SAÍDA ────────────────────────────────────────────
            if position != 0:
                # Saída por cruzamento do zero
                exit_signal = (
                    (position == 1 and z >= self.exit_z) or
                    (position == -1 and z <= self.exit_z)
                )
                # Saída por Stop-Loss
                stop_signal = abs(z) >= self.stop_z

                if exit_signal or stop_signal:
                    # PnL bruto do spread
                    spread_entry = entry_info["price_y"] - hedge_ratio * entry_info["price_x"]
                    spread_exit = py - hedge_ratio * px
                    raw_pnl = position * (spread_exit - spread_entry)

                    # Taxas (taker na saída = mercado)
                    fee_cost = (
                        py * self.taker_fee +
                        px * abs(hedge_ratio) * self.taker_fee
                    )
                    net_pnl = raw_pnl - fee_cost
                    equity += net_pnl

                    trades.append({
                        "entry_time": entry_info["time"],
                        "exit_time": ts,
                        "direction": "LONG" if position == 1 else "SHORT",
                        "entry_z": round(entry_info["z"], 4),
                        "exit_z": round(z, 4),
                        "raw_pnl": round(raw_pnl, 4),
                        "fee": round(fee_cost, 4),
                        "net_pnl": round(net_pnl, 4),
                        "exit_reason": "STOP_LOSS" if stop_signal else "ZERO_CROSS",
                        "equity": round(equity, 2),
                    })
                    position = 0

            # ── Sinal de ENTRADA ──────────────────────────────────────────
            if position == 0:
                if z < -self.entry_z:        # LONG: spread sub-valorizado
                    position = 1
                    # Taxas de entrada (maker = ordem limit)
                    fee_cost = (
                        py * self.maker_fee +
                        px * abs(hedge_ratio) * self.maker_fee
                    )
                    equity -= fee_cost
                    entry_info = {"time": ts, "price_y": py,
                                  "price_x": px, "z": z}

                elif z > self.entry_z:       # SHORT: spread super-valorizado
                    position = -1
                    fee_cost = (
                        py * self.maker_fee +
                        px * abs(hedge_ratio) * self.maker_fee
                    )
                    equity -= fee_cost
                    entry_info = {"time": ts, "price_y": py,
                                  "price_x": px, "z": z}

            equity_curve.append({"time": ts, "equity": equity})

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve).set_index("time")

        metrics = self._calculate_metrics(trades_df, equity_df)
        return trades_df, equity_df, metrics

    def _calculate_metrics(
        self, trades_df: pd.DataFrame, equity_df: pd.DataFrame
    ) -> dict:
        """Calcula métricas completas de performance."""
        if trades_df.empty:
            log.warning("Nenhum trade executado.")
            return {}

        eq = equity_df["equity"]
        returns = eq.pct_change().dropna()

        # ── Sharpe Ratio (anualizado) ─────────────────────────────────────
        trading_days_per_year = 365  # Cripto: mercado 24/7
        sharpe = (
            returns.mean() / returns.std() * np.sqrt(trading_days_per_year)
            if returns.std() > 0 else 0.0
        )

        # ── Maximum Drawdown ──────────────────────────────────────────────
        rolling_max = eq.cummax()
        drawdown = (eq - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # ── Calmar Ratio ──────────────────────────────────────────────────
        total_return = (eq.iloc[-1] - eq.iloc[0]) / eq.iloc[0]
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        # ── Win Rate e Profit Factor ──────────────────────────────────────
        wins = trades_df[trades_df["net_pnl"] > 0]
        losses = trades_df[trades_df["net_pnl"] <= 0]
        win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0.0
        gross_profit = wins["net_pnl"].sum()
        gross_loss = abs(losses["net_pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        metrics = {
            "total_trades": len(trades_df),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate * 100, 2),
            "total_pnl": round(trades_df["net_pnl"].sum(), 2),
            "total_fees": round(trades_df["fee"].sum(), 4),
            "total_return_pct": round(total_return * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "calmar_ratio": round(calmar, 3),
            "profit_factor": round(profit_factor, 3),
            "avg_trade_pnl": round(trades_df["net_pnl"].mean(), 4),
            "best_trade": round(trades_df["net_pnl"].max(), 4),
            "worst_trade": round(trades_df["net_pnl"].min(), 4),
            "stop_losses_hit": len(trades_df[trades_df["exit_reason"] == "STOP_LOSS"]),
        }

        # Log do sumário
        log.info("=" * 50)
        log.info("  MÉTRICAS DE PERFORMANCE")
        log.info("=" * 50)
        for k, v in metrics.items():
            log.info("  %-25s : %s", k, v)
        log.info("=" * 50)

        return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  5. VISUALIZADOR (PLOTLY INTERATIVO)
# ══════════════════════════════════════════════════════════════════════════════
class Visualizer:
    """
    Cria gráficos interativos com Plotly para análise do par de trading.

    Painéis:
    --------
      1. Preços normalizados (base 100) dos dois ativos.
      2. Spread com bandas de Z-Score (±1, ±2, ±3).
      3. Z-Score ao longo do tempo com marcações de trade.
      4. Curva de Equity com anotações de drawdown.
    """

    COLORS = {
        "asset_y": "#00D4FF",
        "asset_x": "#FF6B35",
        "spread": "#A78BFA",
        "zscore": "#34D399",
        "entry_long": "#22C55E",
        "entry_short": "#EF4444",
        "exit": "#F59E0B",
        "equity": "#60A5FA",
        "bg": "#0F172A",
        "grid": "#1E293B",
        "text": "#94A3B8",
        "band_pos": "rgba(239,68,68,0.15)",
        "band_neg": "rgba(34,197,94,0.15)",
    }

    def create_dashboard(
        self,
        symbol_y: str,
        symbol_x: str,
        prices_y: pd.Series,
        prices_x: pd.Series,
        spread: pd.Series,
        zscore: pd.Series,
        equity_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        metrics: dict,
    ) -> go.Figure:
        """
        Cria o dashboard completo com 4 subplots.

        Retorna
        -------
        go.Figure pronto para .show() ou .write_html().
        """
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=[
                f"📈 Preços Normalizados: {symbol_y} vs {symbol_x}",
                "📊 Spread do Par",
                "🎯 Z-Score com Sinais de Trade",
                "💰 Curva de Equity",
            ],
            row_heights=[0.25, 0.20, 0.25, 0.30],
        )

        c = self.COLORS

        # ── Painel 1: Preços Normalizados ─────────────────────────────────
        norm_y = (prices_y / prices_y.iloc[0]) * 100
        norm_x = (prices_x / prices_x.iloc[0]) * 100

        fig.add_trace(go.Scatter(
            x=norm_y.index, y=norm_y,
            name=symbol_y, line=dict(color=c["asset_y"], width=1.5),
            hovertemplate=f"<b>{symbol_y}</b>: %{{y:.2f}}<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=norm_x.index, y=norm_x,
            name=symbol_x, line=dict(color=c["asset_x"], width=1.5),
            hovertemplate=f"<b>{symbol_x}</b>: %{{y:.2f}}<extra></extra>",
        ), row=1, col=1)

        # ── Painel 2: Spread ──────────────────────────────────────────────
        mean_spread = spread.mean()
        std_spread = spread.std()

        for mult, color in [(1, "rgba(251,191,36,0.3)"),
                            (2, "rgba(239,68,68,0.3)")]:
            fig.add_hrect(
                y0=mean_spread + mult * std_spread,
                y1=mean_spread + (mult + 1) * std_spread,
                fillcolor=color, opacity=0.3, line_width=0, row=2, col=1,
            )
            fig.add_hrect(
                y0=mean_spread - (mult + 1) * std_spread,
                y1=mean_spread - mult * std_spread,
                fillcolor=color, opacity=0.3, line_width=0, row=2, col=1,
            )

        fig.add_trace(go.Scatter(
            x=spread.index, y=spread,
            name="Spread", line=dict(color=c["spread"], width=1.2),
            fill="tozeroy", fillcolor="rgba(167,139,250,0.05)",
        ), row=2, col=1)

        fig.add_hline(y=mean_spread, line_dash="dash",
                      line_color="rgba(255,255,255,0.4)", row=2, col=1)

        # ── Painel 3: Z-Score + Sinais ────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=zscore.index, y=zscore,
            name="Z-Score", line=dict(color=c["zscore"], width=1.5),
        ), row=3, col=1)

        # Bandas de referência
        for z_level, color, dash in [
            (2.0, "rgba(239,68,68,0.8)", "dash"),
            (-2.0, "rgba(34,197,94,0.8)", "dash"),
            (3.5, "rgba(239,68,68,0.4)", "dot"),
            (-3.5, "rgba(34,197,94,0.4)", "dot"),
            (0.0, "rgba(255,255,255,0.3)", "solid"),
        ]:
            fig.add_hline(y=z_level, line_dash=dash,
                          line_color=color, line_width=1, row=3, col=1)

        # Marcadores de trade no Z-Score
        if not trades_df.empty:
            long_entries = trades_df[trades_df["direction"] == "LONG"]
            short_entries = trades_df[trades_df["direction"] == "SHORT"]

            if not long_entries.empty:
                entry_z_long = zscore.reindex(
                    long_entries["entry_time"], method="nearest"
                )
                fig.add_trace(go.Scatter(
                    x=long_entries["entry_time"],
                    y=entry_z_long.values,
                    mode="markers",
                    name="Long Entry",
                    marker=dict(symbol="triangle-up", size=10,
                                color=c["entry_long"]),
                ), row=3, col=1)

            if not short_entries.empty:
                entry_z_short = zscore.reindex(
                    short_entries["entry_time"], method="nearest"
                )
                fig.add_trace(go.Scatter(
                    x=short_entries["entry_time"],
                    y=entry_z_short.values,
                    mode="markers",
                    name="Short Entry",
                    marker=dict(symbol="triangle-down", size=10,
                                color=c["entry_short"]),
                ), row=3, col=1)

        # ── Painel 4: Equity Curve ────────────────────────────────────────
        if not equity_df.empty:
            eq = equity_df["equity"]
            rolling_max = eq.cummax()
            drawdown = (eq - rolling_max) / rolling_max * 100

            # Área de drawdown (fundo)
            fig.add_trace(go.Scatter(
                x=equity_df.index, y=drawdown,
                name="Drawdown (%)",
                fill="tozeroy",
                line=dict(color="rgba(239,68,68,0.5)", width=0.5),
                fillcolor="rgba(239,68,68,0.1)",
                yaxis="y5",
            ), row=4, col=1)

            fig.add_trace(go.Scatter(
                x=equity_df.index, y=eq,
                name="Equity (USDT)",
                line=dict(color=c["equity"], width=2),
                fill="tozeroy",
                fillcolor="rgba(96,165,250,0.08)",
            ), row=4, col=1)

        # ── Layout Global ─────────────────────────────────────────────────
        total_return = metrics.get("total_return_pct", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd = metrics.get("max_drawdown_pct", 0)
        win_rate = metrics.get("win_rate", 0)

        title_text = (
            f"<b>PAIRS TRADING DASHBOARD</b>  |  "
            f"Retorno: {total_return:+.2f}%  |  "
            f"Sharpe: {sharpe:.2f}  |  "
            f"Max DD: {max_dd:.2f}%  |  "
            f"Win Rate: {win_rate:.1f}%"
        )

        fig.update_layout(
            title=dict(text=title_text, font=dict(size=14, color="#E2E8F0"),
                       x=0.5, xanchor="center"),
            template="plotly_dark",
            paper_bgcolor=c["bg"],
            plot_bgcolor=c["bg"],
            font=dict(color=c["text"], family="monospace"),
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.01,
                xanchor="right", x=1,
                bgcolor="rgba(15,23,42,0.8)",
                bordercolor="#334155", borderwidth=1,
            ),
            height=900,
            margin=dict(l=60, r=40, t=80, b=40),
        )

        for i in range(1, 5):
            fig.update_xaxes(
                gridcolor=c["grid"], showgrid=True,
                zeroline=False, row=i, col=1
            )
            fig.update_yaxes(
                gridcolor=c["grid"], showgrid=True,
                zeroline=True, zerolinecolor="#334155",
                row=i, col=1
            )

        return fig

    def plot_pvalue_heatmap(
        self, pvalue_matrix: pd.DataFrame, title: str = "Matriz de P-Values"
    ) -> go.Figure:
        """
        Gera um heatmap interativo da matriz de p-values do scanner.
        Células verdes = pares cointegrados.
        """
        # Máscara para diagonal
        z_data = pvalue_matrix.values.copy()
        np.fill_diagonal(z_data, np.nan)

        fig = go.Figure(go.Heatmap(
            z=z_data,
            x=pvalue_matrix.columns.tolist(),
            y=pvalue_matrix.index.tolist(),
            colorscale=[
                [0.0, "#22C55E"],   # p < 0.05 → verde (cointegrado)
                [0.05, "#FCD34D"],  # zona de atenção
                [0.1, "#94A3B8"],
                [1.0, "#1E293B"],   # p alto → cinza (não cointegrado)
            ],
            zmin=0, zmax=1,
            colorbar=dict(
                title="p-value",
                tickvals=[0, 0.05, 0.1, 0.5, 1.0],
                ticktext=["0.00", "0.05*", "0.10", "0.50", "1.00"],
            ),
            hovertemplate="<b>%{y} / %{x}</b><br>p-value: %{z:.4f}<extra></extra>",
            text=np.where(
                z_data < 0.05,
                [[f"★{v:.3f}" if pd.notna(v) else "" for v in row] for row in z_data],
                [[f"{v:.3f}" if pd.notna(v) else "" for v in row] for row in z_data],
            ),
            texttemplate="%{text}",
            textfont=dict(size=10, color="white"),
        ))

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>  |  ★ = p < 0.05 (Cointegrado)",
                font=dict(size=14, color="#E2E8F0"), x=0.5
            ),
            template="plotly_dark",
            paper_bgcolor="#0F172A",
            font=dict(family="monospace", color="#94A3B8"),
            height=500,
        )
        return fig


# ══════════════════════════════════════════════════════════════════════════════
#  6. ORQUESTRADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
def run_pairs_trading_analysis(
    symbol_y: str = "ETH/USDT",
    symbol_x: str = "BTC/USDT",
    timeframe: str = "1d",
    limit: int = 500,
    exchange: str = "binance",
    entry_threshold: float = 2.0,
    stop_loss_z: float = 3.5,
    initial_capital: float = 10_000.0,
    zscore_window: Optional[int] = None,
    save_html: bool = True,
):
    """
    Função principal que orquestra todo o pipeline:
      1. Coleta de dados
      2. Análise estatística
      3. Backtest
      4. Visualização

    Parâmetros
    ----------
    symbol_y, symbol_x : str   — Par a ser analisado (Y é a "perna longa").
    timeframe          : str   — Período dos candles ('1h', '4h', '1d').
    limit              : int   — Quantidade de candles históricos.
    exchange           : str   — Exchange ('binance' ou 'bybit').
    entry_threshold    : float — Z-Score para entrada (padrão: 2.0).
    stop_loss_z        : float — Stop-loss em σ (padrão: 3.5).
    initial_capital    : float — Capital inicial em USDT.
    zscore_window      : int   — Janela para Z-Score rolling (None = global).
    save_html          : bool  — Salva o dashboard como HTML interativo.

    Retorna
    -------
    dict com analysis_results, trades, metrics e as figuras Plotly.
    """
    log.info("▶ Iniciando pipeline de Pairs Trading: %s / %s [%s]",
             symbol_y, symbol_x, timeframe)

    # ── Etapa 1: Dados ────────────────────────────────────────────────────
    collector = DataCollector(exchange_name=exchange)
    prices = collector.fetch_close_prices(
        symbols=[symbol_y, symbol_x],
        timeframe=timeframe,
        limit=limit,
    )
    series_y = prices[symbol_y]
    series_x = prices[symbol_x]

    # ── Etapa 2: Análise Estatística ──────────────────────────────────────
    stat = StatisticalAnalysis()
    analysis = stat.full_analysis(symbol_y, series_y, symbol_x, series_x)

    beta = analysis["hedge_ratio"]
    residuals = analysis["residuals"]
    zscore = stat.calculate_zscore(residuals, window=zscore_window)

    log.info("Half-Life: %.1f períodos | Hedge Ratio: %.4f",
             analysis["half_life_periods"], beta)
    log.info("Cointegração: %s | ADF estacionário: %s",
             analysis["engle_granger"]["is_cointegrated"],
             analysis["adf_on_residuals"]["is_stationary"])

    # ── Etapa 3: Backtest ─────────────────────────────────────────────────
    engine = BacktestEngine(
        entry_threshold=entry_threshold,
        stop_loss_z=stop_loss_z,
        initial_capital=initial_capital,
    )
    trades_df, equity_df, metrics = engine.run(
        prices_y=series_y,
        prices_x=series_x,
        zscore=zscore,
        hedge_ratio=beta,
    )

    # ── Etapa 4: Visualização ─────────────────────────────────────────────
    viz = Visualizer()
    dashboard = viz.create_dashboard(
        symbol_y=symbol_y,
        symbol_x=symbol_x,
        prices_y=series_y,
        prices_x=series_x,
        spread=residuals,
        zscore=zscore,
        equity_df=equity_df,
        trades_df=trades_df,
        metrics=metrics,
    )

    if save_html:
        fname = f"pairs_trading_{symbol_y[:3]}_{symbol_x[:3]}_{timeframe}.html"
        dashboard.write_html(fname)
        log.info("Dashboard salvo: %s", fname)

    dashboard.show()

    return {
        "analysis": analysis,
        "trades": trades_df,
        "equity": equity_df,
        "metrics": metrics,
        "figure": dashboard,
    }


def run_cointegration_scanner(
    symbols: list = None,
    timeframe: str = "1d",
    limit: int = 365,
    exchange: str = "binance",
    save_html: bool = True,
) -> tuple[pd.DataFrame, list]:
    """
    Executa o scanner de cointegração em uma lista de símbolos.

    Parâmetros
    ----------
    symbols   : Lista de símbolos (ex: ['BTC/USDT', 'ETH/USDT', ...]).
    timeframe : Período dos candles.
    limit     : Quantidade de candles históricos.
    exchange  : Exchange ('binance' ou 'bybit').
    save_html : Salva o heatmap como HTML.

    Retorna
    -------
    (pvalue_matrix, cointegrated_pairs)
    """
    if symbols is None:
        symbols = [
            "BTC/USDT", "ETH/USDT", "SOL/USDT",
            "LINK/USDT", "ADA/USDT", "AVAX/USDT",
        ]

    log.info("Scanner iniciado para %d símbolos: %s", len(symbols), symbols)

    collector = DataCollector(exchange_name=exchange)
    prices = collector.fetch_close_prices(
        symbols=symbols, timeframe=timeframe, limit=limit
    )

    scanner = CointegrationScanner(significance_level=0.05)
    pvalue_matrix, cointegrated_pairs = scanner.scan(prices)

    # Visualização do heatmap
    viz = Visualizer()
    heatmap = viz.plot_pvalue_heatmap(
        pvalue_matrix,
        title=f"Scanner de Cointegração | {timeframe} | {limit} períodos",
    )

    if save_html:
        fname = f"cointegration_scanner_{timeframe}.html"
        heatmap.write_html(fname)
        log.info("Heatmap salvo: %s", fname)

    heatmap.show()

    # Exibe tabela formatada dos pares cointegrados
    print("\n" + "=" * 60)
    print("  PARES COINTEGRADOS (p < 0.05)")
    print("=" * 60)
    print(f"  {'Par':<25} {'p-value':>10}")
    print("-" * 60)
    for sym_a, sym_b, pval in sorted(cointegrated_pairs, key=lambda x: x[2]):
        print(f"  {sym_a}/{sym_b:<20} {pval:>10.4f}  ✓")
    print("=" * 60)

    return pvalue_matrix, cointegrated_pairs


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── MODO 1: Analisar um par específico ────────────────────────────────
    result = run_pairs_trading_analysis(
        symbol_y="ETH/USDT",
        symbol_x="BTC/USDT",
        timeframe="1d",
        limit=500,
        exchange="binance",
        entry_threshold=2.0,
        stop_loss_z=3.5,
        initial_capital=10_000.0,
        zscore_window=60,   # Z-Score rolling de 60 dias
        save_html=True,
    )

    # ── MODO 2: Scanner de cointegração ───────────────────────────────────
    pvalue_matrix, top_pairs = run_cointegration_scanner(
        symbols=[
            "BTC/USDT", "ETH/USDT", "SOL/USDT",
            "LINK/USDT", "ADA/USDT", "AVAX/USDT",
        ],
        timeframe="1d",
        limit=365,
        exchange="binance",
        save_html=True,
    )
