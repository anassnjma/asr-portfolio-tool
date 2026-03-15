"""Portfolio model – all financial calculations and data storage."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, skew, kurtosis

from models.market_data import (
    fetch_asset_names,
    fetch_current_prices,
    fetch_historical_prices,
)
from utils.config import SIMULATION_PATHS, SIMULATION_YEARS, TRADING_DAYS_PER_YEAR, BOOTSTRAP_BLOCK_SIZE

RISK_FREE_RATE = 0.04

STRESS_SCENARIOS = {
    "Global Financial Crisis (2008)": ("2008-09-01", "2009-03-09"),
    "COVID-19 Crash (2020)": ("2020-02-19", "2020-03-23"),
    "Rate Hike Sell-off (2022)": ("2022-01-03", "2022-10-12"),
    "EU Debt Crisis (2011)": ("2011-07-01", "2011-11-25"),
    "Volmageddon (Feb 2018)": ("2018-01-26", "2018-02-08"),
}


@dataclass
class Asset:
    """A single holding in the portfolio."""

    ticker: str
    sector: str
    asset_class: str
    quantity: float
    purchase_price: float

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper().strip()
        self.sector = self.sector.strip()
        self.asset_class = self.asset_class.strip()
        self.quantity = float(self.quantity)
        self.purchase_price = float(self.purchase_price)

    @property
    def transaction_value(self) -> float:
        """Original cost basis (quantity x purchase price)."""
        return round(self.quantity * self.purchase_price, 2)


class Portfolio:
    """Collection of assets with financial calculation methods."""

    def __init__(self, assets: list[Asset] | None = None) -> None:
        self.assets: list[Asset] = assets or []
        self._current_prices: dict[str, float | None] = {}
        self._asset_names: dict[str, str] = {}

    @classmethod
    def from_file(cls, path: str) -> Portfolio:
        """Load a portfolio from CSV, JSON, or Excel."""
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Portfolio file not found: {path}")

        ext = p.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(p)
        elif ext == ".json":
            df = pd.DataFrame(json.loads(p.read_text()))
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(p)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use CSV, JSON, or Excel.")

        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        required = {"ticker", "sector", "asset_class", "quantity", "purchase_price"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}.")

        assets = [
            Asset(
                ticker=row["ticker"], sector=row["sector"],
                asset_class=row["asset_class"], quantity=row["quantity"],
                purchase_price=row["purchase_price"],
            )
            for _, row in df.iterrows()
        ]
        return cls(assets)

    @property
    def tickers(self) -> list[str]:
        return [a.ticker for a in self.assets]

    def __len__(self) -> int:
        return len(self.assets)

    def refresh_market_data(self) -> None:
        self._current_prices = fetch_current_prices(self.tickers)
        self._asset_names = fetch_asset_names(self.tickers)

    def current_price(self, ticker: str) -> float | None:
        if not self._current_prices:
            self.refresh_market_data()
        return self._current_prices.get(ticker)

    def asset_name(self, ticker: str) -> str:
        if not self._asset_names:
            self.refresh_market_data()
        return self._asset_names.get(ticker, ticker)

    def total_value(self) -> float:
        return round(sum(a.quantity * (self.current_price(a.ticker) or 0) for a in self.assets), 2)

    def total_cost(self) -> float:
        return round(sum(a.transaction_value for a in self.assets), 2)

    def _get_weights_and_returns(self, period: str = "5y") -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Shared helper: compute portfolio weights and weighted daily returns."""
        prices = fetch_historical_prices(self.tickers, period=period)
        daily_returns = prices.pct_change().dropna()
        if not self._current_prices:
            self.refresh_market_data()
        market_values = np.array([a.quantity * (self.current_price(a.ticker) or 0) for a in self.assets])
        total = market_values.sum()
        if total == 0:
            raise ValueError("Portfolio has zero market value.")
        weights = market_values / total
        common = [t for t in self.tickers if t in daily_returns.columns]
        w = np.array([weights[self.tickers.index(t)] for t in common])
        return w, daily_returns[common].values @ w, common

    def historical_returns(self, freq: str = "M") -> pd.DataFrame:
        """Compute periodic percentage returns for each asset."""
        prices = fetch_historical_prices(self.tickers)
        resampled = prices.resample(freq).last()
        returns = resampled.pct_change().dropna(how="all") * 100
        returns = returns.round(2)
        fmt = {"M": "%Y-%m", "ME": "%Y-%m", "QE": "%Y-Q%q", "Q": "%Y-Q%q", "YE": "%Y", "Y": "%Y"}
        returns.index = returns.index.strftime(fmt.get(freq, "%Y-%m"))
        return returns

    def insights_table(self) -> pd.DataFrame:
        """Detailed per-asset overview with current market data."""
        if not self._current_prices:
            self.refresh_market_data()
        rows = []
        for a in self.assets:
            cp = self.current_price(a.ticker)
            current_val = round(a.quantity * cp, 2) if cp else None
            pnl = round(current_val - a.transaction_value, 2) if current_val else None
            pnl_pct = (
                round((current_val / a.transaction_value - 1) * 100, 2)
                if current_val and a.transaction_value else None
            )
            rows.append({
                "Ticker": a.ticker, "Name": self.asset_name(a.ticker),
                "Sector": a.sector, "Asset Class": a.asset_class,
                "Qty": a.quantity, "Buy Price": a.purchase_price,
                "Cost Basis": a.transaction_value, "Price Now": cp,
                "Market Value": current_val, "P&L ($)": pnl, "P&L (%)": pnl_pct,
            })
        return pd.DataFrame(rows)

    def weights_table(self, group_by: str | None = None) -> pd.DataFrame:
        """Portfolio weights, optionally grouped by sector or asset_class."""
        df = self.insights_table()
        total = df["Market Value"].sum()
        if group_by in ("sector", "asset_class"):
            col = "Sector" if group_by == "sector" else "Asset Class"
            grouped = df.groupby(col).agg(
                Cost_Basis=("Cost Basis", "sum"), Market_Value=("Market Value", "sum")).reset_index()
            grouped.columns = [col, "Cost Basis", "Market Value"]
            grouped["Weight (%)"] = (grouped["Market Value"] / total * 100).round(2)
            grouped["P&L ($)"] = (grouped["Market Value"] - grouped["Cost Basis"]).round(2)
            totals = pd.DataFrame([{col: "TOTAL", "Cost Basis": grouped["Cost Basis"].sum(),
                "Market Value": grouped["Market Value"].sum(), "Weight (%)": 100.0,
                "P&L ($)": grouped["P&L ($)"].sum()}])
            return pd.concat([grouped, totals], ignore_index=True)
        result = df[["Ticker", "Name", "Cost Basis", "Market Value"]].copy()
        result["Weight (%)"] = (result["Market Value"] / total * 100).round(2)
        result["P&L ($)"] = (result["Market Value"] - result["Cost Basis"]).round(2)
        totals = pd.DataFrame([{"Ticker": "TOTAL", "Name": "", "Cost Basis": result["Cost Basis"].sum(),
            "Market Value": result["Market Value"].sum(), "Weight (%)": 100.0,
            "P&L ($)": result["P&L ($)"].sum()}])
        return pd.concat([result, totals], ignore_index=True)

    def simulate(self, years=SIMULATION_YEARS, n_paths=SIMULATION_PATHS,
                 block_size=BOOTSTRAP_BLOCK_SIZE):
        """Block Bootstrap simulation. Resamples actual historical returns in blocks."""
        from scipy.stats import skew, kurtosis

        _, port_returns, _ = self._get_weights_and_returns()
        initial_value = float(self.total_value())
        total_days = years * TRADING_DAYS_PER_YEAR
        n_history = len(port_returns)

        if n_history < block_size:
            raise ValueError(f"Not enough history ({n_history} days) for block size ({block_size}).")

        # Pick random block start positions for all paths at once
        rng = np.random.default_rng(42)
        blocks_needed = int(np.ceil(total_days / block_size))
        starts = rng.integers(0, n_history - block_size + 1, size=(n_paths, blocks_needed))

        # Build index array: for each block start, generate [start, start+1, ..., start+block_size-1]
        offsets = np.arange(block_size)
        block_indices = starts[:, :, None] + offsets[None, None, :]  # (n_paths, blocks_needed, block_size)
        # Flatten blocks into one long sequence per path, then trim to total_days
        flat_indices = block_indices.reshape(n_paths, -1)[:, :total_days]
        # Gather all returns at once — no Python loop
        all_returns = port_returns[flat_indices]

        cumulative = np.cumprod(1 + all_returns, axis=1) * initial_value
        final_values = cumulative[:, -1]
        pct_keys = [5, 10, 25, 50, 75, 90, 95]
        pcts = dict(zip(pct_keys, np.percentile(final_values, pct_keys)))
        prob_loss = float(np.mean(final_values < initial_value)) * 100

        mu = float(np.mean(port_returns))
        sigma = float(np.std(port_returns))
        annual_mu = ((1 + mu) ** TRADING_DAYS_PER_YEAR - 1) * 100
        annual_sigma = sigma * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

        stats = pd.DataFrame({
            "Metric": ["Initial Portfolio Value", "Method",
                "Annualised Return (historical)", "Annualised Volatility (historical)",
                "Historical Skewness", "Historical Excess Kurtosis",
                f"Mean Value ({years}yr)", f"Median Value ({years}yr)",
                "5th Percentile (worst case)", "25th Percentile",
                "75th Percentile", "95th Percentile (best case)", "Probability of Loss"],
            "Value": [f"${initial_value:,.2f}", f"Block Bootstrap (block={block_size}d, paths={n_paths:,})",
                f"{annual_mu:.1f}%", f"{annual_sigma:.1f}%",
                f"{float(skew(port_returns)):.3f}", f"{float(kurtosis(port_returns)):.3f}",
                f"${np.mean(final_values):,.2f}", f"${pcts[50]:,.2f}",
                f"${pcts[5]:,.2f}", f"${pcts[25]:,.2f}",
                f"${pcts[75]:,.2f}", f"${pcts[95]:,.2f}", f"{prob_loss:.1f}%"],
        })
        return {"paths": cumulative, "percentiles": pcts, "stats": stats,
            "initial_value": initial_value, "mean_final": float(np.mean(final_values)),
            "median_final": float(pcts[50]), "annual_mu": annual_mu, "annual_sigma": annual_sigma}

    def risk_metrics(self) -> pd.DataFrame:
        """Sharpe, Sortino, max drawdown, skewness, excess kurtosis."""
        _, port_returns, _ = self._get_weights_and_returns()
        ann_return = float(np.mean(port_returns)) * TRADING_DAYS_PER_YEAR * 100
        ann_vol = float(np.std(port_returns)) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        sharpe = (ann_return / 100 - RISK_FREE_RATE) / (ann_vol / 100) if ann_vol > 0 else 0
        cumulative = (1 + port_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        max_dd = float(np.min((cumulative - running_max) / running_max)) * 100
        downside = port_returns[port_returns < 0]
        downside_std = float(np.std(downside)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        sortino = (np.mean(port_returns) * TRADING_DAYS_PER_YEAR - RISK_FREE_RATE) / downside_std if downside_std > 0 else 0
        return pd.DataFrame({
            "Metric": ["Annualised Return", "Annualised Volatility",
                "Sharpe Ratio (rf=4%)", "Sortino Ratio (rf=4%)",
                "Max Drawdown", "Skewness", "Excess Kurtosis", "Best Day", "Worst Day"],
            "Value": [f"{ann_return:.2f}%", f"{ann_vol:.2f}%", f"{sharpe:.2f}", f"{sortino:.2f}",
                f"{max_dd:.2f}%", f"{float(skew(port_returns)):.3f}",
                f"{float(kurtosis(port_returns)):.3f}",
                f"+{float(np.max(port_returns)) * 100:.2f}%",
                f"{float(np.min(port_returns)) * 100:.2f}%"],
        })

    def value_at_risk(self, confidence: float = 0.95) -> dict:
        """VaR and CVaR via historical, parametric, and Cornish-Fisher methods."""
        _, port_returns, _ = self._get_weights_and_returns()
        portfolio_value = self.total_value()
        alpha = 1 - confidence
        hist_var_pct = float(np.percentile(port_returns, alpha * 100))
        tail = port_returns[port_returns <= hist_var_pct]
        hist_cvar_pct = float(np.mean(tail)) if len(tail) > 0 else hist_var_pct
        mu = float(np.mean(port_returns))
        sigma = float(np.std(port_returns))
        z = norm.ppf(alpha)
        param_var_pct = mu + z * sigma
        param_cvar_pct = mu - sigma * norm.pdf(z) / alpha
        s = float(skew(port_returns))
        k = float(kurtosis(port_returns))
        z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * k / 24 - (2*z**3 - 5*z) * s**2 / 36
        cf_var_pct = mu + z_cf * sigma
        stats = pd.DataFrame({
            "Method": ["Historical", "Historical", "Parametric (Normal)", "Parametric (Normal)", "Cornish-Fisher"],
            "Measure": [f"VaR ({confidence:.0%})", f"CVaR / ES ({confidence:.0%})",
                        f"VaR ({confidence:.0%})", f"CVaR / ES ({confidence:.0%})", f"VaR ({confidence:.0%})"],
            "Daily (%)": [f"{v * 100:.4f}%" for v in [hist_var_pct, hist_cvar_pct, param_var_pct, param_cvar_pct, cf_var_pct]],
            "Daily ($)": [f"${v * portfolio_value:,.2f}" for v in [hist_var_pct, hist_cvar_pct, param_var_pct, param_cvar_pct, cf_var_pct]],
        })
        return {"stats": stats, "returns": port_returns,
            "historical_var": hist_var_pct, "historical_cvar": hist_cvar_pct,
            "parametric_var": param_var_pct, "parametric_cvar": param_cvar_pct,
            "cornish_fisher_var": cf_var_pct, "confidence": confidence, "portfolio_value": portfolio_value}

    def stress_test(self) -> pd.DataFrame:
        """Test portfolio during major historical crises."""
        SCENARIOS = {
            "Global Financial Crisis (2008)": ("2008-09-01", "2009-03-09"),
            "COVID-19 Crash (2020)": ("2020-02-19", "2020-03-23"),
            "Rate Hike Sell-off (2022)": ("2022-01-03", "2022-10-12"),
            "EU Debt Crisis (2011)": ("2011-07-01", "2011-11-25"),
            "Volmageddon (Feb 2018)": ("2018-01-26", "2018-02-08"),
        }
        prices = fetch_historical_prices(self.tickers, period="max")
        daily_returns = prices.pct_change().dropna()
        if not self._current_prices:
            self.refresh_market_data()
        mv = np.array([a.quantity * (self.current_price(a.ticker) or 0) for a in self.assets])
        weights = mv / mv.sum()
        common = [t for t in self.tickers if t in daily_returns.columns]
        w = np.array([weights[self.tickers.index(t)] for t in common])
        rows = []
        for name, (start, end) in SCENARIOS.items():
            try:
                period_ret = daily_returns[common].loc[start:end]
                if period_ret.empty:
                    rows.append({"Scenario": name, "Period": f"{start} to {end}",
                        "Days": "N/A", "Cumulative Return": "No data",
                        "Max Drawdown": "No data", "Worst Day": "No data"})
                    continue
                port_ret = period_ret.values @ w
                cum = float(np.prod(1 + port_ret) - 1) * 100
                cumulative = np.cumprod(1 + port_ret)
                peak = np.maximum.accumulate(cumulative)
                max_dd = float(np.min((cumulative - peak) / peak)) * 100
                rows.append({"Scenario": name, "Period": f"{start} to {end}",
                    "Days": len(port_ret), "Cumulative Return": f"{cum:+.2f}%",
                    "Max Drawdown": f"{max_dd:.2f}%",
                    "Worst Day": f"{float(np.min(port_ret)) * 100:.2f}%"})
            except Exception:
                rows.append({"Scenario": name, "Period": f"{start} to {end}",
                    "Days": "N/A", "Cumulative Return": "Error",
                    "Max Drawdown": "Error", "Worst Day": "Error"})
        return pd.DataFrame(rows)

    def risk_parity(self) -> dict:
        """Equal risk contribution allocation via scipy optimisation.

        More robust than Markowitz: only needs covariance matrix, not
        return estimates. Standard at European pension funds.
        """
        prices = fetch_historical_prices(self.tickers, period="5y")
        daily_returns = prices.pct_change().dropna()
        common = [t for t in self.tickers if t in daily_returns.columns]
        cov = daily_returns[common].cov().values * TRADING_DAYS_PER_YEAR
        n = len(common)
        def risk_contributions(w):
            port_vol = np.sqrt(w @ cov @ w)
            return (w * (cov @ w)) / port_vol
        def objective(w):
            rc = risk_contributions(w)
            target = np.sqrt(w @ cov @ w) / n
            return float(np.sum((rc - target) ** 2))
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.01, 1.0)] * n
        result = minimize(objective, np.ones(n) / n, method="SLSQP", bounds=bounds, constraints=cons)
        w_rp = result.x
        if not self._current_prices:
            self.refresh_market_data()
        mv = np.array([a.quantity * (self.current_price(a.ticker) or 0) for a in self.assets])
        w_cur = np.zeros(n)
        for i, t in enumerate(common):
            w_cur[i] = mv[self.tickers.index(t)]
        w_cur = w_cur / w_cur.sum() if w_cur.sum() > 0 else w_cur
        rc_current = risk_contributions(w_cur)
        rc_rp = risk_contributions(w_rp)
        mu = daily_returns[common].mean().values * TRADING_DAYS_PER_YEAR
        def _stats(w):
            ret = float(w @ mu) * 100
            vol = float(np.sqrt(w @ cov @ w)) * 100
            sharpe = (ret / 100 - RISK_FREE_RATE) / (vol / 100) if vol > 0 else 0
            return ret, vol, sharpe
        cur_ret, cur_vol, cur_sharpe = _stats(w_cur)
        rp_ret, rp_vol, rp_sharpe = _stats(w_rp)
        return {
            "weights": pd.DataFrame({"Ticker": common, "Current (%)": (w_cur * 100).round(2),
                "Risk Parity (%)": (w_rp * 100).round(2), "Difference (%)": ((w_rp - w_cur) * 100).round(2)}),
            "risk_contributions": pd.DataFrame({"Ticker": common,
                "Current RC (%)": (rc_current / rc_current.sum() * 100).round(2),
                "Risk Parity RC (%)": (rc_rp / rc_rp.sum() * 100).round(2)}),
            "summary": pd.DataFrame({"Portfolio": ["Current", "Risk Parity"],
                "Return (%)": [f"{cur_ret:.2f}", f"{rp_ret:.2f}"],
                "Volatility (%)": [f"{cur_vol:.2f}", f"{rp_vol:.2f}"],
                "Sharpe Ratio": [f"{cur_sharpe:.2f}", f"{rp_sharpe:.2f}"]}),
            "current_weights": w_cur, "rp_weights": w_rp, "tickers": common}
