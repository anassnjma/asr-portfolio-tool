"""Portfolio model – all financial calculations and data storage."""
from __future__ import annotations
import json, pathlib
from dataclasses import dataclass
import numpy as np, pandas as pd
from models.market_data import fetch_asset_names, fetch_current_prices, fetch_historical_prices
from utils.config import SIMULATION_PATHS, SIMULATION_YEARS, TRADING_DAYS_PER_YEAR, BOOTSTRAP_BLOCK_SIZE
RISK_FREE_RATE = 0.04
@dataclass
class Asset:
    ticker: str; sector: str; asset_class: str; quantity: float; purchase_price: float
    def __post_init__(self):
        self.ticker = self.ticker.upper().strip(); self.sector = self.sector.strip()
        self.asset_class = self.asset_class.strip()
        self.quantity = float(self.quantity); self.purchase_price = float(self.purchase_price)
    @property
    def transaction_value(self): return round(self.quantity * self.purchase_price, 2)
class Portfolio:
    def __init__(self, assets=None):
        self.assets = assets or []; self._current_prices = {}; self._asset_names = {}
    @classmethod
    def from_file(cls, path):
        p = pathlib.Path(path)
        if not p.exists(): raise FileNotFoundError(f"Portfolio file not found: {path}")
        ext = p.suffix.lower()
        if ext == ".csv": df = pd.read_csv(p)
        elif ext == ".json": df = pd.DataFrame(json.loads(p.read_text()))
        elif ext in (".xlsx", ".xls"): df = pd.read_excel(p)
        else: raise ValueError(f"Unsupported format: {ext}")
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        required = {"ticker", "sector", "asset_class", "quantity", "purchase_price"}
        missing = required - set(df.columns)
        if missing: raise ValueError(f"Missing columns: {missing}")
        return cls([Asset(ticker=r["ticker"], sector=r["sector"], asset_class=r["asset_class"],
            quantity=r["quantity"], purchase_price=r["purchase_price"]) for _, r in df.iterrows()])
    @property
    def tickers(self): return [a.ticker for a in self.assets]
    def __len__(self): return len(self.assets)
    def refresh_market_data(self):
        self._current_prices = fetch_current_prices(self.tickers)
        self._asset_names = fetch_asset_names(self.tickers)
    def current_price(self, t):
        if not self._current_prices: self.refresh_market_data()
        return self._current_prices.get(t)
    def asset_name(self, t):
        if not self._asset_names: self.refresh_market_data()
        return self._asset_names.get(t, t)
    def total_value(self):
        return round(sum(a.quantity * (self.current_price(a.ticker) or 0) for a in self.assets), 2)
    def total_cost(self): return round(sum(a.transaction_value for a in self.assets), 2)
    def _get_weights_and_returns(self, period="5y"):
        prices = fetch_historical_prices(self.tickers, period=period)
        dr = prices.pct_change().dropna()
        if not self._current_prices: self.refresh_market_data()
        mv = np.array([a.quantity * (self.current_price(a.ticker) or 0) for a in self.assets])
        total = mv.sum()
        if total == 0: raise ValueError("Portfolio has zero market value.")
        w_all = mv / total
        common = [t for t in self.tickers if t in dr.columns]
        w = np.array([w_all[self.tickers.index(t)] for t in common])
        return w, dr[common].values @ w, common
    def historical_returns(self, freq="M"):
        prices = fetch_historical_prices(self.tickers)
        ret = prices.resample(freq).last().pct_change().dropna(how="all") * 100
        ret = ret.round(2)
        fmt = {"M": "%Y-%m", "ME": "%Y-%m", "QE": "%Y-Q%q", "Q": "%Y-Q%q", "YE": "%Y", "Y": "%Y"}
        ret.index = ret.index.strftime(fmt.get(freq, "%Y-%m"))
        return ret

    def insights_table(self):
        if not self._current_prices: self.refresh_market_data()
        rows = []
        for a in self.assets:
            cp = self.current_price(a.ticker)
            cv = round(a.quantity * cp, 2) if cp else None
            pnl = round(cv - a.transaction_value, 2) if cv else None
            pp = round((cv / a.transaction_value - 1) * 100, 2) if cv and a.transaction_value else None
            rows.append({"Ticker": a.ticker, "Name": self.asset_name(a.ticker), "Sector": a.sector,
                "Asset Class": a.asset_class, "Qty": a.quantity, "Buy Price": a.purchase_price,
                "Cost Basis": a.transaction_value, "Price Now": cp, "Market Value": cv,
                "P&L ($)": pnl, "P&L (%)": pp})
        return pd.DataFrame(rows)

    def weights_table(self, group_by=None):
        df = self.insights_table(); total = df["Market Value"].sum()
        if group_by in ("sector", "asset_class"):
            col = "Sector" if group_by == "sector" else "Asset Class"
            g = df.groupby(col).agg(Cost_Basis=("Cost Basis", "sum"), Market_Value=("Market Value", "sum")).reset_index()
            g.columns = [col, "Cost Basis", "Market Value"]
            g["Weight (%)"] = (g["Market Value"] / total * 100).round(2)
            g["P&L ($)"] = (g["Market Value"] - g["Cost Basis"]).round(2)
            t = pd.DataFrame([{col: "TOTAL", "Cost Basis": g["Cost Basis"].sum(),
                "Market Value": g["Market Value"].sum(), "Weight (%)": 100.0, "P&L ($)": g["P&L ($)"].sum()}])
            return pd.concat([g, t], ignore_index=True)
        r = df[["Ticker", "Name", "Cost Basis", "Market Value"]].copy()
        r["Weight (%)"] = (r["Market Value"] / total * 100).round(2)
        r["P&L ($)"] = (r["Market Value"] - r["Cost Basis"]).round(2)
        t = pd.DataFrame([{"Ticker": "TOTAL", "Name": "", "Cost Basis": r["Cost Basis"].sum(),
            "Market Value": r["Market Value"].sum(), "Weight (%)": 100.0, "P&L ($)": r["P&L ($)"].sum()}])
        return pd.concat([r, t], ignore_index=True)

    def simulate(self, years=SIMULATION_YEARS, n_paths=SIMULATION_PATHS, block_size=BOOTSTRAP_BLOCK_SIZE):
        """Block Bootstrap simulation. Resamples actual historical returns in blocks."""
        from scipy.stats import skew, kurtosis
        _, port_returns, _ = self._get_weights_and_returns()
        initial_value = float(self.total_value()); total_days = years * TRADING_DAYS_PER_YEAR
        n_history = len(port_returns)
        if n_history < block_size: raise ValueError(f"Not enough history for block size.")
        blocks_per_path = int(np.ceil(total_days / block_size))
        rng = np.random.default_rng(42); max_start = n_history - block_size
        start_indices = rng.integers(0, max_start + 1, size=(n_paths, blocks_per_path))
        all_returns = np.empty((n_paths, total_days))
        for b in range(blocks_per_path):
            cs = b * block_size; ce = min(cs + block_size, total_days)
            if cs >= total_days: break
            starts = start_indices[:, b]
            for i in range(ce - cs): all_returns[:, cs + i] = port_returns[starts + i]
        cumulative = np.cumprod(1 + all_returns, axis=1) * initial_value
        fv = cumulative[:, -1]; pct_keys = [5, 10, 25, 50, 75, 90, 95]
        pcts = dict(zip(pct_keys, np.percentile(fv, pct_keys)))
        prob_loss = float(np.mean(fv < initial_value)) * 100
        mu = float(np.mean(port_returns)); sigma = float(np.std(port_returns))
        amu = ((1 + mu) ** TRADING_DAYS_PER_YEAR - 1) * 100; asig = sigma * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        stats = pd.DataFrame({"Metric": ["Initial Portfolio Value", "Method",
            "Annualised Return (historical)", "Annualised Volatility (historical)",
            "Historical Skewness", "Historical Excess Kurtosis",
            f"Mean Value ({years}yr)", f"Median Value ({years}yr)",
            "5th Percentile (worst case)", "25th Percentile", "75th Percentile",
            "95th Percentile (best case)", "Probability of Loss"],
            "Value": [f"${initial_value:,.2f}", f"Block Bootstrap (block={block_size}d, paths={n_paths:,})",
            f"{amu:.1f}%", f"{asig:.1f}%", f"{float(skew(port_returns)):.3f}",
            f"{float(kurtosis(port_returns)):.3f}", f"${np.mean(fv):,.2f}", f"${pcts[50]:,.2f}",
            f"${pcts[5]:,.2f}", f"${pcts[25]:,.2f}", f"${pcts[75]:,.2f}", f"${pcts[95]:,.2f}",
            f"{prob_loss:.1f}%"]})
        return {"paths": cumulative, "percentiles": pcts, "stats": stats, "initial_value": initial_value,
            "mean_final": float(np.mean(fv)), "median_final": float(pcts[50]), "annual_mu": amu, "annual_sigma": asig}

    def risk_metrics(self):
        from scipy.stats import skew, kurtosis
        _, pr, _ = self._get_weights_and_returns()
        ar = float(np.mean(pr)) * TRADING_DAYS_PER_YEAR * 100
        av = float(np.std(pr)) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        sh = (ar / 100 - RISK_FREE_RATE) / (av / 100) if av > 0 else 0
        c = (1 + pr).cumprod(); rm = np.maximum.accumulate(c)
        mdd = float(np.min((c - rm) / rm)) * 100
        ds = pr[pr < 0]; dstd = float(np.std(ds)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        so = (np.mean(pr) * TRADING_DAYS_PER_YEAR - RISK_FREE_RATE) / dstd if dstd > 0 else 0
        return pd.DataFrame({"Metric": ["Annualised Return", "Annualised Volatility",
            "Sharpe Ratio (rf=4%)", "Sortino Ratio (rf=4%)", "Max Drawdown", "Skewness",
            "Excess Kurtosis", "Best Day", "Worst Day"],
            "Value": [f"{ar:.2f}%", f"{av:.2f}%", f"{sh:.2f}", f"{so:.2f}", f"{mdd:.2f}%",
            f"{float(skew(pr)):.3f}", f"{float(kurtosis(pr)):.3f}",
            f"+{float(np.max(pr))*100:.2f}%", f"{float(np.min(pr))*100:.2f}%"]})
