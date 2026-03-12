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
