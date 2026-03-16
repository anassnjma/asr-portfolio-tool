"""Routes user queries to the right model operations.

Two modes: Gemini function-calling (default) or direct keyword commands (--no-llm).
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from models.portfolio import Portfolio
from views import cli_view as view
import pandas as pd

# ---------------------------------------------------------------------------
# Gemini tool declarations
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "show_historical_returns",
        "description": (
            "Show historical returns of each asset in the portfolio. "
            "Supports monthly, quarterly, and yearly frequencies."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "frequency": {
                    "type": "string",
                    "enum": ["monthly", "quarterly", "yearly"],
                    "description": "The return frequency to display.",
                }
            },
            "required": ["frequency"],
        },
    },
    {
        "name": "show_portfolio_insights",
        "description": (
            "Display detailed portfolio insights: asset name, sector, "
            "asset class, quantity, purchase price, cost basis, current "
            "price, current market value and profit/loss."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "show_portfolio_weights",
        "description": (
            "Show total portfolio value and the relative weight of each "
            "asset. Can group by 'sector' or 'asset_class'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "string",
                    "enum": ["none", "sector", "asset_class"],
                    "description": "Grouping level. 'none' = per asset.",
                }
            },
            "required": ["group_by"],
        },
    },
    {
        "name": "run_simulation",
        "description": (
            "Run a Block Bootstrap simulation over 15 years with 100,000 "
            "paths. Unlike traditional Monte Carlo which assumes normally "
            "distributed returns, this resamples actual historical returns "
            "in blocks, preserving fat tails, volatility clustering, and "
            "cross-asset correlations."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "show_risk_metrics",
        "description": (
            "Display portfolio risk metrics: annualised return, volatility, "
            "Sharpe ratio, Sortino ratio, maximum drawdown, skewness, kurtosis."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "show_value_at_risk",
        "description": (
            "Calculate Value at Risk (VaR) and Expected Shortfall (CVaR) "
            "using historical, parametric, and Cornish-Fisher methods. "
            "These are the standard risk measures used by risk desks and "
            "required under Basel III/IV regulations."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "show_risk_parity",
        "description": (
            "Compute a Risk Parity allocation where each asset contributes "
            "equally to total portfolio risk. Compares the suggested weights "
            "and risk contributions against the current portfolio. This is "
            "the approach used by major asset managers and pension funds."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "compare_peers",
        "description": (
            "Compare a given ticker against its industry peers. Shows live "
            "market data including price, market cap, P/E ratio, and dividend "
            "yield for the target and up to 5 peers in the same industry."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The ticker symbol to compare (e.g. 'ASML').",
                }
            },
            "required": ["ticker"],
        },
    },
]


# ---------------------------------------------------------------------------
# Shared tool execution logic
# ---------------------------------------------------------------------------

FREQ_MAP = {"monthly": "ME", "quarterly": "QE", "yearly": "YE"}
FREQ_LABEL = {"monthly": "Monthly", "quarterly": "Quarterly", "yearly": "Yearly"}


def execute_tool(portfolio: Portfolio, name: str, args: dict) -> str:
    """Run the named tool and display output.

    Returns a text summary that can be fed back to the LLM.
    """
    dispatch = {
        "show_historical_returns": _tool_historical_returns,
        "show_portfolio_insights": _tool_insights,
        "show_portfolio_weights": _tool_weights,
        "run_simulation": _tool_simulation,
        "show_risk_metrics": _tool_risk,
        "show_value_at_risk": _tool_var,
        "show_risk_parity": _tool_risk_parity,
        "compare_peers": _tool_compare_peers,
    }
    handler = dispatch.get(name)
    if handler:
        return handler(portfolio, args)
    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Tool handlers (each displays output and returns a text summary for Gemini)
# ---------------------------------------------------------------------------

def _tool_historical_returns(portfolio: Portfolio, args: dict) -> str:
    freq_key = args.get("frequency", "monthly")
    freq = FREQ_MAP.get(freq_key, "ME")
    label = FREQ_LABEL.get(freq_key, "Monthly")

    view.print_loading(f"Calculating {label.lower()} returns…")
    df = portfolio.historical_returns(freq)
    display_df = df.tail(12)
    view.render_dataframe(display_df, title=f"{label} Returns (%) – Last 12 Periods")
    view.plot_returns_heatmap(display_df, label)
    return display_df.to_string()


def _tool_insights(portfolio: Portfolio, args: dict) -> str:
    view.print_loading("Fetching portfolio insights…")
    df = portfolio.insights_table()
    view.render_dataframe(df, title="Portfolio Insights")
    return df.to_string()


def _tool_weights(portfolio: Portfolio, args: dict) -> str:
    group_by_raw = args.get("group_by", "none")
    group_by = None if group_by_raw == "none" else group_by_raw
    label = f"by {group_by}" if group_by else "per asset"

    view.print_loading(f"Calculating weights ({label})…")
    df = portfolio.weights_table(group_by=group_by)
    view.render_dataframe(df, title=f"Portfolio Weights – {label}")
    view.plot_weights_pie(df, label)
    return df.to_string()


def _tool_simulation(portfolio: Portfolio, args: dict) -> str:
    view.print_loading("Running Block Bootstrap simulation (100k paths, 15 years)…")
    sim = portfolio.simulate()
    view.render_dataframe(sim["stats"], title="Block Bootstrap Simulation Results")
    view.plot_simulation(sim)
    return sim["stats"].to_string()


def _tool_risk(portfolio: Portfolio, args: dict) -> str:
    view.print_loading("Calculating risk metrics…")
    df = portfolio.risk_metrics()
    view.render_dataframe(df, title="Portfolio Risk Metrics (5yr history)")
    return df.to_string()


def _tool_var(portfolio: Portfolio, args: dict) -> str:
    view.print_loading("Calculating Value at Risk and Expected Shortfall…")
    result = portfolio.value_at_risk(confidence=0.95)
    view.render_dataframe(result["stats"], title="Value at Risk & Expected Shortfall (95%)")
    view.plot_var_histogram(result)
    return result["stats"].to_string()


def _tool_risk_parity(portfolio: Portfolio, args: dict) -> str:
    view.print_loading("Computing Risk Parity allocation…")
    result = portfolio.risk_parity()
    view.render_dataframe(result["summary"], title="Current vs Risk Parity")
    view.render_dataframe(result["weights"], title="Weight Allocation")
    view.render_dataframe(result["risk_contributions"], title="Risk Contribution per Asset")
    view.plot_risk_parity(result)
    return result["summary"].to_string() + "\n\n" + result["weights"].to_string()


def _tool_compare_peers(portfolio: Portfolio, args: dict) -> str:
    """Find industry peers for a ticker and show live comparison data."""
    import time as _time

    try:
        import financedatabase as fd
        import yfinance as yf
    except ImportError:
        view.print_error("Missing packages. Run: pip install financedatabase yfinance")
        return "Error: financedatabase package not installed."

    ticker = (args.get("ticker") or "").upper().strip()
    if not ticker:
        view.print_info("Usage: compare <ticker>  e.g. 'compare ASML'")
        return "No ticker provided."

    view.print_loading(f"Finding peers for {ticker} and fetching live data…")
    try:
        equities = fd.Equities()
        all_eq = equities.select()

        if ticker not in all_eq.index:
            matches = [t for t in all_eq.index if t.startswith(ticker.split(".")[0])]
            if matches:
                ticker = matches[0]
            else:
                view.print_error(f"{ticker} not found in database")
                return f"{ticker} not found in database."

        industry = all_eq.loc[ticker, "industry"]
        country = all_eq.loc[ticker, "country"]
        view.print_info(f"{ticker} ({all_eq.loc[ticker, 'name']}) — {industry}, {country}")

        peers = all_eq[all_eq["industry"] == industry]
        same_country = peers[peers["country"] == country].index.tolist()
        other = peers[peers["country"] != country].index.tolist()
        seen_names = {all_eq.loc[ticker, "name"]}
        peer_tickers = [ticker]
        for t in same_country + other:
            if len(peer_tickers) >= 6:
                break
            name = all_eq.loc[t, "name"]
            if t != ticker and name not in seen_names:
                seen_names.add(name)
                peer_tickers.append(t)

        view.print_loading("Fetching live market data from Yahoo Finance…")
        rows = []
        for t in peer_tickers:
            try:
                info = yf.Ticker(t).info
                dy = info.get("dividendYield")
                if dy and dy > 1:
                    dy = dy / 100
                rows.append({
                    "Ticker": t,
                    "Name": (info.get("shortName") or info.get("longName") or t)[:30],
                    "Price": info.get("currentPrice") or info.get("regularMarketPrice"),
                    "Market Cap (B)": round(info.get("marketCap", 0) / 1e9, 1) if info.get("marketCap") else None,
                    "P/E": round(info.get("trailingPE"), 2) if info.get("trailingPE") else None,
                    "Div Yield (%)": round(dy * 100, 2) if dy else None,
                    "Country": all_eq.loc[t, "country"] if t in all_eq.index else "N/A",
                })
            except Exception:
                rows.append({"Ticker": t, "Name": "Error fetching data"})
            _time.sleep(0.3)

        if rows:
            df = pd.DataFrame(rows)
            view.render_dataframe(df, title=f"Peer Comparison — {industry}")
            return df.to_string()
        else:
            view.print_error("Could not fetch data for any peers")
            return "No peer data available."
    except Exception as e:
        view.print_error(f"Compare failed: {e}")
        return f"Compare failed: {e}"


# ---------------------------------------------------------------------------
# LLM Controller (Gemini)
# ---------------------------------------------------------------------------

class GeminiController:
    """Gemini-based controller."""

    def __init__(self, portfolio: Portfolio, api_key: str) -> None:
        import google.generativeai as genai
        from models.config import GEMINI_MODEL

        self.portfolio = portfolio
        self.genai = genai

        genai.configure(api_key=api_key)

        # Build tools using the dict-based format (compatible with all versions)
        self.model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            tools=self._build_tools(),
            system_instruction=(
                "You are the Portfolio Assistant for a.s.r. Vermogensbeheer. "
                "Help users analyse their investment portfolio by calling the "
                "appropriate tool for their question. After receiving the tool "
                "output, provide a brief, insightful summary highlighting the "
                "key takeaways. Be professional, concise, and data-driven. "
                "If the question doesn't require a tool, answer directly. "
                "You have access to industry-standard tools including VaR, "
            ),
        )
        self.chat = self.model.start_chat(enable_automatic_function_calling=False)

    @staticmethod
    def _build_tools():
        """Convert our tool defs to Gemini format."""
        import google.generativeai as genai

        declarations = []
        for t in TOOL_DEFINITIONS:
            params = t.get("parameters", {})
            properties = params.get("properties", {})
            required = params.get("required", [])

            if not properties:
                # No parameters — define with empty schema
                declarations.append(
                    genai.types.FunctionDeclaration(
                        name=t["name"],
                        description=t["description"],
                        parameters=None,
                    )
                )
            else:
                # Build parameter schema using genai.types
                schema_props = {}
                for prop_name, prop_def in properties.items():
                    prop_type = prop_def.get("type", "string").upper()
                    schema_prop = {"type": prop_type}
                    if "description" in prop_def:
                        schema_prop["description"] = prop_def["description"]
                    if "enum" in prop_def:
                        schema_prop["enum"] = prop_def["enum"]
                    schema_props[prop_name] = schema_prop

                declarations.append(
                    genai.types.FunctionDeclaration(
                        name=t["name"],
                        description=t["description"],
                        parameters={
                            "type": "OBJECT",
                            "properties": schema_props,
                            "required": required,
                        },
                    )
                )

        return [genai.types.Tool(function_declarations=declarations)]

    def handle_query(self, user_input: str) -> None:
        """Send user question to Gemini and handle the response."""
        response = self._send_with_retry(user_input)

        for part in response.parts:
            if fn := part.function_call:
                tool_name = fn.name
                args = dict(fn.args) if fn.args else {}
                result_text = execute_tool(self.portfolio, tool_name, args)

                # Send tool result back to Gemini for a summary
                response = self._send_with_retry(
                    self.genai.protos.Content(
                        parts=[
                            self.genai.protos.Part(
                                function_response=self.genai.protos.FunctionResponse(
                                    name=tool_name,
                                    response={"result": result_text},
                                )
                            )
                        ]
                    )
                )

                text = self._extract_text(response)
                if text:
                    view.print_agent_response(text)
                return

        text = self._extract_text(response)
        if text:
            view.print_agent_response(text)

    def _send_with_retry(self, message, max_retries: int = 3):
        """Send with retry on 429 rate limits."""
        import time
        for attempt in range(max_retries):
            try:
                return self.chat.send_message(message)
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = (attempt + 1) * 10  # 10s, 20s, 30s
                    view.print_loading(f"Rate limited, retrying in {wait}s…")
                    time.sleep(wait)
                else:
                    raise

    @staticmethod
    def _extract_text(response) -> str:
        return "\n".join(part.text for part in response.parts if part.text)


# ---------------------------------------------------------------------------
# Direct Controller (no LLM)
# ---------------------------------------------------------------------------

class DirectController:
    """Keyword-based controller (no LLM)."""

    def __init__(self, portfolio: Portfolio) -> None:
        self.portfolio = portfolio

    def handle_query(self, user_input: str) -> None:
        tokens = user_input.lower().strip().split()
        if not tokens:
            return

        cmd = tokens[0]

        if cmd in ("returns", "return", "historical"):
            freq = "monthly"
            for t in tokens[1:]:
                if t.startswith("month"):
                    freq = "monthly"
                elif t.startswith("quarter"):
                    freq = "quarterly"
                elif t.startswith("year") or t.startswith("annual"):
                    freq = "yearly"
            execute_tool(self.portfolio, "show_historical_returns", {"frequency": freq})

        elif cmd in ("insights", "insight", "overview", "portfolio", "holdings"):
            execute_tool(self.portfolio, "show_portfolio_insights", {})

        elif cmd in ("weights", "weight", "allocation"):
            group_by = "none"
            for t in tokens[1:]:
                if t.startswith("sector"):
                    group_by = "sector"
                elif t.startswith("asset") or t.startswith("class"):
                    group_by = "asset_class"
            execute_tool(self.portfolio, "show_portfolio_weights", {"group_by": group_by})

        elif cmd in ("simulate", "simulation", "bootstrap", "sim"):
            execute_tool(self.portfolio, "run_simulation", {})

        elif cmd in ("risk", "metrics", "sharpe", "sortino", "drawdown"):
            execute_tool(self.portfolio, "show_risk_metrics", {})

        elif cmd in ("var", "cvar", "es", "expected_shortfall", "value_at_risk"):
            execute_tool(self.portfolio, "show_value_at_risk", {})

        elif cmd in ("parity", "riskparity", "rp", "equalrisk"):
            execute_tool(self.portfolio, "show_risk_parity", {})
        
        elif cmd in ("compare", "peers", "peer", "competitors"):
            ticker = " ".join(tokens[1:])
            execute_tool(self.portfolio, "compare_peers", {"ticker": ticker})

        elif cmd in ("help", "h", "?", "commands"):
            view.print_help()

        elif cmd in ("refresh", "reload"):
            view.print_loading("Refreshing market data…")
            from models.market_data import clear_cache
            clear_cache()
            self.portfolio.refresh_market_data()
            view.print_info(f"Portfolio value: ${self.portfolio.total_value():,.2f}")

        else:
            view.print_error(
                f"Unknown command: '{cmd}'. Type [bold]help[/bold] for available commands."
            )
