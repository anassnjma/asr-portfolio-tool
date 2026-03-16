# Portfolio Assistant

Tool for analysing investment portfolios. Uses Gemini as the LLM to interpret questions and runs the appropriate analysis.

## Setup

You need Python 3.10+ and a Gemini API key (this tool is set for gemini 2.5 fast (in case your api doesnt have the 2.5 fast model you can change this to something else in the config file).
```bash
git clone https://github.com/anassnjma/asr-portfolio-tool.git
cd asr-portfolio-tool
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```
paste your Gemini API key in .env <br />
Then run
```bash
python main.py
```

Or without the LLM (no API key needed):
```bash
python main.py --no-llm
```

You can also pass your own portfolio: `python main.py --portfolio my_file.csv`

## What this tool does

The tool lets you analyse a portfolio from the terminal, check historical returns, get an overview of your holdings, see how your allocation is spread out, and run a 15-year simulation with 100k paths. On top of that I added risk metrics, VaR & CVaR using three methods, stress testing against crises like 2008 and COVID, and a Risk Parity optimizer. In LLM mode you just ask things normally and Gemini figures out what to run. In --no-llm mode you type commands like `simulate`, `var`, `stress`, `parity`. Type `help` in the terminal for the full list of options.

## Functions

**Historical Returns** — Shows the percentage return of each asset over a chosen period. 
Supports monthly, quarterly, and yearly frequency. Displays the last 12 periods in a table.

**Portfolio Insights** — Full overview of every holding: asset name, sector, asset class, 
quantity, purchase price, cost basis, current market price, current market value, and P&L 
in both dollars and percentage.

**Portfolio Weights** — Shows total portfolio value and the relative weight of each position. 
Can group weights by sector or asset class.

**Simulation** — Block Bootstrap simulation over 15 years with 100,000 paths. Instead of 
assuming normally distributed returns like a Monte Carlo simulation would, it resamples historical 
returns in blocks of 21 trading days (one trading month), preserving fat tails and volatility clustering.

**Risk Metrics** — Annualised return and volatility, Sharpe ratio, Sortino ratio, maximum 
drawdown, skewness, and excess kurtosis based on 5 years of historical data.

**Value at Risk** — Daily VaR and CVaR (Expected Shortfall) at 95% confidence 
using the historical simulation method. Plots a return distribution histogram with the 
VaR and CVaR thresholds marked.

**Risk Parity** — Computes an optimal allocation where each asset contributes equally to 
total portfolio risk using scipy's SLSQP optimiser. Compares the suggested weights and 
risk contributions side by side against your current allocation.

**Peer Comparison** — Given any ticker, finds industry peers using the FinanceDatabase package and 
fetches live data from Yahoo Finance. All prices and market caps are converted to EUR using 
live exchange rates.

## Available commands (--no-llm mode)

| Command | Description |
|---|---|
| `returns monthly\|quarterly\|yearly` | Historical returns per asset |
| `insights` | Full portfolio overview |
| `weights` | Value & weight per asset |
| `weights sector\|asset_class` | Weights grouped by category |
| `simulate` | Block Bootstrap simulation of 15 years |
| `risk` | Risk metrics (Sharpe, Sortino, drawdown) |
| `var` | Historical Value at Risk & Expected Shortfall |
| `parity` | Risk Parity optimal allocation |
| `compare <ticker>` | Peer comparison |
| `refresh` | Re-fetch live market data |
| `help` | Show all commands |
| `quit` | Exit |

## Portfolio format

CSV with columns: `ticker`, `sector`, `asset_class`, `quantity`, `purchase_price`. The sample file has European stocks (ASML, Shell, Unilever, Adyen, Siemens, Sanofi) and a couple of ETFs.

## Known issues

- Simulation can take a minute with 100k paths
- Yahoo Finance sometimes rate limits if you restart the app too often, just wait a bit until it works again.
