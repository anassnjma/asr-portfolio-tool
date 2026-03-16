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

The tool lets you analyse a portfolio from the terminal, check historical returns, get an overview of your holdings, see how your allocation is spread out, and run a 15-year simulation with 100k paths. On top of that I added risk metrics, VaR & CVaR using three methods, stress testing against crises like 2008 and COVID, and a Risk Parity optimizer. In LLM mode you just ask things normally and Gemini figures out what to run. In --no-llm mode you type commands like `simulate`, `var`, `stress`, `parity` — type `help` for the full list.

## Portfolio format

CSV with columns: `ticker`, `sector`, `asset_class`, `quantity`, `purchase_price`. The sample file has European stocks (ASML, Shell, Unilever, Adyen, Siemens, Sanofi) and a couple of ETFs.

## Known issues

- Simulation can take a minute with 100k paths
- Yahoo Finance sometimes rate limits if you restart the app too often, just wait a bit until it works again.
