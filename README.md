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
