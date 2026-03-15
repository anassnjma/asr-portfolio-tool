#!/usr/bin/env python3
"""CLI entry point for the portfolio assistant."""

from __future__ import annotations

import argparse
import sys
import warnings

# Suppress deprecation warning from google-generativeai package
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*generativeai.*")

from models.portfolio import Portfolio
from controllers.agent_controller import DirectController, GeminiController
from views.cli_view import console, print_error, print_info, print_loading, print_welcome
from models.config import get_gemini_api_key, GEMINI_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="a.s.r. Vermogensbeheer – Portfolio Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                            Use sample portfolio with Gemini\n"
            "  python main.py --portfolio my.csv         Use custom portfolio file\n"
            "  python main.py --no-llm                   Run without API key (direct commands)\n"
        ),
    )
    parser.add_argument(
        "--portfolio", "-p",
        type=str,
        default="sample_portfolio.csv",
        help="Path to portfolio file: CSV, JSON, or Excel (default: sample_portfolio.csv)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run in direct-command mode without a Gemini API key",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_welcome()

    # ── Load portfolio ──────────────────────────────────────────────
    try:
        portfolio = Portfolio.from_file(args.portfolio)
        print_info(f"Loaded [bold]{len(portfolio)}[/bold] assets from [accent]{args.portfolio}[/accent]")
    except (FileNotFoundError, ValueError) as e:
        print_error(str(e))
        sys.exit(1)

    # ── Fetch market data ───────────────────────────────────────────
    print_loading("Connecting to Yahoo Finance")
    try:
        portfolio.refresh_market_data()
        total = portfolio.total_value()
        cost = portfolio.total_cost()
        if total > 0:
            pnl = total - cost
            sign = "+" if pnl >= 0 else ""
            print_info(f"Portfolio value: [bold]${total:,.2f}[/bold]  (cost basis: ${cost:,.2f}, P&L: {sign}${pnl:,.2f})")
        else:
            print_error("Could not fetch live prices due to too many requists. Please try again later.")
    except Exception as e:
        print_error(f"Market data unavailable: {e}")

    # ── Initialise controller ───────────────────────────────────────
    if args.no_llm:
        print_info("Running in [bold]direct-command mode[/bold] (no LLM). Type [accent]help[/accent] for commands.\n")
        controller = DirectController(portfolio)
    else:
        try:
            api_key = get_gemini_api_key()
            controller = GeminiController(portfolio, api_key)
            print_info(f"Connected to [bold]{GEMINI_MODEL}[/bold]. Ask anything in natural language!\n")
        except EnvironmentError as e:
            print_error(str(e))
            print_info("Falling back to [bold]direct-command mode[/bold].\n")
            controller = DirectController(portfolio)

    # ── Conversation loop ───────────────────────────────────────────
    while True:
        try:
            user_input = console.input("[bold cyan]You →[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [dim]Goodbye![/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            console.print("  [dim]Goodbye![/dim]")
            break

        try:
            controller.handle_query(user_input)
        except Exception as e:
            print_error(f"Something went wrong: {e}")

        console.print()


if __name__ == "__main__":
    main()
