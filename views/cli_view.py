"""CLI output using Rich for tables and Matplotlib for charts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from models.config import OUTPUT_DIR

# Rich console with custom theme
theme = Theme({
    "info": "bold green",
    "warn": "bold yellow",
    "err": "bold red",
    "accent": "bold cyan",
})
console = Console(theme=theme)


# ---------------------------------------------------------------------------
# Banners & messages
# ---------------------------------------------------------------------------

LOGO = r"""
  ╔═══════════════════════════════════════════╗
  ║       Portfolio Assistant                 ║
  ╚═══════════════════════════════════════════╝
"""


def print_welcome() -> None:
    console.print(LOGO, style="accent")
    console.print(
        "  Ask any question about your portfolio, or type a command:\n"
        "  [accent]returns[/accent]  ·  [accent]insights[/accent]  ·  "
        "[accent]weights[/accent]  ·  [accent]simulate[/accent]  ·  "
        "[accent]risk[/accent]  ·  [accent]var[/accent]  ·  "
        "[accent]parity[/accent]  ·  [accent]compare[/accent]  ·  "
        "[accent]help[/accent]  ·  [accent]quit[/accent]\n"
    )


def print_help() -> None:
    console.print(Panel(
        "[accent]returns monthly|quarterly|yearly[/accent]  — Historical returns per asset\n"
        "[accent]insights[/accent]                         — Full portfolio overview\n"
        "[accent]weights[/accent]                          — Value & weight per asset\n"
        "[accent]weights sector|asset_class[/accent]       — Weights grouped by category\n"
        "[accent]simulate[/accent]                         — Block Bootstrap simulation (15yr)\n"
        "[accent]risk[/accent]                             — Risk metrics (Sharpe, Sortino, drawdown)\n"
        "[accent]var[/accent]                              — Historical Value at Risk & Expected Shortfall\n"
        "[accent]parity[/accent]                           — Risk Parity optimal allocation\n"
        "[accent]compare <ticker>[/accent]                 — Peer comparison (e.g. compare ASML)\n"
        "[accent]refresh[/accent]                          — Re-fetch market data\n"
        "[accent]quit[/accent]                             — Exit the application",
        title="Available Commands",
        border_style="cyan",
    ))


def print_loading(msg: str) -> None:
    console.print(f"  [dim]⏳ {msg}[/dim]")


def print_error(msg: str) -> None:
    console.print(f"  [err]✗ {msg}[/err]")


def print_info(msg: str) -> None:
    console.print(f"  [info]✓[/info] {msg}")


def print_agent_response(text: str) -> None:
    """Display the LLM's natural-language answer."""
    console.print()
    console.print(Panel(text, border_style="blue", title="💬 Assistant", padding=(1, 2)))


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def render_dataframe(df: pd.DataFrame, title: str = "") -> None:
    """Render a pandas DataFrame as a formatted Rich table."""
    table = Table(
        title=title, show_lines=True,
        header_style="bold cyan", border_style="dim", pad_edge=True,
    )
    for col in df.columns:
        justify = "right" if _is_numeric_col(df[col]) else "left"
        table.add_column(str(col), justify=justify)

    for _, row in df.iterrows():
        table.add_row(*[_format_cell(row[col], col) for col in df.columns])

    console.print()
    console.print(table)
    console.print()


def _is_numeric_col(series: pd.Series) -> bool:
    return series.dtype.kind in ("f", "i")


def _format_cell(val, col_name: str = "") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "[dim]–[/dim]"
    if isinstance(val, float):
        if "P&L" in str(col_name) or "Return" in str(col_name):
            color = "green" if val >= 0 else "red"
            prefix = "+" if val > 0 else ""
            if "%" in str(col_name):
                return f"[{color}]{prefix}{val:.2f}%[/{color}]"
            return f"[{color}]{prefix}{val:,.2f}[/{color}]"
        return f"{val:,.2f}"
    return str(val)


# ---------------------------------------------------------------------------
# Matplotlib setup
# ---------------------------------------------------------------------------

def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    return plt


# ---------------------------------------------------------------------------
# Chart: Simulation fan chart
# ---------------------------------------------------------------------------

def plot_simulation(sim_result: dict) -> str:
    plt = _get_plt()
    paths = sim_result["paths"]
    n_days = paths.shape[1]
    years = n_days // 252
    x = np.arange(n_days)

    fig, ax = plt.subplots(figsize=(14, 6))
    for plo, phi, a in [(5, 95, 0.12), (25, 75, 0.25)]:
        ax.fill_between(
            x, np.percentile(paths, plo, axis=0), np.percentile(paths, phi, axis=0),
            alpha=a, color="#2563eb", label=f"{plo}th–{phi}th pctl",
        )
    ax.plot(x, np.percentile(paths, 50, axis=0), color="#2563eb", lw=2.5, label="Median")
    ax.axhline(sim_result["initial_value"], color="#dc2626", ls="--", lw=1.2, label="Starting value")

    ax.set_xticks([i * 252 for i in range(years + 1)])
    ax.set_xticklabels([f"Year {i}" for i in range(years + 1)], fontsize=9)
    ax.set_title(f"Block Bootstrap Simulation · {years} Years · {paths.shape[0]:,} Paths", fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = str(OUTPUT_DIR / "simulation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print_info(f"Chart saved → [accent]{path}[/accent]")
    return path


# ---------------------------------------------------------------------------
# Chart: Returns heatmap
# ---------------------------------------------------------------------------

def plot_returns_heatmap(returns_df: pd.DataFrame, freq_label: str) -> str:
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(
        max(10, len(returns_df.columns) * 1.4),
        max(4, len(returns_df) * 0.4),
    ))
    im = ax.imshow(returns_df.values, aspect="auto", cmap="RdYlGn", interpolation="nearest")
    ax.set_xticks(range(len(returns_df.columns)))
    ax.set_xticklabels(returns_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(returns_df.index)))
    ax.set_yticklabels(returns_df.index, fontsize=9)
    for i in range(len(returns_df.index)):
        for j in range(len(returns_df.columns)):
            v = returns_df.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"{freq_label} Returns (%)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Return (%)", shrink=0.8)
    fig.tight_layout()

    path = str(OUTPUT_DIR / "returns_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print_info(f"Chart saved → [accent]{path}[/accent]")
    return path


# ---------------------------------------------------------------------------
# Chart: Weights pie
# ---------------------------------------------------------------------------

def plot_weights_pie(weights_df: pd.DataFrame, group_label: str) -> str:
    plt = _get_plt()
    df = weights_df[weights_df.iloc[:, 0] != "TOTAL"].copy()
    labels = df.iloc[:, 0].values
    sizes = df["Weight (%)"].values

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, pctdistance=0.80, startangle=90)
    ax.set_title(f"Portfolio Weights ({group_label})", fontsize=13, fontweight="bold")
    fig.tight_layout()

    filename = f"weights_{group_label.replace(' ', '_')}.png"
    path = str(OUTPUT_DIR / filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print_info(f"Chart saved → [accent]{path}[/accent]")
    return path


# ---------------------------------------------------------------------------
# Chart: Risk Parity comparison
# ---------------------------------------------------------------------------

def plot_risk_parity(rp_result: dict) -> str:
    """Side-by-side bar charts comparing current vs risk parity allocation."""
    plt = _get_plt()

    tickers = rp_result["tickers"]
    w_cur = rp_result["current_weights"] * 100
    w_rp = rp_result["rp_weights"] * 100
    n = len(tickers)
    x = np.arange(n)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Weight comparison
    ax1.bar(x - width / 2, w_cur, width, color="#94a3b8", label="Current")
    ax1.bar(x + width / 2, w_rp, width, color="#2563eb", label="Risk Parity")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Weight (%)")
    ax1.set_title("Weight Allocation", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # Panel 2: Risk contribution comparison
    rc_df = rp_result["risk_contributions"]
    rc_cur = rc_df["Current RC (%)"].values
    rc_rp = rc_df["Risk Parity RC (%)"].values
    target = 100 / n

    ax2.bar(x - width / 2, rc_cur, width, color="#94a3b8", label="Current")
    ax2.bar(x + width / 2, rc_rp, width, color="#2563eb", label="Risk Parity")
    ax2.axhline(target, color="#dc2626", ls="--", lw=1.2, label=f"Equal target ({target:.1f}%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Risk Contribution (%)")
    ax2.set_title("Risk Contribution per Asset", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Risk Parity Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = str(OUTPUT_DIR / "risk_parity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print_info(f"Chart saved → [accent]{path}[/accent]")
    return path