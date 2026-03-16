"""
Production-ready automated stock trading agent.
Strategy: Multi-Indicator Momentum with Groq AI confirmation.
Broker: Alpaca Paper Trading (alpaca-py).
Logging & learning: Supabase.
Notifications: Telegram.
"""

import os
import json
import datetime
import pytz
import requests
import numpy as np
import pandas as pd
import ta
import yfinance as yf

from groq import Groq
from supabase import create_client, Client as SupabaseClient
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

# ── Configuration ────────────────────────────────────────────────────────────

WATCHLIST = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "META", "AMD", "GOOGL", "NFLX", "SPY"]

MAX_POSITIONS      = 3       # max simultaneous open positions
POSITION_PCT       = 0.10    # 10% of portfolio per trade
ATR_SL_MULTIPLIER  = 1.5     # stop-loss = entry - ATR * 1.5
ATR_TP_MULTIPLIER  = 3.0     # take-profit = entry + ATR * 3.0
MIN_SCORE          = 4       # minimum technical score to consider a buy
GROQ_MIN_CONF      = 65      # minimum Groq confidence to execute
FALLBACK_MIN_SCORE = 5       # score required when Groq is unavailable
EARNINGS_LOOKOUT   = 7       # skip if earnings within N days
DATA_DAYS          = 150     # days of historical data to download
MIN_ROWS           = 50      # skip stock if fewer rows returned
HISTORY_LIMIT      = 30      # trades to load for Groq learning

# ── Client factories ──────────────────────────────────────────────────────────

def get_trading_client() -> TradingClient:
    return TradingClient(
        api_key    = os.environ["ALPACA_API_KEY"],
        secret_key = os.environ["ALPACA_SECRET_KEY"],
        paper      = True,
    )


def get_groq_client() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def get_supabase_client() -> SupabaseClient:
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_KEY"],
    )


# ── Market-hours check ────────────────────────────────────────────────────────

def market_is_open(client: TradingClient) -> bool:
    return client.get_clock().is_open


# ── Telegram notifications ────────────────────────────────────────────────────

def send_telegram(message: str) -> None:
    """Send a message via Telegram Bot API. Silently skips if env vars missing."""
    token   = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message},
            timeout=10,
        )
    except Exception as exc:
        print(f"  [telegram error] {exc}")


# ── Supabase trade logging ─────────────────────────────────────────────────────

def save_buy_trade(
    sb: SupabaseClient,
    symbol: str,
    price: float,
    shares: int,
    score: int,
    rsi: float,
    adx: float,
    groq_decision: str,
    groq_conf: int,
    reason: str,
    portfolio: float,
) -> None:
    """Insert a new BUY row into the trades table."""
    try:
        sb.table("trades").insert({
            "symbol":        symbol,
            "action":        "BUY",
            "price":         round(price, 4),
            "shares":        shares,
            "score":         score,
            "rsi":           round(rsi, 2),
            "adx":           round(adx, 2),
            "groq_decision": groq_decision,
            "groq_conf":     groq_conf,
            "reason":        reason,
            "result":        "OPEN",
            "portfolio":     round(portfolio, 2),
        }).execute()
    except Exception as exc:
        print(f"  [supabase buy log error] {exc}")


def update_sell_trade(
    sb: SupabaseClient,
    symbol: str,
    exit_price: float,
) -> None:
    """Find the most recent OPEN trade for symbol and update it with sell info."""
    try:
        # Fetch the most recent OPEN trade for this symbol
        resp = (
            sb.table("trades")
            .select("id, price, shares")
            .eq("symbol", symbol)
            .eq("result", "OPEN")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return
        row        = resp.data[0]
        entry_price = float(row["price"])
        shares      = int(row["shares"])
        pnl         = round((exit_price - entry_price) * shares, 2)
        result      = "WIN" if pnl > 0 else "LOSS"
        sb.table("trades").update({
            "action": "SELL",
            "pnl":    pnl,
            "result": result,
        }).eq("id", row["id"]).execute()
    except Exception as exc:
        print(f"  [supabase sell log error] {exc}")


# ── Trade history & performance stats ────────────────────────────────────────

def load_trade_history(sb: SupabaseClient) -> list[dict]:
    """Fetch the last HISTORY_LIMIT closed trades (WIN or LOSS)."""
    try:
        resp = (
            sb.table("trades")
            .select("symbol, score, result, pnl")
            .in_("result", ["WIN", "LOSS"])
            .order("created_at", desc=True)
            .limit(HISTORY_LIMIT)
            .execute()
        )
        return resp.data or []
    except Exception as exc:
        print(f"  [supabase history error] {exc}")
        return []


def calc_win_rate(trades: list[dict]) -> float:
    """Return win rate 0-100 for a list of trades. Returns 0 if empty."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t["result"] == "WIN")
    return round(wins / len(trades) * 100, 1)


def build_performance_stats(trades: list[dict]) -> dict:
    """
    Returns a dict with:
      - overall_win_rate
      - per_symbol: {symbol: {"win_rate": X, "n": N}}
      - per_score:  {4: X%, 5: X%, 6: X%}
      - best_symbol, worst_symbol  (min 2 trades to qualify)
    """
    if not trades:
        return {
            "overall_win_rate": 0.0,
            "per_symbol": {},
            "per_score":  {},
            "best_symbol":  None,
            "worst_symbol": None,
        }

    overall_win_rate = calc_win_rate(trades)

    # Per-symbol
    symbols: dict[str, list] = {}
    for t in trades:
        symbols.setdefault(t["symbol"], []).append(t)
    per_symbol = {
        sym: {"win_rate": calc_win_rate(ts), "n": len(ts)}
        for sym, ts in symbols.items()
    }

    # Per-score (only scores 4/5/6)
    per_score: dict[int, float] = {}
    for lvl in (4, 5, 6):
        lvl_trades = [t for t in trades if t.get("score") == lvl]
        if lvl_trades:
            per_score[lvl] = calc_win_rate(lvl_trades)

    # Best / worst (require >= 2 trades)
    qualified = {s: v for s, v in per_symbol.items() if v["n"] >= 2}
    best_symbol  = max(qualified, key=lambda s: qualified[s]["win_rate"]) if qualified else None
    worst_symbol = min(qualified, key=lambda s: qualified[s]["win_rate"]) if qualified else None

    return dict(
        overall_win_rate = overall_win_rate,
        per_symbol       = per_symbol,
        per_score        = per_score,
        best_symbol      = best_symbol,
        worst_symbol     = worst_symbol,
    )


def build_history_prompt_block(symbol: str, stats: dict) -> str:
    """Build the history section injected into every Groq prompt."""
    if not stats["per_symbol"]:
        return ""

    sym_info = stats["per_symbol"].get(symbol)
    sym_line = (
        f"- {symbol} win rate: {sym_info['win_rate']}% ({sym_info['n']} trades)"
        if sym_info else f"- {symbol}: no history yet"
    )

    score_lines = "\n".join(
        f"- Score {lvl}/6 win rate: {wr}%"
        for lvl, wr in sorted(stats["per_score"].items())
    ) or "- Score breakdown: insufficient data"

    return (
        f"\nMy recent trading history (last {HISTORY_LIMIT} trades):\n"
        f"- Overall win rate: {stats['overall_win_rate']}%\n"
        f"{sym_line}\n"
        f"{score_lines}\n"
        f"Use this data to adjust your confidence up or down.\n"
    )


# ── Data & indicator calculation ──────────────────────────────────────────────

def fetch_data(symbol: str) -> pd.DataFrame | None:
    """Download 150 days of daily OHLCV data via yfinance."""
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=DATA_DAYS)
    df    = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or len(df) < MIN_ROWS:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df


def calculate_indicators(df: pd.DataFrame) -> dict | None:
    """Return a dict of the latest indicator values, or None on failure."""
    try:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        # RSI(14)
        rsi_series = ta.momentum.RSIIndicator(close=close, window=14).rsi()

        # MACD(12,26,9)
        macd_obj    = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        macd_line   = macd_obj.macd()
        signal_line = macd_obj.macd_signal()
        hist_line   = macd_obj.macd_diff()

        # EMAs
        ema9   = ta.trend.EMAIndicator(close=close, window=9).ema_indicator()
        ema21  = ta.trend.EMAIndicator(close=close, window=21).ema_indicator()
        ema50  = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
        ema200 = ta.trend.EMAIndicator(close=close, window=200).ema_indicator()

        # ATR(14)
        atr_series = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()

        # ADX(14)
        adx_series = ta.trend.ADXIndicator(
            high=high, low=low, close=close, window=14
        ).adx()

        # Volume 20-day average
        vol_ma20 = volume.rolling(20).mean()

        idx       = -1
        rsi       = float(rsi_series.iloc[idx])
        macd_val  = float(macd_line.iloc[idx])
        signal    = float(signal_line.iloc[idx])
        hist      = float(hist_line.iloc[idx])
        hist_prev = float(hist_line.iloc[-2]) if len(hist_line) >= 2 else hist

        e9   = float(ema9.iloc[idx])
        e21  = float(ema21.iloc[idx])
        e50  = float(ema50.iloc[idx])
        e200 = float(ema200.iloc[idx])
        atr  = float(atr_series.iloc[idx])
        adx  = float(adx_series.iloc[idx])

        price     = float(close.iloc[idx])
        vol_today = float(volume.iloc[idx])
        vol_avg20 = float(vol_ma20.iloc[idx])
        vol_ratio = vol_today / vol_avg20 if vol_avg20 > 0 else 0.0

        return dict(
            price=price,
            rsi=rsi,
            macd=macd_val,
            signal=signal,
            hist=hist,
            hist_prev=hist_prev,
            ema9=e9, ema21=e21, ema50=e50, ema200=e200,
            atr=atr,
            adx=adx,
            vol_ratio=vol_ratio,
        )
    except Exception as exc:
        print(f"  [indicator error] {exc}")
        return None


# ── Scoring & filtering ───────────────────────────────────────────────────────

def compute_score(ind: dict) -> int:
    """Return technical score 0-6."""
    score = 0
    if 45 <= ind["rsi"] <= 70:
        score += 1
    if ind["macd"] > ind["signal"]:
        score += 1
    if ind["hist"] > 0 and ind["hist"] > ind["hist_prev"]:
        score += 1
    if ind["price"] > ind["ema9"] and ind["price"] > ind["ema21"]:
        score += 1
    if ind["price"] > ind["ema50"]:
        score += 1
    if ind["vol_ratio"] >= 1.2:
        score += 1
    return score


def passes_filters(ind: dict, score: int) -> bool:
    """All mandatory filters must pass."""
    if ind["price"] <= ind["ema200"]:
        return False
    if ind["adx"] <= 20:
        return False
    if score < MIN_SCORE:
        return False
    return True


# ── Groq AI confirmation ──────────────────────────────────────────────────────

def groq_confirm(
    groq_client: Groq,
    symbol: str,
    ind: dict,
    score: int,
    stats: dict,
) -> tuple[str, int, str]:
    """
    Ask Groq whether to BUY or SKIP, injecting trade history stats.
    Returns (decision, confidence, reason).
    Falls back to ("FALLBACK", 0, "groq_error") on failure.
    """
    stop_loss   = ind["price"] - ind["atr"] * ATR_SL_MULTIPLIER
    take_profit = ind["price"] + ind["atr"] * ATR_TP_MULTIPLIER
    sl_pct      = (ind["price"] - stop_loss) / ind["price"] * 100
    tp_pct      = (take_profit - ind["price"]) / ind["price"] * 100
    trend_label = "Above" if ind["price"] > ind["ema200"] else "Below"
    history_block = build_history_prompt_block(symbol, stats)

    prompt = (
        f"You are a professional swing trading AI.\n"
        f"Analyze these indicators for {symbol} and decide BUY or SKIP.\n\n"
        f"Current Price: ${ind['price']:.2f}\n"
        f"RSI(14): {ind['rsi']:.1f} (45-70 = momentum zone)\n"
        f"MACD: {ind['macd']:.3f} | Signal: {ind['signal']:.3f} | Hist: {ind['hist']:.3f}\n"
        f"EMA9: ${ind['ema9']:.2f} | EMA21: ${ind['ema21']:.2f} | EMA50: ${ind['ema50']:.2f} | EMA200: ${ind['ema200']:.2f}\n"
        f"Price vs EMA200: {trend_label} (major trend)\n"
        f"ADX(14): {ind['adx']:.1f} (>20 = trending market)\n"
        f"ATR(14): ${ind['atr']:.2f}\n"
        f"Proposed Stop-Loss: ${stop_loss:.2f} ({sl_pct:.1f}% risk)\n"
        f"Proposed Take-Profit: ${take_profit:.2f} ({tp_pct:.1f}% reward)\n"
        f"Volume vs 20d avg: {ind['vol_ratio']:.1f}x\n"
        f"Technical Score: {score}/6\n"
        f"{history_block}\n"
        'Reply ONLY with valid JSON, no extra text:\n'
        '{"decision": "BUY or SKIP", "confidence": 0-100, "reason": "max 12 words"}'
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=120,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        data       = json.loads(raw)
        decision   = str(data.get("decision", "SKIP")).upper()
        confidence = int(data.get("confidence", 0))
        reason     = str(data.get("reason", ""))
        return decision, confidence, reason
    except Exception as exc:
        print(f"  [groq error] {exc}")
        return "FALLBACK", 0, "groq_error"


# ── Earnings proximity check ──────────────────────────────────────────────────

def earnings_soon(symbol: str) -> bool:
    """Return True if earnings are within EARNINGS_LOOKOUT days."""
    try:
        ticker = yf.Ticker(symbol)
        cal    = ticker.calendar
        if cal is None or cal.empty:
            return False
        if "Earnings Date" in cal.columns:
            dates = pd.to_datetime(cal["Earnings Date"], errors="coerce").dropna()
        elif "Earnings Date" in cal.index:
            dates = pd.to_datetime([cal.loc["Earnings Date"]], errors="coerce")
        else:
            return False
        today = pd.Timestamp.today().normalize()
        for d in dates:
            if pd.NaT == d:
                continue
            if 0 <= (d.normalize() - today).days <= EARNINGS_LOOKOUT:
                return True
        return False
    except Exception:
        return False


# ── Alpaca order helpers ──────────────────────────────────────────────────────

def get_existing_positions(client: TradingClient) -> dict[str, object]:
    try:
        return {p.symbol: p for p in client.get_all_positions()}
    except Exception:
        return {}


def place_bracket_order(
    client: TradingClient,
    symbol: str,
    qty: int,
    stop_loss: float,
    take_profit: float,
) -> bool:
    try:
        client.submit_order(MarketOrderRequest(
            symbol        = symbol,
            qty           = qty,
            side          = OrderSide.BUY,
            time_in_force = TimeInForce.DAY,
            order_class   = OrderClass.BRACKET,
            stop_loss     = StopLossRequest(stop_price=round(stop_loss, 2)),
            take_profit   = TakeProfitRequest(limit_price=round(take_profit, 2)),
        ))
        return True
    except Exception as exc:
        print(f"  [order error] {exc}")
        return False


def close_position(client: TradingClient, symbol: str) -> bool:
    try:
        client.close_position(symbol)
        return True
    except Exception as exc:
        print(f"  [close error] {exc}")
        return False


# ── Sell logic ────────────────────────────────────────────────────────────────

def should_sell(ind: dict, entry_price: float) -> tuple[bool, str]:
    stop_loss   = entry_price - ind["atr"] * ATR_SL_MULTIPLIER
    take_profit = entry_price + ind["atr"] * ATR_TP_MULTIPLIER

    if ind["rsi"] > 75:
        return True, f"RSI overbought ({ind['rsi']:.1f})"
    if ind["rsi"] < 40:
        return True, f"RSI reversal ({ind['rsi']:.1f})"
    if ind["macd"] < ind["signal"]:
        return True, "MACD bearish cross"
    if ind["price"] < ind["ema21"]:
        return True, f"Price < EMA21 (${ind['ema21']:.2f})"
    if ind["price"] <= stop_loss:
        return True, f"ATR stop-loss hit (${stop_loss:.2f})"
    if ind["price"] >= take_profit:
        return True, f"ATR take-profit hit (${take_profit:.2f})"
    return False, ""


# ── Weekly summary (Fridays 15:30 EST) ───────────────────────────────────────

def is_weekly_summary_time() -> bool:
    """True on Fridays at the 15:30 EST run (20:30 UTC)."""
    now = datetime.datetime.utcnow()
    return now.weekday() == 4 and now.hour == 20


def print_weekly_summary(groq_client: Groq, sb: SupabaseClient) -> None:
    """Fetch all closed trades from the past 7 days and print a summary."""
    try:
        since = (datetime.datetime.utcnow() - datetime.timedelta(days=7)).isoformat()
        resp  = (
            sb.table("trades")
            .select("symbol, score, result, pnl")
            .in_("result", ["WIN", "LOSS"])
            .gte("created_at", since)
            .execute()
        )
        trades = resp.data or []
    except Exception as exc:
        print(f"[weekly summary] Supabase error: {exc}")
        return

    if not trades:
        print("\n=== WEEKLY REPORT ===\nNo closed trades this week.\n====================")
        send_telegram("📈 WEEKLY REPORT\nNo closed trades this week.")
        return

    stats        = build_performance_stats(trades)
    total        = len(trades)
    wins         = sum(1 for t in trades if t["result"] == "WIN")
    losses       = total - wins
    win_rate     = stats["overall_win_rate"]
    best_sym     = stats["best_symbol"]
    worst_sym    = stats["worst_symbol"]
    per_sym      = stats["per_symbol"]

    best_line  = (
        f"{best_sym} ({per_sym[best_sym]['win_rate']}% win rate)"
        if best_sym else "N/A"
    )
    worst_line = (
        f"{worst_sym} ({per_sym[worst_sym]['win_rate']}% win rate)"
        if worst_sym else "N/A"
    )

    score_lines = "\n".join(
        f"  Score {lvl}/6 win rate: {wr}%"
        for lvl, wr in sorted(stats["per_score"].items())
    ) or "  Insufficient data"

    best_score_lvl = (
        max(stats["per_score"], key=lambda k: stats["per_score"][k])
        if stats["per_score"] else "N/A"
    )
    best_score_wr = stats["per_score"].get(best_score_lvl, 0) if isinstance(best_score_lvl, int) else 0

    # Ask Groq for a 2-sentence recommendation
    recommendation = _groq_weekly_recommendation(groq_client, stats, trades)

    print(
        f"\n{'='*20} WEEKLY REPORT {'='*20}\n"
        f"Trades: {total} | Wins: {wins} | Losses: {losses} | Win Rate: {win_rate}%\n"
        f"Best symbol:  {best_line}\n"
        f"Worst symbol: {worst_line}\n"
        f"Score breakdown:\n{score_lines}\n"
        f"Best score level: {best_score_lvl}/6 ({best_score_wr}% win rate)\n"
        f"Recommendation: {recommendation}\n"
        f"{'='*55}"
    )
    send_telegram(
        f"📈 WEEKLY REPORT\n"
        f"Trades: {total} | Win Rate: {win_rate}%\n"
        f"Best: {best_sym or 'N/A'} | Worst: {worst_sym or 'N/A'}\n"
        f"{recommendation}"
    )


def _groq_weekly_recommendation(groq_client: Groq, stats: dict, trades: list[dict]) -> str:
    """Ask Groq for a 2-sentence strategy recommendation based on the week."""
    per_sym   = stats["per_symbol"]
    sym_lines = "\n".join(
        f"  {s}: {v['win_rate']}% ({v['n']} trades)"
        for s, v in sorted(per_sym.items(), key=lambda x: -x[1]["win_rate"])
    ) or "  No data"
    score_lines = "\n".join(
        f"  Score {lvl}/6: {wr}%"
        for lvl, wr in sorted(stats["per_score"].items())
    ) or "  No data"

    prompt = (
        f"You are a professional trading analyst.\n"
        f"Here is last week's performance summary:\n\n"
        f"Overall win rate: {stats['overall_win_rate']}%\n"
        f"Symbol performance:\n{sym_lines}\n"
        f"Score level performance:\n{score_lines}\n\n"
        f"Write exactly 2 sentences recommending how to adjust the strategy next week. "
        f"Be specific and concise."
    )
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=120,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"(Groq unavailable: {exc})"


# ── Main run loop ─────────────────────────────────────────────────────────────

def run():
    trading_client = get_trading_client()
    groq_client    = get_groq_client()
    sb             = get_supabase_client()

    send_telegram("🤖 Agent started - testing Telegram connection")

    # Market-hours check
    if not market_is_open(trading_client):
        print("Market closed, exiting.")
        return

    # ── Opening range protection ───────────────────────────────────────────
    et = pytz.timezone('America/New_York')
    now_et = datetime.datetime.now(et)
    first_trade_allowed = now_et.replace(hour=10, minute=0, second=0, microsecond=0)

    if now_et < first_trade_allowed:
        print("Opening range protection: market open less than 30 min ago")
        print("Running SCAN ONLY mode — no trades will be placed")
        SCAN_ONLY = True
    else:
        SCAN_ONLY = False

    # ── Load history & build stats once per run ───────────────────────────
    history = load_trade_history(sb)
    stats   = build_performance_stats(history)
    if history:
        print(
            f"[history] Loaded {len(history)} trades | "
            f"Overall win rate: {stats['overall_win_rate']}%"
        )

    account         = trading_client.get_account()
    portfolio_value = float(account.portfolio_value)
    existing_pos    = get_existing_positions(trading_client)

    buys_executed = 0
    sells_executed = 0
    skipped        = 0

    print(f"\n{'='*60}")
    print(f"Portfolio value: ${portfolio_value:,.2f} | Open positions: {len(existing_pos)}")
    print(f"{'='*60}\n")

    # ── 1. Check existing positions for sell signals ──────────────────────
    for symbol, position in existing_pos.items():
        try:
            entry_price = float(position.avg_entry_price)
            df          = fetch_data(symbol)
            if df is None:
                continue
            ind = calculate_indicators(df)
            if ind is None:
                continue
            sell, reason = should_sell(ind, entry_price)
            if sell:
                success = close_position(trading_client, symbol)
                if success:
                    sells_executed += 1
                    pnl     = round((ind["price"] - entry_price) * float(position.qty), 2)
                    pnl_pct = round((ind["price"] - entry_price) / entry_price * 100, 2)
                    result  = "WIN" if pnl > 0 else "LOSS"
                    print(f"[{symbol}] SELL | {reason} | Price: ${ind['price']:.2f}")
                    update_sell_trade(sb, symbol, ind["price"])
                    send_telegram(
                        f"🔴 SOLD {symbol}\n"
                        f"Exit: ${ind['price']:.2f}\n"
                        f"PnL: ${pnl} ({pnl_pct}%)\n"
                        f"Result: {result}\n"
                        f"Reason: {reason}\n"
                        f"Portfolio: ${portfolio_value:,.2f}"
                    )
        except Exception as exc:
            print(f"[{symbol}] sell-check error: {exc}")

    # Refresh positions after sells
    existing_pos = get_existing_positions(trading_client)

    # ── 2. Scan watchlist for buy signals ─────────────────────────────────
    for symbol in WATCHLIST:
        try:
            if symbol in existing_pos:
                skipped += 1
                continue

            if len(existing_pos) + buys_executed >= MAX_POSITIONS:
                skipped += 1
                continue

            if earnings_soon(symbol):
                print(f"[{symbol}] Skipping — earnings within {EARNINGS_LOOKOUT} days")
                skipped += 1
                continue

            df = fetch_data(symbol)
            if df is None:
                print(f"[{symbol}] Skipping — insufficient data")
                skipped += 1
                continue

            ind = calculate_indicators(df)
            if ind is None:
                print(f"[{symbol}] Skipping — indicator calculation failed")
                skipped += 1
                continue

            score       = compute_score(ind)
            stop_loss   = ind["price"] - ind["atr"] * ATR_SL_MULTIPLIER
            take_profit = ind["price"] + ind["atr"] * ATR_TP_MULTIPLIER
            sl_pct      = (ind["price"] - stop_loss) / ind["price"] * 100
            tp_pct      = (take_profit - ind["price"]) / ind["price"] * 100

            if not passes_filters(ind, score):
                print(
                    f"[{symbol}] ${ind['price']:.2f} | RSI:{ind['rsi']:.1f} | "
                    f"ADX:{ind['adx']:.1f} | Score:{score}/6 | "
                    f"ATR-SL:${stop_loss:.2f} | ATR-TP:${take_profit:.2f} | "
                    f"Decision:SKIP | filter_fail"
                )
                skipped += 1
                continue

            # Groq confirmation — with history injected
            decision, confidence, reason = groq_confirm(
                groq_client, symbol, ind, score, stats
            )

            if decision == "FALLBACK":
                final_buy  = score >= FALLBACK_MIN_SCORE
                groq_label = f"FALLBACK(score={score})"
            elif decision == "BUY" and confidence >= GROQ_MIN_CONF:
                final_buy  = True
                groq_label = f"BUY({confidence}%) - {reason}"
            else:
                final_buy  = False
                groq_label = f"SKIP({confidence}%) - {reason}"

            print(
                f"[{symbol}] ${ind['price']:.2f} | RSI:{ind['rsi']:.1f} | "
                f"ADX:{ind['adx']:.1f} | Score:{score}/6 | "
                f"ATR-SL:${stop_loss:.2f} | ATR-TP:${take_profit:.2f} | "
                f"Decision:{'BUY' if final_buy else 'SKIP'} | Groq:{groq_label}"
            )

            if not final_buy:
                skipped += 1
                continue

            position_value = portfolio_value * POSITION_PCT
            qty            = int(position_value // ind["price"])
            if qty < 1:
                print(f"  [skip] Calculated qty < 1 (price too high for position size)")
                skipped += 1
                continue

            if SCAN_ONLY:
                print(f"  [SCAN ONLY] Would have bought {symbol} — waiting for 10:00 AM ET")
                skipped += 1
                continue

            success = place_bracket_order(
                trading_client, symbol, qty, stop_loss, take_profit
            )
            if success:
                buys_executed += 1
                print(
                    f"  -> Bracket order placed: {qty} shares | "
                    f"SL:${stop_loss:.2f} | TP:${take_profit:.2f}"
                )
                save_buy_trade(
                    sb,
                    symbol        = symbol,
                    price         = ind["price"],
                    shares        = qty,
                    score         = score,
                    rsi           = ind["rsi"],
                    adx           = ind["adx"],
                    groq_decision = decision,
                    groq_conf     = confidence,
                    reason        = reason,
                    portfolio     = portfolio_value,
                )
                send_telegram(
                    f"🟢 BUY {symbol}\n"
                    f"Price: ${ind['price']:.2f}\n"
                    f"Shares: {qty}\n"
                    f"Stop-Loss: ${stop_loss:.2f}\n"
                    f"Take-Profit: ${take_profit:.2f}\n"
                    f"Score: {score}/6 | Groq: {confidence}%\n"
                    f"Portfolio: ${portfolio_value:,.2f}"
                )
            else:
                skipped += 1

        except Exception as exc:
            print(f"[{symbol}] unexpected error: {exc}")
            skipped += 1

    # ── Summary ───────────────────────────────────────────────────────────
    try:
        portfolio_value = float(trading_client.get_account().portfolio_value)
    except Exception:
        pass

    print(
        f"\n{'='*60}\n"
        f"=== RUN COMPLETE | Buys:{buys_executed} | Sells:{sells_executed} | "
        f"Skipped:{skipped} | Portfolio:${portfolio_value:,.2f} ===\n"
        f"{'='*60}"
    )
    scan_time = datetime.datetime.utcnow().strftime("%H:%M UTC")
    send_telegram(
        f"📊 SCAN COMPLETE {scan_time}\n"
        f"Buys: {buys_executed} | Sells: {sells_executed} | Skipped: {skipped}\n"
        f"Portfolio: ${portfolio_value:,.2f}"
    )

    # ── Weekly report (Fridays only) ──────────────────────────────────────
    if is_weekly_summary_time():
        print_weekly_summary(groq_client, sb)


if __name__ == "__main__":
    run()
