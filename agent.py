"""
Production-ready automated stock trading agent.
Strategy: Multi-Indicator Momentum with Groq AI confirmation.
Broker: Alpaca Paper Trading (alpaca-py).
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import ta
import yfinance as yf

from groq import Groq
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    GetAssetsRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.trading.requests import LimitOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.requests import MarketOrderRequest

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

# ── Alpaca & Groq clients ─────────────────────────────────────────────────────

def get_trading_client() -> TradingClient:
    return TradingClient(
        api_key    = os.environ["ALPACA_API_KEY"],
        secret_key = os.environ["ALPACA_SECRET_KEY"],
        paper      = True,
    )


def get_groq_client() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"])


# ── Market-hours check ────────────────────────────────────────────────────────

def market_is_open(client: TradingClient) -> bool:
    clock = client.get_clock()
    return clock.is_open


# ── Data & indicator calculation ──────────────────────────────────────────────

def fetch_data(symbol: str) -> pd.DataFrame | None:
    """Download 150 days of daily OHLCV data via yfinance."""
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=DATA_DAYS)
    df    = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or len(df) < MIN_ROWS:
        return None
    # Flatten MultiIndex columns that yfinance sometimes returns
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
        macd_obj  = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        macd_line = macd_obj.macd()
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

        # Grab last valid row
        idx = -1
        rsi      = float(rsi_series.iloc[idx])
        macd_val = float(macd_line.iloc[idx])
        signal   = float(signal_line.iloc[idx])
        hist     = float(hist_line.iloc[idx])
        # Previous histogram for "increasing" check
        hist_prev = float(hist_line.iloc[-2]) if len(hist_line) >= 2 else hist

        e9   = float(ema9.iloc[idx])
        e21  = float(ema21.iloc[idx])
        e50  = float(ema50.iloc[idx])
        e200 = float(ema200.iloc[idx])
        atr  = float(atr_series.iloc[idx])
        adx  = float(adx_series.iloc[idx])

        price       = float(close.iloc[idx])
        vol_today   = float(volume.iloc[idx])
        vol_avg20   = float(vol_ma20.iloc[idx])
        vol_ratio   = vol_today / vol_avg20 if vol_avg20 > 0 else 0.0

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
) -> tuple[str, int, str]:
    """
    Ask Groq whether to BUY or SKIP.
    Returns (decision, confidence, reason).
    Falls back to ("FALLBACK", 0, "groq_error") on failure.
    """
    stop_loss   = ind["price"] - ind["atr"] * ATR_SL_MULTIPLIER
    take_profit = ind["price"] + ind["atr"] * ATR_TP_MULTIPLIER
    sl_pct      = (ind["price"] - stop_loss) / ind["price"] * 100
    tp_pct      = (take_profit - ind["price"]) / ind["price"] * 100
    trend_label = "Above" if ind["price"] > ind["ema200"] else "Below"

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
        f"Technical Score: {score}/6\n\n"
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
        # Strip markdown fences if present
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
        ticker    = yf.Ticker(symbol)
        cal       = ticker.calendar
        if cal is None or cal.empty:
            return False
        # calendar is a DataFrame with 'Earnings Date' as column or index
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
            diff = (d.normalize() - today).days
            if 0 <= diff <= EARNINGS_LOOKOUT:
                return True
        return False
    except Exception:
        return False


# ── Alpaca order helpers ──────────────────────────────────────────────────────

def get_existing_positions(client: TradingClient) -> dict[str, object]:
    """Return {symbol: position} for all open positions."""
    try:
        positions = client.get_all_positions()
        return {p.symbol: p for p in positions}
    except Exception:
        return {}


def get_open_position_count(client: TradingClient) -> int:
    return len(get_existing_positions(client))


def place_bracket_order(
    client: TradingClient,
    symbol: str,
    qty: int,
    stop_loss: float,
    take_profit: float,
) -> bool:
    """Place a bracket order (entry + SL + TP). Returns True on success."""
    try:
        req = MarketOrderRequest(
            symbol        = symbol,
            qty           = qty,
            side          = OrderSide.BUY,
            time_in_force = TimeInForce.DAY,
            order_class   = OrderClass.BRACKET,
            stop_loss     = StopLossRequest(stop_price=round(stop_loss, 2)),
            take_profit   = TakeProfitRequest(limit_price=round(take_profit, 2)),
        )
        client.submit_order(req)
        return True
    except Exception as exc:
        print(f"  [order error] {exc}")
        return False


def close_position(client: TradingClient, symbol: str) -> bool:
    """Market-sell entire position. Returns True on success."""
    try:
        client.close_position(symbol)
        return True
    except Exception as exc:
        print(f"  [close error] {exc}")
        return False


# ── Sell logic ────────────────────────────────────────────────────────────────

def should_sell(ind: dict, entry_price: float) -> tuple[bool, str]:
    """
    Returns (True, reason) if any sell signal fires, else (False, "").
    ATR-based SL/TP are managed by Alpaca bracket orders, but we re-check
    here as a safety net in case the bracket was never filled.
    """
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


# ── Main run loop ─────────────────────────────────────────────────────────────

def run():
    trading_client = get_trading_client()
    groq_client    = get_groq_client()

    # Market-hours check
    if not market_is_open(trading_client):
        print("Market closed, exiting.")
        return

    account          = trading_client.get_account()
    portfolio_value  = float(account.portfolio_value)
    existing_pos     = get_existing_positions(trading_client)

    buys_executed    = 0
    sells_executed   = 0
    skipped          = 0

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
                    print(f"[{symbol}] SELL | {reason} | Price: ${ind['price']:.2f}")
        except Exception as exc:
            print(f"[{symbol}] sell-check error: {exc}")

    # Refresh positions after sells
    existing_pos = get_existing_positions(trading_client)

    # ── 2. Scan watchlist for buy signals ─────────────────────────────────
    for symbol in WATCHLIST:
        try:
            # Skip already-held positions
            if symbol in existing_pos:
                skipped += 1
                continue

            # Enforce max-position cap
            if len(existing_pos) + buys_executed >= MAX_POSITIONS:
                skipped += 1
                continue

            # Skip if earnings imminent
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

            # Groq confirmation
            decision, confidence, reason = groq_confirm(groq_client, symbol, ind, score)

            # Determine final action
            if decision == "FALLBACK":
                # Groq unavailable — use fallback score threshold
                final_buy = score >= FALLBACK_MIN_SCORE
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

            # Calculate position size
            position_value = portfolio_value * POSITION_PCT
            qty            = int(position_value // ind["price"])
            if qty < 1:
                print(f"  [skip] Calculated qty < 1 (price too high for position size)")
                skipped += 1
                continue

            success = place_bracket_order(
                trading_client, symbol, qty, stop_loss, take_profit
            )
            if success:
                buys_executed += 1
                print(f"  -> Bracket order placed: {qty} shares | SL:${stop_loss:.2f} | TP:${take_profit:.2f}")
            else:
                skipped += 1

        except Exception as exc:
            print(f"[{symbol}] unexpected error: {exc}")
            skipped += 1

    # ── Summary ───────────────────────────────────────────────────────────
    # Refresh portfolio value
    try:
        account         = trading_client.get_account()
        portfolio_value = float(account.portfolio_value)
    except Exception:
        pass

    print(
        f"\n{'='*60}\n"
        f"=== RUN COMPLETE | Buys:{buys_executed} | Sells:{sells_executed} | "
        f"Skipped:{skipped} | Portfolio:${portfolio_value:,.2f} ===\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    run()
