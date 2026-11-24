from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import httpx
import numpy as np
import base64
import json
import requests
from datetime import datetime
import google.generativeai as genai
from mangum import Mangum

app = FastAPI(title="Forex AI Trading Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENVIRONMENT VARIABLES ---
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
META_API_TOKEN = os.getenv("META_API_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MT5_ACCOUNT_ID = os.getenv("MT5_ACCOUNT_ID", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

META_API_BASE = "https://mt-client-api-v1.agiliumtrade.agiliumtrade.ai"
META_API_HEADERS = {
    "auth-token": META_API_TOKEN,
    "Content-Type": "application/json"
} if META_API_TOKEN else {}

RISK_PER_TRADE_PERCENT = 1.0
MAX_OPEN_POSITIONS = 3
MAX_DAILY_LOSS = 50.0

# --- MODELS ---
class TradeRequest(BaseModel):
    symbol: str
    side: str
    lot: float
    sl: float
    tp: float
    signal_id: Optional[str] = None

class ChartAnalysisRequest(BaseModel):
    symbol: str
    timeframe: str
    image_base64: str

# --- RISK MANAGEMENT ---
def calculate_safe_lot_size(equity: float, risk_percent: float, sl_pips: float, symbol: str) -> float:
    risk_amount = equity * (risk_percent / 100)
    pip_value = 10.0
    if symbol == "XAUUSD": pip_value = 1.0
    lot_size = risk_amount / (sl_pips * pip_value)
    lot_size = round(lot_size * 100) / 100
    return max(0.01, min(lot_size, 10.0))

# --- MARKET DATA ---
class MarketDataService:
    def __init__(self):
        self.api_key = TWELVE_DATA_API_KEY
        self.base_url = "https://api.twelvedata.com"
    
    async def get_candles(self, symbol: str, interval: str, outputsize: int = 100) -> List[Dict]:
        interval_map = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h", "D1": "1day"}
        td_interval = interval_map.get(interval, "1h")
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/time_series", params={"symbol": symbol, "interval": td_interval, "outputsize": outputsize, "apikey": self.api_key})
            data = response.json()
            if "values" not in data: return []
            candles = [{"timestamp": v["datetime"], "open": float(v["open"]), "high": float(v["high"]), "low": float(v["low"]), "close": float(v["close"]), "volume": int(v.get("volume", 0))} for v in data["values"]]
            candles.reverse()
            return candles
    
    async def get_quote(self, symbol: str) -> Dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/quote", params={"symbol": symbol, "apikey": self.api_key})
            data = response.json()
            return {"symbol": symbol, "price": float(data.get("close", 0)), "bid": float(data.get("bid", 0)), "ask": float(data.get("ask", 0)), "spread": float(data.get("ask", 0)) - float(data.get("bid", 0))}

market_data = MarketDataService()

# --- INDICATORS ---
def calculate_ema(closes: List[float], period: int) -> List[float]:
    if len(closes) < period: return []
    multiplier = 2 / (period + 1)
    ema = [np.mean(closes[:period])]
    for i in range(period, len(closes)): ema.append((closes[i] - ema[-1]) * multiplier + ema[-1])
    return ema

def calculate_rsi(closes: List[float], period: int = 14) -> List[float]:
    if len(closes) < period + 1: return []
    deltas = np.diff(closes)
    gains, losses = np.where(deltas > 0, deltas, 0), np.where(deltas < 0, -deltas, 0)
    avg_gain, avg_loss = np.mean(gains[:period]), np.mean(losses[:period])
    rsi_values = []
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi_values.append(100 if avg_loss == 0 else 100 - (100 / (1 + (avg_gain / avg_loss))))
    return rsi_values

def calculate_bollinger_bands(closes: List[float], period: int = 20, std_dev: int = 2) -> Dict:
    if len(closes) < period: return {"upper": [], "middle": [], "lower": []}
    sma = [np.mean(closes[i-period:i]) for i in range(period, len(closes)+1)]
    upper, lower = [], []
    for i in range(period, len(closes) + 1):
        std = np.std(closes[i-period:i])
        upper.append(sma[i-period-1] + (std_dev * std))
        lower.append(sma[i-period-1] - (std_dev * std))
    return {"upper": upper, "middle": sma, "lower": lower}

# --- STRATEGIES ---
class TrendFollowingStrategy:
    def analyze(self, candles: List[Dict]) -> Optional[Dict]:
        if len(candles) < 200: return None
        closes = [c['close'] for c in candles]
        ema_50, ema_200 = calculate_ema(closes, 50), calculate_ema(closes, 200)
        if len(ema_50) < 2 or len(ema_200) < 2: return None
        prev_diff, curr_diff = ema_50[-2] - ema_200[-2], ema_50[-1] - ema_200[-1]
        price = closes[-1]
        if prev_diff <= 0 and curr_diff > 0:
            return {"signal": "BUY", "strategy": "Trend_Following", "entry_price": price, "stop_loss": price * 0.98, "take_profit": price * 1.06, "confidence": 0.75, "reasoning": "Golden Cross: EMA 50 crossed above EMA 200"}
        elif prev_diff >= 0 and curr_diff < 0:
            return {"signal": "SELL", "strategy": "Trend_Following", "entry_price": price, "stop_loss": price * 1.02, "take_profit": price * 0.94, "confidence": 0.75, "reasoning": "Death Cross: EMA 50 crossed below EMA 200"}
        return None

class BollingerBandsStrategy:
    def analyze(self, candles: List[Dict]) -> Optional[Dict]:
        if len(candles) < 50: return None
        closes = [c['close'] for c in candles]
        bb, rsi = calculate_bollinger_bands(closes, 20, 2), calculate_rsi(closes, 14)
        if len(bb['upper']) < 2 or len(rsi) < 2: return None
        price, lower, upper, middle, current_rsi = closes[-1], bb['lower'][-1], bb['upper'][-1], bb['middle'][-1], rsi[-1]
        if price <= lower and current_rsi < 30:
            return {"signal": "BUY", "strategy": "Bollinger_Bands", "entry_price": price, "stop_loss": lower - (middle - lower) * 0.5, "take_profit": middle, "confidence": 0.70, "reasoning": f"Price at lower BB with RSI oversold ({current_rsi:.1f})"}
        elif price >= upper and current_rsi > 70:
            return {"signal": "SELL", "strategy": "Bollinger_Bands", "entry_price": price, "stop_loss": upper + (upper - middle) * 0.5, "take_profit": middle, "confidence": 0.70, "reasoning": f"Price at upper BB with RSI overbought ({current_rsi:.1f})"}
        return None

class SignalEngine:
    def __init__(self): self.strategies = [TrendFollowingStrategy(), BollingerBandsStrategy()]
    async def generate_signals(self, symbol: str) -> List[Dict]:
        candles_h1 = await market_data.get_candles(symbol, "H1", 200)
        candles_m15 = await market_data.get_candles(symbol, "M15", 100)
        if not candles_h1: return []
        signals = []
        for strategy in self.strategies:
            signal = strategy.analyze(candles_m15 if strategy.__class__.__name__ == "BollingerBandsStrategy" else candles_h1)
            if signal:
                signal['symbol'] = symbol
                signals.append(signal)
        signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return signals[:5]

signal_engine = SignalEngine()

# --- ENDPOINTS ---
@app.get("/")
def root(): return {"message": "Forex AI Trading Platform API", "status": "online"}

@app.get("/api/v1/signals/{symbol}")
async def get_signals(symbol: str):
    try:
        signals = await signal_engine.generate_signals(symbol)
        return {"symbol": symbol, "signals": signals, "count": len(signals)}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/market/quote/{symbol}")
async def get_quote(symbol: str):
    try: return await market_data.get_quote(symbol)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/market/scanner")
async def market_scanner():
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    results = {}
    for symbol in symbols:
        try:
            signals = await signal_engine.generate_signals(symbol)
            quote = await market_data.get_quote(symbol)
            results[symbol] = {"price": quote['price'], "signals": signals[:2]}
        except Exception as e: results[symbol] = {"error": str(e)}
    return results

@app.get("/api/v1/account/info")
async def get_account_info():
    if not META_API_TOKEN or not MT5_ACCOUNT_ID: raise HTTPException(status_code=500, detail="MetaAPI not configured.")
    try:
        acc_res = requests.get(f"{META_API_BASE}/users/current/accounts/{MT5_ACCOUNT_ID}", headers=META_API_HEADERS)
        acc_res.raise_for_status()
        acc_data = acc_res.json()
        pos_res = requests.get(f"{META_API_BASE}/users/current/accounts/{MT5_ACCOUNT_ID}/positions", headers=META_API_HEADERS)
        pos_data = pos_res.json() if pos_res.status_code == 200 else []
        return {
            "balance": float(acc_data.get('balance', 0)), "equity": float(acc_data.get('equity', 0)),
            "margin": float(acc_data.get('margin', 0)), "free_margin": float(acc_data.get('marginFree', 0)),
            "profit": float(acc_data.get('profit', 0)), "open_positions": len(pos_data),
            "positions": [{"symbol": p.get('symbol'), "type": p.get('type'), "volume": float(p.get('volume', 0)), "profit": float(p.get('currentProfit', 0)), "sl": float(p.get('stopLoss', 0)), "tp": float(p.get('takeProfit', 0))} for p in pos_data]
        }
    except Exception as e: raise HTTPException(status_code=500, detail=f"MetaAPI error: {str(e)}")

@app.post("/api/v1/trade/execute")
async def execute_live_trade(trade: TradeRequest):
    if not META_API_TOKEN or not MT5_ACCOUNT_ID: raise HTTPException(status_code=500, detail="MetaAPI not configured")
    try:
        pos_res = requests.get(f"{META_API_BASE}/users/current/accounts/{MT5_ACCOUNT_ID}/positions", headers=META_API_HEADERS)
        positions = pos_res.json() if pos_res.status_code == 200 else []
        if len(positions) >= MAX_OPEN_POSITIONS: return {"status": "rejected", "reason": f"Max open positions reached ({MAX_OPEN_POSITIONS})"}
        
        acc_res = requests.get(f"{META_API_BASE}/users/current/accounts/{MT5_ACCOUNT_ID}", headers=META_API_HEADERS)
        acc_data = acc_res.json()
        equity, daily_pnl = float(acc_data.get('equity', 0)), float(acc_data.get('profit', 0))
        if daily_pnl <= -MAX_DAILY_LOSS: return {"status": "rejected", "reason": f"Daily loss limit reached (${MAX_DAILY_LOSS})"}
        
        sl_pips = abs(trade.sl - trade.tp) * 10000 if "JPY" not in trade.symbol else abs(trade.sl - trade.tp) * 100
        final_lot = min(trade.lot, calculate_safe_lot_size(equity, RISK_PER_TRADE_PERCENT, sl_pips, trade.symbol))
        
        trade_payload = {"actionType": "ORDER_TYPE_BUY" if trade.side == "BUY" else "ORDER_TYPE_SELL", "symbol": trade.symbol, "volume": final_lot, "stopLoss": trade.sl, "takeProfit": trade.tp, "comment": f"AI_{trade.signal_id or 'manual'}", "magic": 998877}
        trade_res = requests.post(f"{META_API_BASE}/users/current/accounts/{MT5_ACCOUNT_ID}/trade", headers=META_API_HEADERS, json=trade_payload)
        trade_res.raise_for_status()
        result = trade_res.json()
        
        return {"status": "success", "order_id": result.get('orderId'), "ticket": result.get('ticket'), "lot_size": final_lot, "message": f"{trade.side} {final_lot} {trade.symbol} executed"}
    except Exception as e: raise HTTPException(status_code=500, detail=f"Trade failed: {str(e)}")

@app.post("/api/v1/trade/close-all")
async def close_all_positions():
    if not META_API_TOKEN or not MT5_ACCOUNT_ID: raise HTTPException(status_code=500, detail="MetaAPI not configured")
    try:
        pos_res = requests.get(f"{META_API_BASE}/users/current/accounts/{MT5_ACCOUNT_ID}/positions", headers=META_API_HEADERS)
        positions = pos_res.json() if pos_res.status_code == 200 else []
        closed = 0
        for pos in positions:
            try:
                requests.post(f"{META_API_BASE}/users/current/accounts/{MT5_ACCOUNT_ID}/trade", headers=META_API_HEADERS, json={"actionType": "POSITION_CLOSE_ID", "positionId": pos['id']})
                closed += 1
            except: pass
        return {"status": "success", "closed_positions": closed}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ai/analyze-chart")
async def analyze_chart(request: ChartAnalysisRequest):
    if not GEMINI_API_KEY: raise HTTPException(status_code=500, detail="Gemini API not configured")
    try:
        image_data = base64.b64decode(request.image_base64)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""You are an elite Forex Smart Money Concepts (SMC) analyst. Analyze this {request.symbol} {request.timeframe} chart screenshot.
Identify: 1. Trend direction 2. Support/Resistance 3. Order Blocks (OB) 4. Fair Value Gaps (FVG) 5. BOS/CHOCH 6. Liquidity sweeps 7. Candlestick patterns 8. Entry zone 9. Stop loss 10. Take profit levels (TP1, TP2, TP3).
Output STRICTLY in this JSON format (no markdown, no code blocks):
{{"signal": "BUY or SELL or NO_TRADE", "trend": "bullish or bearish or ranging", "confidence_score": 0.0 to 1.0, "entry_price": number, "stop_loss": number, "take_profit_1": number, "take_profit_2": number, "take_profit_3": number, "risk_reward_ratio": number, "key_levels": {{"support": [numbers], "resistance": [numbers]}}, "patterns_detected": ["list of patterns"], "reasoning": "detailed explanation"}}
If unclear, set signal to "NO_TRADE" and explain why."""
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": image_data}])
        response_text = response.text.strip()
        if response_text.startswith("```"): response_text = '\n'.join(response_text.split('\n')[1:-1])
        analysis = json.loads(response_text)
        analysis.update({"symbol": request.symbol, "timeframe": request.timeframe, "timestamp": str(datetime.utcnow())})
        return analysis
    except json.JSONDecodeError as e: raise HTTPException(status_code=500, detail=f"AI response not valid JSON: {str(e)}")
    except Exception as e: raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# --- VERCEL HANDLER ---
handler = Mangum(app)
