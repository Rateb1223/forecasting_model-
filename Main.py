from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.googlesearch import GoogleSearch
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
from scipy.signal import find_peaks, argrelextrema
import requests
from bs4 import BeautifulSoup
import re
import json

# Current market information - Updated with latest timestamp
CURRENT_DATE_TIME = "2025-06-20 00:35:23"  # Updated exact UTC time
CURRENT_USER = "Rateb1223"
TIMEFRAME = "15M"  # Exclusively focused on 15-minute timeframe
ACCOUNT_BALANCE = 77.60  # Account balance

# Enhanced profit strategy settings
PRICE_SAMPLES = 1000  # Analyze 1000+ price samples for thorough market understanding
FIBONACCI_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.0]  # Standard Fib levels
ADVANCED_CANDLESTICK_DETECTION = True  # Enhanced candlestick pattern recognition
SMART_SUPPORT_RESISTANCE = True  # Smart support/resistance detection
RISK_PER_TRADE = 2.5  # Higher risk percentage for higher returns (2.5%)
PROFIT_TARGET_MULTIPLIER = 3.0  # More aggressive profit targets (3x risk)
MULTI_STRATEGY_CONFIRMATION = True  # Only take trades with multiple strategy confirmations
PRICE_ACTION_FOCUS = True  # Emphasize price action over indicators

# Top 10 forex news sources
TOP_FOREX_NEWS_SOURCES = [
    {
        "name": "ForexLive",
        "url": "https://www.forexlive.com/",
        "type": "breaking news",
        "reliability": 90
    },
    {
        "name": "DailyFX",
        "url": "https://www.dailyfx.com/market-news",
        "type": "analysis and news",
        "reliability": 95
    },
    {
        "name": "FXStreet",
        "url": "https://www.fxstreet.com/news",
        "type": "comprehensive news",
        "reliability": 92
    },
    {
        "name": "Investing.com Forex",
        "url": "https://www.investing.com/news/forex-news",
        "type": "market coverage",
        "reliability": 89
    },
    {
        "name": "Bloomberg Markets",
        "url": "https://www.bloomberg.com/markets/currencies",
        "type": "institutional coverage",
        "reliability": 97
    },
    {
        "name": "Reuters Forex",
        "url": "https://www.reuters.com/markets/currencies/",
        "type": "global news",
        "reliability": 96
    },
    {
        "name": "BabyPips",
        "url": "https://www.babypips.com/news",
        "type": "educational news",
        "reliability": 88
    },
    {
        "name": "FX Empire",
        "url": "https://www.fxempire.com/news/forex-news",
        "type": "market analysis",
        "reliability": 87
    },
    {
        "name": "CNBC Forex",
        "url": "https://www.cnbc.com/currencies/",
        "type": "financial news",
        "reliability": 94
    },
    {
        "name": "Financial Times Forex",
        "url": "https://www.ft.com/currencies",
        "type": "institutional insight",
        "reliability": 95
    }
]

# Create a specialized profit maximizer agent with news integration
forex_news_profit_agent = Agent(
    name="Forex News-Integrated Profit Maximizer",
    role="Generate maximum profits by combining technical analysis and top forex news",
    model=Gemini(id="gemini-1.5-flash"),  # Using Google's Gemini model
    tools=[
        GoogleSearch(),  # For market news searches
    ],
    instructions=[
        f"You are {CURRENT_USER}'s forex profit maximizer with integrated news analysis",
        f"Current time is {CURRENT_DATE_TIME} UTC",
        f"EXCLUSIVELY analyze and trade on the 15M chart timeframe ONLY",
        "Your PRIMARY GOAL is to MAXIMIZE PROFITS using advanced trading strategies AND news catalysts",
        "Utilize top 10 forex news sources for market-moving events and sentiment",
        "Employ multiple confirmation strategies: Technical Analysis + News + Price Action",
        "Only recommend trades with technical confirmation AND supporting news sentiment",
        "ALWAYS present trading signals in a clear markdown table format",
        "Table MUST include: Pair, Signal, Entry, Stop Loss, TP Targets, Win Probability, News Impact",
        "Focus on high-probability setups with exceptional risk:reward ratios",
        "Pay special attention to how news catalysts affect price action",
        "Identify key support/resistance levels and how news might trigger breakouts",
        "Apply Fibonacci retracement and extension for precise entries with news confirmation",
        "Rate each setup by profit potential, news impact, and statistical edge",
        "Present trading signals in the table first, followed by detailed strategy analysis",
        "Make specific recommendations on entry timing relative to news events",
        "Detail exact price levels and how news might affect price volatility",
        "ONLY recommend trades with exceptional profit potential, strong technical setup AND news catalyst"
    ],
    show_tool_calls=True,
    markdown=True,
)

# Function to calculate the next several 15-minute candle times
def get_15m_candle_times(current_time_str=CURRENT_DATE_TIME, num_candles=8):
    # Parse current time
    current_time = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M:%S")
    
    # Round down to the nearest 15-minute mark
    minute = current_time.minute
    nearest_15min = (minute // 15) * 15
    current_candle_start = current_time.replace(minute=nearest_15min, second=0, microsecond=0)
    
    # Calculate the next candles
    candle_times = []
    for i in range(num_candles):
        next_candle = current_candle_start + timedelta(minutes=15 * i)
        candle_times.append(next_candle.strftime("%H:%M"))
    
    return {
        "current_candle": candle_times[0],
        "next_candles": candle_times[1:],
        "current_candle_end": (current_candle_start + timedelta(minutes=15)).strftime("%H:%M"),
        "candle_list": candle_times
    }

# Function to fetch forex news from top sources using DuckDuckGo
def fetch_forex_news(currency_pair, duckduckgo_tool, max_results=5):
    """Fetch specialized forex news for a specific currency pair"""
    
    print(f"Fetching specialized forex news for {currency_pair}...")
    
    news_results = []
    
    # Create search queries for different sources
    search_queries = [
        f"{currency_pair} forex news today site:forexlive.com",
        f"{currency_pair} analysis today site:dailyfx.com",
        f"{currency_pair} forecast today site:fxstreet.com",
        f"{currency_pair} outlook today site:investing.com",
        f"{currency_pair} market news site:bloomberg.com",
        f"{currency_pair} trading analysis site:reuters.com",
        f"{currency_pair} news today site:babypips.com",
        f"{currency_pair} market analysis site:fxempire.com",
        f"{currency_pair} forecast site:cnbc.com",
        f"{currency_pair} analysis site:ft.com"
    ]
    
    collected_news = []
    
    # Search each specialized forex source
    for query in search_queries:
        try:
            results = duckduckgo_tool.search(query)
            if results and len(results) > 0:
                # Extract top result
                top_news = results[0]
                source = extract_domain(top_news.get('href', ''))
                
                # Add to collected news if not already included
                if not any(news['source'] == source for news in collected_news):
                    collected_news.append({
                        'title': top_news.get('title', 'No title available'),
                        'snippet': top_news.get('snippet', 'No snippet available'),
                        'url': top_news.get('href', ''),
                        'source': source,
                        'date': 'Today'  # Assuming recent news due to search query
                    })
                
                # Break after collecting enough news
                if len(collected_news) >= max_results:
                    break
        except Exception as e:
            print(f"Error fetching news from query '{query}': {str(e)}")
    
    # Format the news for display
    for item in collected_news:
        source_info = next((s for s in TOP_FOREX_NEWS_SOURCES if item['source'] in s['url']), 
                          {'name': item['source'], 'reliability': 70})
        
        news_results.append({
            'title': item['title'],
            'source': source_info.get('name', item['source']),
            'snippet': item['snippet'],
            'reliability': source_info.get('reliability', 70),
            'url': item['url'],
            'sentiment': analyze_news_sentiment(item['title'] + ' ' + item['snippet'])
        })
    
    return news_results

# Helper function to extract domain from URL
def extract_domain(url):
    """Extract domain name from URL"""
    match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    if match:
        return match.group(1)
    return url

# Function to analyze news sentiment
def analyze_news_sentiment(text):
    """Basic sentiment analysis for forex news"""
    
    # Bullish keywords
    bullish_words = ['bullish', 'rally', 'surge', 'gain', 'jump', 'soar', 'rise', 'strengthen', 
                     'outperform', 'upside', 'higher', 'increase', 'positive', 'hawkish',
                     'growth', 'strong', 'breakthrough', 'recovery', 'optimistic', 'uptrend']
    
    # Bearish keywords
    bearish_words = ['bearish', 'plunge', 'fall', 'drop', 'decline', 'weaken', 'slide', 'tumble',
                    'underperform', 'downside', 'lower', 'decrease', 'negative', 'dovish',
                    'recession', 'weak', 'breakdown', 'pessimistic', 'downtrend']
    
    # Count sentiment words
    bullish_count = sum(1 for word in bullish_words if word in text.lower())
    bearish_count = sum(1 for word in bearish_words if word in text.lower())
    
    # Determine sentiment
    if bullish_count > bearish_count:
        strength = min(100, int(bullish_count * 10))
        return {"direction": "bullish", "strength": strength}
    elif bearish_count > bullish_count:
        strength = min(100, int(bearish_count * 10))
        return {"direction": "bearish", "strength": strength}
    else:
        return {"direction": "neutral", "strength": 50}

# Enhanced candlestick pattern detection function
def detect_advanced_candlestick_patterns(df):
    """Detect advanced candlestick patterns with profit implications"""
    
    patterns = {}
    
    # Store last 5 candles for recent pattern detection
    last_5_opens = df["Open"].iloc[-5:].values
    last_5_highs = df["High"].iloc[-5:].values
    last_5_lows = df["Low"].iloc[-5:].values
    last_5_closes = df["Close"].iloc[-5:].values
    
    # Calculate candle properties
    bodies = abs(last_5_opens - last_5_closes)
    body_ranges = bodies / (last_5_highs - last_5_lows)
    upper_shadows = last_5_highs - np.maximum(last_5_opens, last_5_closes)
    lower_shadows = np.minimum(last_5_opens, last_5_closes) - last_5_lows
    
    # Recent candle directions (bullish/bearish)
    candle_direction = np.where(last_5_closes > last_5_opens, "Bullish", "Bearish")
    
    # Check for Doji (tiny body)
    doji_present = any(body_ranges < 0.1)
    if doji_present:
        doji_idx = np.where(body_ranges < 0.1)[0][-1]  # Get most recent
        patterns["Doji"] = {
            "index": len(df) - 5 + doji_idx,
            "confidence": 90 if body_ranges[doji_idx] < 0.05 else 70
        }
    
    # Check for Hammer/Hanging Man (small body, long lower shadow, small upper shadow)
    hammer_conditions = (
        (body_ranges < 0.3) & 
        (lower_shadows > 2 * bodies) & 
        (upper_shadows < 0.2 * lower_shadows)
    )
    
    if any(hammer_conditions):
        hammer_idx = np.where(hammer_conditions)[0][-1]  # Get most recent
        pattern_name = "Hammer" if candle_direction[hammer_idx] == "Bullish" else "Hanging Man"
        patterns[pattern_name] = {
            "index": len(df) - 5 + hammer_idx,
            "confidence": 80
        }
    
    # Check for Engulfing patterns (current candle's body engulfs previous candle's body)
    for i in range(1, len(last_5_opens)):
        prev_body_low = min(last_5_opens[i-1], last_5_closes[i-1])
        prev_body_high = max(last_5_opens[i-1], last_5_closes[i-1])
        curr_body_low = min(last_5_opens[i], last_5_closes[i])
        curr_body_high = max(last_5_opens[i], last_5_closes[i])
        
        # Bullish engulfing
        if (candle_direction[i] == "Bullish" and 
            candle_direction[i-1] == "Bearish" and
            curr_body_low <= prev_body_low and
            curr_body_high >= prev_body_high):
            patterns["Bullish Engulfing"] = {
                "index": len(df) - 5 + i,
                "confidence": 85
            }
        
        # Bearish engulfing
        elif (candle_direction[i] == "Bearish" and 
              candle_direction[i-1] == "Bullish" and
              curr_body_low <= prev_body_low and
              curr_body_high >= prev_body_high):
            patterns["Bearish Engulfing"] = {
                "index": len(df) - 5 + i,
                "confidence": 85
            }
    
    # Check for Morning/Evening Star (three candle pattern)
    if len(last_5_opens) >= 3:
        # Morning Star (bearish, small, bullish)
        if (candle_direction[-3] == "Bearish" and
            body_ranges[-2] < 0.3 and
            candle_direction[-1] == "Bullish" and
            last_5_closes[-1] > (last_5_opens[-3] + last_5_closes[-3]) / 2):
            patterns["Morning Star"] = {
                "index": len(df) - 1,
                "confidence": 90
            }
        
        # Evening Star (bullish, small, bearish)
        if (candle_direction[-3] == "Bullish" and
            body_ranges[-2] < 0.3 and
            candle_direction[-1] == "Bearish" and
            last_5_closes[-1] < (last_5_opens[-3] + last_5_closes[-3]) / 2):
            patterns["Evening Star"] = {
                "index": len(df) - 1,
                "confidence": 90
            }
    
    # Check for Shooting Star (small body at bottom, long upper shadow)
    shooting_star_conditions = (
        (body_ranges < 0.3) & 
        (upper_shadows > 2 * bodies) & 
        (lower_shadows < 0.2 * upper_shadows)
    )
    
    if any(shooting_star_conditions):
        ss_idx = np.where(shooting_star_conditions)[0][-1]  # Get most recent
        patterns["Shooting Star"] = {
            "index": len(df) - 5 + ss_idx,
            "confidence": 80
        }
    
    # Check for Three White Soldiers / Three Black Crows (three strong candles in same direction)
    if len(last_5_opens) >= 3:
        # Three White Soldiers (three bullish candles with higher highs and higher lows)
        if (all(candle_direction[-3:] == "Bullish") and
            all(bodies[-3:] / (last_5_highs[-3:] - last_5_lows[-3:]) > 0.5) and  # Strong bodies
            all(np.diff(last_5_highs[-3:]) > 0) and  # Higher highs
            all(np.diff(last_5_lows[-3:]) > 0)):     # Higher lows
            patterns["Three White Soldiers"] = {
                "index": len(df) - 1,
                "confidence": 95
            }
        
        # Three Black Crows (three bearish candles with lower highs and lower lows)
        if (all(candle_direction[-3:] == "Bearish") and
            all(bodies[-3:] / (last_5_highs[-3:] - last_5_lows[-3:]) > 0.5) and  # Strong bodies
            all(np.diff(last_5_highs[-3:]) < 0) and  # Lower highs
            all(np.diff(last_5_lows[-3:]) < 0)):     # Lower lows
            patterns["Three Black Crows"] = {
                "index": len(df) - 1,
                "confidence": 95
            }
    
    return patterns

# Enhanced support and resistance detection with fractal analysis
def detect_smart_support_resistance(df, window=100, sensitivity=0.0005):
    """Detect smart support and resistance levels using fractal analysis"""
    
    if len(df) < window:
        window = len(df) // 2
    
    # Get recent price data for analysis
    recent_df = df.iloc[-window:]
    highs = recent_df["High"].values
    lows = recent_df["Low"].values
    closes = recent_df["Close"].values
    current_price = closes[-1]
    
    # Find local maxima and minima using scipy
    high_idx = argrelextrema(highs, np.greater, order=5)[0]  # Local maxima with 5-bar window
    low_idx = argrelextrema(lows, np.less, order=5)[0]       # Local minima with 5-bar window
    
    # Extract price levels from these points
    resistance_levels = [highs[i] for i in high_idx]
    support_levels = [lows[i] for i in low_idx]
    
    # Add round number levels (psychological levels)
    # For regular pairs, use 0.05 intervals, for JPY pairs use 0.5 intervals
    is_jpy_pair = "JPY" in df.name if hasattr(df, "name") else False
    
    if is_jpy_pair:
        # Round numbers for JPY pairs (e.g. 150.00, 150.50)
        step = 0.5
        base = np.floor(current_price / 10) * 10  # Base 10
        for i in range(-10, 11):
            level = base + i * step
            if abs(level - current_price) / current_price < 0.02:  # Within 2%
                if level > current_price:
                    resistance_levels.append(level)
                else:
                    support_levels.append(level)
    else:
        # Round numbers for regular pairs (e.g. 1.1000, 1.1050)
        step = 0.0050
        base = np.floor(current_price * 100) / 100  # Base 0.01
        for i in range(-10, 11):
            level = base + i * step
            if abs(level - current_price) / current_price < 0.01:  # Within 1%
                if level > current_price:
                    resistance_levels.append(level)
                else:
                    support_levels.append(level)
    
    # Cluster similar levels (within sensitivity threshold)
    def cluster_levels(levels, sensitivity):
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for i in range(1, len(levels)):
            if (levels[i] - current_cluster[-1]) / current_cluster[-1] < sensitivity:
                current_cluster.append(levels[i])
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [levels[i]]
        
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
        
        return clusters
    
    # Cluster the levels
    resistance_clusters = cluster_levels(resistance_levels, sensitivity)
    support_clusters = cluster_levels(support_levels, sensitivity)
    
    # Sort by proximity to current price
    resistance_levels = sorted([r for r in resistance_clusters if r > current_price])
    support_levels = sorted([s for s in support_clusters if s < current_price], reverse=True)
    
    # Test the strength of each level by counting touches
    level_strength = {}
    
    # Function to count how many times price approached a level
    def count_level_touches(price_series, level, sensitivity):
        touches = 0
        for price in price_series:
            if abs(price - level) / level < sensitivity:
                touches += 1
        return touches
    
    # Count touches for each resistance level
    for level in resistance_levels:
        touches = count_level_touches(highs, level, sensitivity)
        level_strength[level] = {
            "touches": touches,
            "strength": min(touches * 20, 100)  # Scale: 1 touch = 20%, max 100%
        }
    
    # Count touches for each support level
    for level in support_levels:
        touches = count_level_touches(lows, level, sensitivity)
        level_strength[level] = {
            "touches": touches,
            "strength": min(touches * 20, 100)  # Scale: 1 touch = 20%, max 100%
        }
    
    # Return final levels with strength data
    return {
        "support": support_levels[:5],  # Top 5 support levels
        "resistance": resistance_levels[:5],  # Top 5 resistance levels
        "strength": level_strength
    }

# Advanced Fibonacci analysis function
def perform_fibonacci_analysis(df, window=100):
    """Analyze price using Fibonacci retracement and extension levels"""
    
    if len(df) < window:
        window = len(df) // 2
    
    # Get recent price data
    recent_df = df.iloc[-window:]
    highs = recent_df["High"].values
    lows = recent_df["Low"].values
    closes = recent_df["Close"].values
    current_price = closes[-1]
    
    # Identify the most recent significant swing high and swing low
    high_idx = argrelextrema(highs, np.greater, order=10)[0]  # More significant swings
    low_idx = argrelextrema(lows, np.less, order=10)[0]
    
    if len(high_idx) == 0 or len(low_idx) == 0:
        return {
            "valid": False,
            "message": "Insufficient swing points for Fibonacci analysis"
        }
    
    # Get the most recent swing points
    last_swing_high_idx = high_idx[-1] if len(high_idx) > 0 else None
    last_swing_low_idx = low_idx[-1] if len(low_idx) > 0 else None
    
    # If no valid swing points were found
    if last_swing_high_idx is None or last_swing_low_idx is None:
        return {
            "valid": False,
            "message": "No valid swing points identified"
        }
    
    # Determine the most recent swing (high-to-low or low-to-high)
    if last_swing_high_idx > last_swing_low_idx:
        # Downtrend: High to Low
        swing_high = highs[last_swing_high_idx]
        swing_low = min(lows[last_swing_high_idx:])
        direction = "downtrend"
        
        # Calculate Fibonacci retracement levels (upward retracement in downtrend)
        fib_levels = {}
        for level in FIBONACCI_LEVELS:
            fib_levels[level] = swing_low + level * (swing_high - swing_low)
        
        # Calculate extension levels for continued downtrend
        extensions = {}
        for level in [1.272, 1.618, 2.0, 2.618]:
            extensions[level] = swing_low - level * (swing_high - swing_low)
        
    else:
        # Uptrend: Low to High
        swing_low = lows[last_swing_low_idx]
        swing_high = max(highs[last_swing_low_idx:])
        direction = "uptrend"
        
        # Calculate Fibonacci retracement levels (downward retracement in uptrend)
        fib_levels = {}
        for level in FIBONACCI_LEVELS:
            fib_levels[level] = swing_high - level * (swing_high - swing_low)
        
        # Calculate extension levels for continued uptrend
        extensions = {}
        for level in [1.272, 1.618, 2.0, 2.618]:
            extensions[level] = swing_high + level * (swing_high - swing_low)
    
    # Find active Fibonacci levels (closest to current price)
    active_levels = []
    
    # Check retracement levels
    for level, price in fib_levels.items():
        if abs(price - current_price) / current_price < 0.005:  # Within 0.5%
            active_levels.append({
                "type": "Retracement", 
                "level": level, 
                "price": price,
                "distance": abs(price - current_price) / current_price,
                "direction": "support" if price < current_price else "resistance"
            })
    
    # Check extension levels
    for level, price in extensions.items():
        if abs(price - current_price) / current_price < 0.01:  # Within 1%
            active_levels.append({
                "type": "Extension", 
                "level": level, 
                "price": price,
                "distance": abs(price - current_price) / current_price,
                "direction": "support" if price < current_price else "resistance"
            })
    
    # Sort active levels by distance to current price
    active_levels.sort(key=lambda x: x["distance"])
    
    return {
        "valid": True,
        "direction": direction,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "retracement_levels": fib_levels,
        "extension_levels": extensions,
        "active_levels": active_levels[:3]  # Top 3 closest levels
    }

# Function to get MT5 data with advanced indicators
def get_mt5_advanced_data(symbol="EURUSD", days=30, timeframe=mt5.TIMEFRAME_M15):
    """Fetch and process MT5 data with advanced indicators for profit maximization"""
    
    print(f"Connecting to MetaTrader5 for {symbol} data analysis...")
    
    # Initialize MT5 connection
    if not mt5.initialize():
        print(f"MT5 initialization failed, error code = {mt5.last_error()}")
        return None
    
    print(f"Connected to MT5 successfully!")
    
    # Set time range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days) 
    
    # Fetch data
    print(f"Fetching {symbol} {TIMEFRAME} data for the past {days} days...")
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    
    # Check if data was retrieved successfully
    if rates is None or len(rates) == 0:
        print(f"Failed to retrieve data for {symbol}")
        mt5.shutdown()
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.drop(['spread', 'real_volume'], axis=1, inplace=True)
    df.columns = ["Time", "Open", "High", "Low", "Close", "Volume"]
    
    # Set Time as index
    df.set_index("Time", inplace=True)
    df.name = symbol  # Store symbol name for later reference
    
    print(f"Data retrieved successfully! {len(df)} samples | Last price: {df['Close'].iloc[-1]}")
    
    # Add comprehensive technical indicators
    print("Adding advanced technical indicators...")
    
    # Moving Averages & Trend indicators
    df["SMA_20"] = talib.SMA(df["Close"], timeperiod=20)
    df["SMA_50"] = talib.SMA(df["Close"], timeperiod=50)
    df["SMA_200"] = talib.SMA(df["Close"], timeperiod=200)
    df["EMA_9"] = talib.EMA(df["Close"], timeperiod=9)
    df["EMA_21"] = talib.EMA(df["Close"], timeperiod=21)
    df["EMA_55"] = talib.EMA(df["Close"], timeperiod=55)
    
    # Momentum indicators
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = talib.MACD(df["Close"])
    df["RSI"] = talib.RSI(df["Close"], timeperiod=14)
    df["ADX"] = talib.ADX(df["High"], df["Low"], df["Close"], timeperiod=14)
    df["SlowK"], df["SlowD"] = talib.STOCH(df["High"], df["Low"], df["Close"])
    
    # Volatility indicators
    df["ATR"] = talib.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
    df["Upper_Band"], df["Middle_Band"], df["Lower_Band"] = talib.BBANDS(df["Close"], 
                                                                       timeperiod=20, 
                                                                       nbdevup=2, 
                                                                       nbdevdn=2)
    # Volume indicators
    df["OBV"] = talib.OBV(df["Close"], df["Volume"])
    df["MFI"] = talib.MFI(df["High"], df["Low"], df["Close"], df["Volume"], timeperiod=14)
    
    # Advanced indicators
    
    # 1. Heikin Ashi calculations (trend filtering)
    df["HA_Open"] = (df["Open"].shift(1) + df["Close"].shift(1)) / 2
    df["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    df["HA_High"] = df[["High", "HA_Open", "HA_Close"]].max(axis=1)
    df["HA_Low"] = df[["Low", "HA_Open", "HA_Close"]].min(axis=1)
    
    # 2. SuperTrend Indicator (10,3)
    atr = df["ATR"]
    factor = 3.0
    
    # Calculate basic upper and lower bands
    basic_upper_band = ((df["High"] + df["Low"]) / 2) + (factor * atr)
    basic_lower_band = ((df["High"] + df["Low"]) / 2) - (factor * atr)
    
    # Initialize SuperTrend columns
    df["ST_UpTrend"] = True
    df["ST_DownTrend"] = False
    df["SuperTrend"] = 0.0
    
    # Calculate SuperTrend recursively
    for i in range(1, len(df)):
        # Lower/upper band adjustments
        if basic_upper_band[i] < df["SuperTrend"][i-1] or df["Close"][i-1] > df["SuperTrend"][i-1]:
            final_upper_band = basic_upper_band[i]
        else:
            final_upper_band = df["SuperTrend"][i-1]
            
        if basic_lower_band[i] > df["SuperTrend"][i-1] or df["Close"][i-1] < df["SuperTrend"][i-1]:
            final_lower_band = basic_lower_band[i]
        else:
            final_lower_band = df["SuperTrend"][i-1]
        
        # Determine trend direction
        if df["Close"][i] > final_upper_band:
            df.loc[df.index[i], "SuperTrend"] = final_lower_band
            df.loc[df.index[i], "ST_UpTrend"] = True
            df.loc[df.index[i], "ST_DownTrend"] = False
        else:
            df.loc[df.index[i], "SuperTrend"] = final_upper_band
            df.loc[df.index[i], "ST_UpTrend"] = False
            df.loc[df.index[i], "ST_DownTrend"] = True
    
    # Close the MT5 connection
    mt5.shutdown()
    print("MT5 connection closed")
    
    # Perform advanced analysis on the data
    
    # 1. Detect smart support/resistance levels
    print("Detecting smart support and resistance levels...")
    df.support_resistance = detect_smart_support_resistance(df)
    
    # 2. Perform Fibonacci analysis
    print("Performing Fibonacci analysis...")
    df.fibonacci = perform_fibonacci_analysis(df)
    
    # 3. Detect advanced candlestick patterns
    print("Detecting advanced candlestick patterns...")
    df.candlestick_patterns = detect_advanced_candlestick_patterns(df)
    
    print(f"Technical analysis complete with {len(df)} samples!")
    
    return df

# Function to get MT5 current price
def get_mt5_current_price(symbol="EURUSD"):
    """Get the current live price from MT5"""
    
    # Initialize MT5 connection
    if not mt5.initialize():
        print(f"MT5 initialization failed, error code = {mt5.last_error()}")
        return None
    
    # Get the last price
    last_tick = mt5.symbol_info_tick(symbol)
    
    # Close MT5 connection
    mt5.shutdown()
    
    if last_tick is None:
        print(f"Failed to retrieve price for {symbol}")
        return None
        
    # Use the average of bid and ask for the current price
    current_price = (last_tick.bid + last_tick.ask) / 2
    print(f"Current {symbol} price: {current_price:.5f}")
    
    return current_price

# Function to generate news-enhanced profit signals
def generate_news_enhanced_signal(currency_pair="EURUSD", account_balance=ACCOUNT_BALANCE, risk_percentage=RISK_PER_TRADE):
    """Generate profit-maximizing signals enhanced with specialized forex news analysis"""
    
    print(f"Generating news-enhanced profit signal for {currency_pair}...")
    
    # Calculate 15-minute candle times
    candle_times = get_15m_candle_times()
    
    try:
        # Get current price first to ensure we have it
        current_price = get_mt5_current_price(currency_pair)
        if current_price is None:
            return "Failed to get current price from MT5. Check your connection."
            
        # Get technical analysis data
        df = get_mt5_advanced_data(currency_pair, days=30)
        if df is None:
            return "Failed to retrieve historical data from MT5. Check your connection."
        
        # Create DuckDuckGo tool instance for news search
        duckduckgo_tool = DuckDuckGo()
        
        # Fetch specialized forex news
        news_results = fetch_forex_news(currency_pair, duckduckgo_tool)
        
        # Get sample count and last data point time
        sample_count = len(df)
        last_data_time = df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate pip multiplier based on currency pair
        pip_multiplier = 10000 if "JPY" not in currency_pair else 100
        
        # Calculate ATR for stop loss
        current_atr = df["ATR"].iloc[-1]
        suggested_sl_pips = round(current_atr * pip_multiplier * 1.5)  # Convert ATR to pips, 1.5x ATR
        
        # Get strategy analysis results
        support_resistance = df.support_resistance if hasattr(df, "support_resistance") else {}
        fibonacci = df.fibonacci if hasattr(df, "fibonacci") else {"valid": False}
        candlestick_patterns = df.candlestick_patterns if hasattr(df, "candlestick_patterns") else {}
        
        # Get current technical indicator values
        rsi = df["RSI"].iloc[-1]
        macd = df["MACD"].iloc[-1]
        macd_signal = df["MACD_Signal"].iloc[-1]
        adx = df["ADX"].iloc[-1]
        trend = "Uptrend" if df["EMA_9"].iloc[-1] > df["EMA_21"].iloc[-1] else "Downtrend"
        
        # Determine if price is near key levels
        near_support = False
        near_resistance = False
        key_level = None
        
        if "support" in support_resistance and support_resistance["support"]:
            closest_support = min(support_resistance["support"], key=lambda x: abs(current_price - x))
            if abs(current_price - closest_support) / current_price < 0.001:  # Within 0.1%
                near_support = True
                key_level = closest_support
        
        if "resistance" in support_resistance and support_resistance["resistance"]:
            closest_resistance = min(support_resistance["resistance"], key=lambda x: abs(current_price - x))
            if abs(current_price - closest_resistance) / current_price < 0.001:  # Within 0.1%
                near_resistance = True
                key_level = closest_resistance
        
        # Analyze news sentiment and get average direction
        news_sentiment = "neutral"
        news_strength = 0
        
        if news_results:
            bullish_count = sum(1 for item in news_results if item['sentiment']['direction'] == 'bullish')
            bearish_count = sum(1 for item in news_results if item['sentiment']['direction'] == 'bearish')
            
            if bullish_count > bearish_count:
                news_sentiment = "bullish"
                news_strength = sum(item['sentiment']['strength'] for item in news_results 
                                   if item['sentiment']['direction'] == 'bullish') / max(bullish_count, 1)
            elif bearish_count > bullish_count:
                news_sentiment = "bearish"
                news_strength = sum(item['sentiment']['strength'] for item in news_results 
                                   if item['sentiment']['direction'] == 'bearish') / max(bearish_count, 1)
        
        # Determine signal direction based on technical analysis and news
        technical_direction = "NO TRADE"
        
        # Technical analysis signals
        bullish_signals = 0
        bearish_signals = 0
        
        # Count bullish technical signals
        if df["EMA_9"].iloc[-1] > df["EMA_21"].iloc[-1]: bullish_signals += 1
        if df["MACD"].iloc[-1] > df["MACD_Signal"].iloc[-1]: bullish_signals += 1
        if rsi < 30: bullish_signals += 1  # Oversold
        if near_support: bullish_signals += 1
        if "Hammer" in candlestick_patterns or "Morning Star" in candlestick_patterns or "Bullish Engulfing" in candlestick_patterns:
            bullish_signals += 1
        
        # Count bearish technical signals
        if df["EMA_9"].iloc[-1] < df["EMA_21"].iloc[-1]: bearish_signals += 1
        if df["MACD"].iloc[-1] < df["MACD_Signal"].iloc[-1]: bearish_signals += 1
        if rsi > 70: bearish_signals += 1  # Overbought
        if near_resistance: bearish_signals += 1
        if "Shooting Star" in candlestick_patterns or "Evening Star" in candlestick_patterns or "Bearish Engulfing" in candlestick_patterns:
            bearish_signals += 1
        
        # Determine technical direction
        if bullish_signals >= 3 and bullish_signals > bearish_signals:
            technical_direction = "BUY"
        elif bearish_signals >= 3 and bearish_signals > bullish_signals:
            technical_direction = "SELL"
        
        # Combine technical and news analysis
        final_direction = "NO TRADE"
        
        # Only trade when news confirms technical analysis
        if technical_direction == "BUY" and news_sentiment == "bullish":
            final_direction = "BUY"
        elif technical_direction == "SELL" and news_sentiment == "bearish":
            final_direction = "SELL"
        elif technical_direction != "NO TRADE" and news_sentiment == "neutral":
            # With neutral news, we can still trade but with lower confidence
            final_direction = technical_direction
        
        # Calculate win probability based on technicals and news
        technical_probability = 50 + (5 * max(bullish_signals, bearish_signals))
        
        # News confirmation bonus (0-20%)
        news_bonus = 0
        if (final_direction == "BUY" and news_sentiment == "bullish") or (final_direction == "SELL" and news_sentiment == "bearish"):
            news_bonus = min(20, int(news_strength / 5))
        
        # Calculate final win probability
        win_probability = min(technical_probability + news_bonus, 95)  # Cap at 95%
        
        # Format news headlines for display
        news_items_text = "\n".join([f"- {item['source']}: {item['title']} ({item['sentiment']['direction'].title()}, {item['sentiment']['strength']}%)" 
                                 for item in news_results[:3]]) if news_results else "No relevant news found"
        
        # Create prompt focused on news-enhanced profit maximization
        prompt = f"""
        # 15-MINUTE NEWS-ENHANCED PROFIT SIGNAL
        
        ## Market Overview
        - Current Time (UTC): {CURRENT_DATE_TIME}
        - Current 15M Candle: {candle_times['current_candle']}-{candle_times['current_candle_end']}
        - Currency Pair: {currency_pair}
        - Current Price: {current_price:.5f}
        - Account Balance: ${account_balance}
        
        ## Technical Analysis Summary
        - Trend: {trend}
        - RSI(14): {rsi:.1f} (Overbought > 70, Oversold < 30)
        - MACD: {macd:.5f} vs Signal {macd_signal:.5f}
        - ADX: {adx:.1f} ({"Strong" if adx > 25 else "Weak"} trend)
        - Key Level: {"Support at "+str(key_level) if near_support else "Resistance at "+str(key_level) if near_resistance else "None nearby"}
        - Technical Direction: {technical_direction}
        
        ## Latest Specialized Forex News
        {news_items_text}
        
        ## News Sentiment Analysis
        - Overall News Sentiment: {news_sentiment.title()}
        - Sentiment Strength: {news_strength:.1f}%
        - News Impact on Trading: {"Confirming" if (final_direction == "BUY" and news_sentiment == "bullish") or (final_direction == "SELL" and news_sentiment == "bearish") else "Contradicting" if (final_direction == "BUY" and news_sentiment == "bearish") or (final_direction == "SELL" and news_sentiment == "bullish") else "Neutral"}
        
        ## Trading Signal Analysis
        - Technical Signals: {max(bullish_signals, bearish_signals)} {"Bullish" if bullish_signals > bearish_signals else "Bearish" if bearish_signals > bullish_signals else "Neutral"}
        - News Confirmation Bonus: +{news_bonus}%
        - Final Win Probability: {win_probability}%
        - Multi-Factor Direction: {final_direction}
        
        Based on this COMPREHENSIVE ANALYSIS combining TECHNICAL INDICATORS and SPECIALIZED FOREX NEWS, generate a profit-maximizing trading signal for {currency_pair}.
        
        ## REQUIRED NEWS-ENHANCED PROFIT TABLE FORMAT:
        
        The table must include these columns:
        - Pair: {currency_pair}
        - Current Price: {current_price:.5f}
        - Signal: {final_direction}
        - Entry Price: Specific price level close to current price
        - Stop Loss: Price level and exact pip distance
        - TP1: First take profit (1:1 risk:reward)
        - TP2: Second take profit (1:2 risk:reward)
        - TP3: Third take profit (1:3 risk:reward)
        - Win Probability: {win_probability}%
        - News Sentiment: {news_sentiment.title()}
        - News Impact: {"Confirming" if (final_direction == "BUY" and news_sentiment == "bullish") or (final_direction == "SELL" and news_sentiment == "bearish") else "Contradicting" if (final_direction == "BUY" and news_sentiment == "bearish") or (final_direction == "SELL" and news_sentiment == "bullish") else "Neutral"}
        
        After the table, provide:
        
        ## Technical & News Analysis:
        - Explain how technical analysis and news complement each other in this setup
        - Detail why this specific combination has exceptional profit potential
        - Address how news sentiment supports the technical direction
        
        ## Entry & Exit Strategy:
        - Provide specific entry techniques considering both technicals and news timing
        - Detail exact exit points based on technical levels and potential news developments
        - Explain how to adjust the trade if breaking news emerges
        
        ## News-Aware Risk Management:
        - Explain how to manage position size considering news volatility
        - Provide guidance on stop loss placement to account for news-driven price spikes
        - Detail how to protect profits if news sentiment shifts
        
        IMPORTANT: Present the TABLE first in your response, followed by detailed analysis.
        Make sure the table is visually clear and easy to read.
        Focus EXCLUSIVELY on setups where TECHNICAL ANALYSIS and NEWS SENTIMENT are ALIGNED for maximum profit.
        If technical analysis and news contradict each other, explicitly recommend NO TRADE.
        """
        
        return forex_news_profit_agent.print_response(prompt, stream=True)
        
    except Exception as e:
        error_message = f"Error generating news-enhanced profit signal: {str(e)}"
        print(error_message)
        return error_message

# Run the program
if __name__ == "__main__":
    print(f"💰 15-MINUTE FOREX NEWS-ENHANCED PROFIT MAXIMIZER - {CURRENT_DATE_TIME}")
    print(f"👤 User: {CURRENT_USER}")
    print(f"⏱️ Timeframe: 15M (News-Enhanced Analysis)")
    print(f"💼 Account Balance: ${ACCOUNT_BALANCE} | Risk: {RISK_PER_TRADE}%")
    print("="*80)
    
    print("Specialized Forex News Sources:")
    for i, source in enumerate(TOP_FOREX_NEWS_SOURCES, 1):
        print(f"{i}. {source['name']} ({source['type']}, Reliability: {source['reliability']}%)")
    print("="*80)
    
    # Generate news-enhanced profit signal
    generate_news_enhanced_signal(currency_pair="EURUSD", account_balance=ACCOUNT_BALANCE, risk_percentage=RISK_PER_TRADE)