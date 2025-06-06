
import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import traceback
import concurrent.futures
import threading
from functools import partial
import matplotlib.pyplot as plt
import io
import base64

# Import the full list of IDX tickers
from idx_all_tickers import IDX_ALL_TICKERS_YF # Assuming idx_all_tickers.py is in the same directory

# Constants
MAX_TICKERS = 950
# Relaxed Default Fundamental Filters
DEFAULT_MIN_NI = 0.1  # Default minimum Net Income in trillion IDR (Relaxed from 1.0)
DEFAULT_MAX_PE = 30.0  # Default maximum P/E ratio (Relaxed from 15.0)
DEFAULT_MAX_PB = 2.5  # Default maximum P/B ratio (Relaxed from 1.5)
DEFAULT_MIN_GROWTH = -20.0 # Default minimum YoY growth (Relaxed from 0.0)

RSI_PERIOD = 25  # Period for RSI calculation
OVERSOLD_THRESHOLD = 30
OVERBOUGHT_THRESHOLD = 70
MAX_WORKERS = 10
BATCH_SIZE = 50

# --- Helper function for Wilder's RSI ---
def calculate_rsi_wilder(prices, period=RSI_PERIOD, ticker="N/A"):
    '''Calculate RSI using Wilder's smoothing method.'''
    print(f"[{ticker}] Calculating RSI Wilder: Input prices length = {len(prices)}")
    delta = prices.diff()
    delta = delta[1:]
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Check if enough data for initial SMA
    if len(gain) < period:
        print(f"[{ticker}] RSI Wilder Error: Not enough gain/loss data (need {period}, got {len(gain)}).")
        return pd.Series(dtype=float)

    # Calculate initial average gain and loss using SMA
    try:
        avg_gain_series = gain.rolling(window=period, min_periods=period).mean()
        avg_loss_series = loss.rolling(window=period, min_periods=period).mean()
        
        first_valid_index = period - 1
        while first_valid_index < len(avg_gain_series) and pd.isna(avg_gain_series.iloc[first_valid_index]):
            first_valid_index += 1
            
        if first_valid_index >= len(avg_gain_series):
             print(f"[{ticker}] RSI Wilder Error: Not enough valid data points after rolling SMA.")
             return pd.Series(dtype=float)
             
        avg_gain = avg_gain_series.iloc[first_valid_index]
        avg_loss = avg_loss_series.iloc[first_valid_index]
        
        if pd.isna(avg_gain) or pd.isna(avg_loss):
             print(f"[{ticker}] RSI Wilder Error: Initial SMA calculation resulted in NaN (AvgGain: {avg_gain}, AvgLoss: {avg_loss}).")
             return pd.Series(dtype=float)

        print(f"[{ticker}] RSI Wilder Initial AvgGain: {avg_gain:.4f}, AvgLoss: {avg_loss:.4f}")

    except Exception as e:
        print(f"[{ticker}] RSI Wilder Error during initial SMA: {e}")
        return pd.Series(dtype=float)

    # Initialize arrays for Wilder's averages
    wilder_avg_gain = np.array([avg_gain])
    wilder_avg_loss = np.array([avg_loss])

    # Calculate subsequent averages using Wilder's smoothing
    start_calc_index = first_valid_index + 1
    for i in range(start_calc_index, len(gain)):
        current_gain = gain.iloc[i] if not pd.isna(gain.iloc[i]) else 0
        current_loss = loss.iloc[i] if not pd.isna(loss.iloc[i]) else 0
        
        avg_gain = (wilder_avg_gain[-1] * (period - 1) + current_gain) / period
        avg_loss = (wilder_avg_loss[-1] * (period - 1) + current_loss) / period
        wilder_avg_gain = np.append(wilder_avg_gain, avg_gain)
        wilder_avg_loss = np.append(wilder_avg_loss, avg_loss)

    # Handle division by zero for avg_loss
    rs = np.divide(wilder_avg_gain, wilder_avg_loss, out=np.full_like(wilder_avg_gain, np.inf), where=wilder_avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))

    # Return the full RSI series aligned with the original price index
    rsi_index = prices.index[start_calc_index + 1 : start_calc_index + 1 + len(rsi)]
    if len(rsi) != len(rsi_index):
         print(f"[{ticker}] RSI Wilder Warning: Index alignment mismatch (RSI len {len(rsi)}, Index len {len(rsi_index)}). Returning values only.")
         rsi_series = pd.Series(rsi)
    else:
        rsi_series = pd.Series(rsi, index=rsi_index)
        
    print(f"[{ticker}] RSI Wilder Calculation successful. Output series length: {len(rsi_series)}")
    # print(f"[{ticker}] RSI Wilder Output Head:\n{rsi_series.head()}") # Optional: Log head/tail if needed
    # print(f"[{ticker}] RSI Wilder Output Tail:\n{rsi_series.tail()}")
    return rsi_series


# Cache technical data for 5 minutes (300 seconds)
@st.cache_data(ttl=300)
def get_rsi_yfinance(ticker):
    '''
    Calculate RSI for a given ticker using Wilder's smoothing (yfinance version with enhanced logging).
    Returns: (rsi_value, signal, rsi_history) or None if data unavailable or calculation fails
    '''
    print(f"[{ticker}] --- Starting get_rsi_yfinance --- ")
    try:
        print(f"[{ticker}] Initializing yf.Ticker...")
        stock = yf.Ticker(ticker)
        if not stock:
             print(f"[{ticker}] yf.Ticker initialization failed.")
             st.session_state.setdefault("errors", {})
             st.session_state.errors[ticker] = f"RSI Error: yf.Ticker initialization failed."
             return None
             
        # Fetch enough history for RSI calculation
        fetch_period = "6mo"
        fetch_interval = "1d"
        print(f"[{ticker}] Fetching history: period='{fetch_period}', interval='{fetch_interval}'...")
        hist = stock.history(period=fetch_period, interval=fetch_interval)
        print(f"[{ticker}] History fetched. Shape: {hist.shape}")

        if hist.empty:
            print(f"[{ticker}] RSI Error: Fetched history is empty.")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = f"RSI Error: Fetched history is empty."
            return None
            
        if "Close" not in hist.columns:
            print(f"[{ticker}] RSI Error: 'Close' column not found in fetched history.")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = f"RSI Error: 'Close' column not found in history."
            return None
            
        close_prices = hist["Close"].dropna() # Drop NaNs from close prices
        print(f"[{ticker}] Close prices extracted. Length after dropna: {len(close_prices)}")
        # print(f"[{ticker}] Close prices tail:\n{close_prices.tail()}") # Optional: Log tail

        if len(close_prices) < RSI_PERIOD + 1:
            print(f"[{ticker}] RSI Error: Not enough valid historical data points (need {RSI_PERIOD + 1}, got {len(close_prices)}).")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = f"RSI Error: Not enough valid historical data (need {RSI_PERIOD + 1}, got {len(close_prices)})."
            return None

        # Calculate RSI using Wilder's method
        print(f"[{ticker}] Calling calculate_rsi_wilder...")
        rsi_series = calculate_rsi_wilder(close_prices, period=RSI_PERIOD, ticker=ticker)

        if rsi_series.empty or rsi_series.isna().all():
            print(f"[{ticker}] RSI Error: Wilder calculation resulted in empty or all-NaN series.")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = f"RSI Error: Wilder calculation resulted in empty or NaN series."
            return None

        # Get the latest RSI value
        try:
            latest_rsi = rsi_series.iloc[-1]
            print(f"[{ticker}] Latest RSI value extracted: {latest_rsi:.2f}")
        except IndexError:
            print(f"[{ticker}] RSI Error: Could not get latest RSI value from series (IndexError). Series length: {len(rsi_series)}")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = f"RSI Error: Could not get latest RSI value (IndexError)."
            return None

        # Check if latest RSI is valid
        if pd.isna(latest_rsi):
             print(f"[{ticker}] RSI Error: Latest RSI value is NaN.")
             st.session_state.setdefault("errors", {})
             st.session_state.errors[ticker] = f"RSI Error: Latest RSI value is NaN."
             return None

        # Determine signal based on RSI value
        if latest_rsi < OVERSOLD_THRESHOLD:
            signal = "Oversold"
        elif latest_rsi > OVERBOUGHT_THRESHOLD:
            signal = "Overbought"
        else:
            signal = "Neutral"
        print(f"[{ticker}] RSI Signal determined: {signal}")

        # Return latest RSI, signal, and the last RSI_PERIOD values for the chart
        rsi_history = rsi_series.dropna().tail(RSI_PERIOD).values
        if len(rsi_history) == 0:
             print(f"[{ticker}] RSI Error: No valid RSI history values found for chart after dropna/tail.")
             st.session_state.setdefault("errors", {})
             st.session_state.errors[ticker] = f"RSI Error: No valid RSI history values found for chart."
             return None # Cannot create chart without history
        print(f"[{ticker}] RSI History extracted for chart. Length: {len(rsi_history)}")

        print(f"[{ticker}] --- Successfully completed get_rsi_yfinance --- ")
        return (latest_rsi, signal, rsi_history)

    except Exception as e:
        error_msg = f"RSI yfinance Error: {e}\n{traceback.format_exc()}"
        print(f"[{ticker}] !!! EXCEPTION in get_rsi_yfinance: {error_msg}")
        st.session_state.setdefault("errors", {})
        st.session_state.errors[ticker] = error_msg
        return None

# Cache fundamentals data for 24 hours (86400 seconds) - KEEP USING YFINANCE
@st.cache_data(ttl=86400)
def get_fundamentals(ticker):
    '''
    Retrieve fundamental financial data for a given ticker using yfinance.
    Returns: (net_income, prev_net_income, pe_ratio, pb_ratio) or None if essential data unavailable/invalid
    '''
    print(f"[{ticker}] --- Starting get_fundamentals --- ")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials
        # balance_sheet = stock.balance_sheet # Not currently used
        print(f"[{ticker}] yfinance info and financials fetched.")

        # Initialize metrics
        net_income, prev_net_income, pe_ratio, pb_ratio = None, 0, None, None # Default prev_ni to 0

        # --- Net Income ---
        if not financials.empty and "Net Income" in financials.index:
            ni_series = financials.loc["Net Income"]
            if not ni_series.empty:
                try:
                    # Handle potential MultiIndex or different structures
                    if isinstance(ni_series, pd.Series):
                        net_income = ni_series.iloc[0] / 1e12 # Current NI in Trillion IDR
                        if len(ni_series) > 1:
                            prev_net_income = ni_series.iloc[1] / 1e12 # Previous NI in Trillion IDR
                        else:
                            print(f"[{ticker}] Fund. Warning: Previous Net Income missing (Series format). Growth may be inaccurate.")
                            st.session_state.setdefault("warnings", {})
                            st.session_state.warnings[ticker] = f"Fund. Warning: Previous Net Income missing, growth calculation may be inaccurate."
                    else: # Handle DataFrame case if structure changes
                         net_income = ni_series.iloc[0, 0] / 1e12
                         if ni_series.shape[1] > 1:
                             prev_net_income = ni_series.iloc[0, 1] / 1e12
                         else:
                             print(f"[{ticker}] Fund. Warning: Previous Net Income missing (DataFrame format). Growth may be inaccurate.")
                             st.session_state.setdefault("warnings", {})
                             st.session_state.warnings[ticker] = f"Fund. Warning: Previous Net Income missing (DataFrame format)."
                    print(f"[{ticker}] Net Income extracted: Current={net_income:.3f}T, Previous={prev_net_income:.3f}T")

                except (IndexError, TypeError, ValueError, KeyError) as e:
                     error_msg = f"Fund. Error extracting Net Income: {e}"
                     print(f"[{ticker}] {error_msg}")
                     st.session_state.setdefault("errors", {})
                     st.session_state.errors[ticker] = error_msg
                     net_income = None # Mark as invalid if extraction failed
            else:
                print(f"[{ticker}] Fund. Warning: 'Net Income' series is empty in financials.")
                st.session_state.setdefault("warnings", {})
                st.session_state.warnings[ticker] = f"Fund. Warning: 'Net Income' series is empty in financials."
        else:
            print(f"[{ticker}] Fund. Warning: 'Net Income' not found or financials empty.")
            st.session_state.setdefault("warnings", {})
            st.session_state.warnings[ticker] = f"Fund. Warning: 'Net Income' not found or empty in yfinance financials."

        # --- P/E Ratio ---
        pe_ratio = info.get("trailingPE", None)
        if pe_ratio is None or not isinstance(pe_ratio, (int, float)) or np.isnan(pe_ratio):
            warning_msg = f"Fund. Warning: Trailing P/E missing or invalid ({pe_ratio}). Will not filter by P/E."
            print(f"[{ticker}] {warning_msg}")
            st.session_state.setdefault("warnings", {})
            st.session_state.warnings[ticker] = warning_msg
            pe_ratio = None # Ensure it's None if invalid
        else:
            print(f"[{ticker}] P/E Ratio extracted: {pe_ratio:.2f}")

        # --- P/B Ratio ---
        pb_ratio = info.get("priceToBook", None)
        if pb_ratio is None or not isinstance(pb_ratio, (int, float)) or np.isnan(pb_ratio):
            warning_msg = f"Fund. Warning: P/B ratio missing or invalid ({pb_ratio}). Will not filter by P/B."
            print(f"[{ticker}] {warning_msg}")
            st.session_state.setdefault("warnings", {})
            st.session_state.warnings[ticker] = warning_msg
            pb_ratio = None # Ensure it's None if invalid
        else:
             print(f"[{ticker}] P/B Ratio extracted: {pb_ratio:.2f}")

        # --- Check Essential Data for Filtering ---
        if net_income is None or not isinstance(net_income, (int, float)) or np.isnan(net_income):
            error_msg = f"Fund. Error: Net Income is missing or invalid ({net_income}). Cannot apply fundamental filters."
            print(f"[{ticker}] {error_msg}")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = error_msg
            return None # Cannot proceed without Net Income

        # Return collected data (prev_net_income defaults to 0 if missing)
        print(f"[{ticker}] --- Successfully completed get_fundamentals --- ")
        return (net_income, prev_net_income, pe_ratio, pb_ratio)

    except Exception as e:
        error_msg = f"Fund. yfinance Error: {e}\n{traceback.format_exc()}"
        print(f"[{ticker}] !!! EXCEPTION in get_fundamentals: {error_msg}")
        st.session_state.setdefault("errors", {})
        st.session_state.errors[ticker] = error_msg
        return None


def process_ticker_technical_first(ticker, rsi_min, rsi_max, show_oversold, show_overbought, show_neutral):
    '''
    Process a single ticker with technical filters first (yfinance version).
    Returns: [ticker_symbol, rsi, signal, rsi_history] or None if not matching criteria
    '''
    print(f"[{ticker}] Processing technical filters...")
    try:
        # Use the yfinance-based function with enhanced logging
        rsi_data = get_rsi_yfinance(ticker)
        if not rsi_data:
            # Error/Warning logged in get_rsi_yfinance
            print(f"[{ticker}] Skipping technical processing due to RSI fetch/calc failure.")
            return None

        rsi, signal, rsi_history = rsi_data

        # Apply RSI range filter
        if (rsi_min > 0 and rsi < rsi_min) or (rsi_max < 100 and rsi > rsi_max):
            print(f"[{ticker}] Filtering out: RSI {rsi:.1f} outside range {rsi_min}-{rsi_max}.")
            st.session_state.setdefault("filtered_out_technical", {})
            st.session_state.filtered_out_technical[ticker] = f"Filtered out: RSI {rsi:.1f} outside range {rsi_min}-{rsi_max}."
            return None

        # Apply RSI signal filters
        if (signal == "Oversold" and not show_oversold) or \
           (signal == "Overbought" and not show_overbought) or \
           (signal == "Neutral" and not show_neutral):
            print(f"[{ticker}] Filtering out: Signal '{signal}' not selected.")
            st.session_state.setdefault("filtered_out_technical", {})
            st.session_state.filtered_out_technical[ticker] = f"Filtered out: Signal '{signal}' not selected."
            return None

        ticker_symbol = ticker.replace(".JK", "")
        print(f"[{ticker}] Passed technical filters: {ticker_symbol} (RSI: {rsi:.1f}, Signal: {signal})")
        return [ticker_symbol, rsi, signal, rsi_history]

    except Exception as e:
        error_msg = f"Tech. Process Error: {e}\n{traceback.format_exc()}"
        print(f"[{ticker}] !!! EXCEPTION in process_ticker_technical_first: {error_msg}")
        st.session_state.setdefault("errors", {})
        st.session_state.errors[ticker] = error_msg
        return None

def apply_fundamental_filters(technical_results, min_ni, max_pe, max_pb, min_growth):
    '''
    Apply fundamental filters to stocks that passed technical screening.
    Returns: List of stocks with both technical and fundamental data
    '''
    final_results = []
    print(f"Applying fundamental filters to {len(technical_results)} stocks...")

    # Use threading for fundamental data fetching (as it uses yfinance)
    fund_results = {}
    def fetch_fund(ticker_symbol):
        ticker = f"{ticker_symbol}.JK"
        print(f"[{ticker}] Submitting fundamental fetch job...")
        fund_data = get_fundamentals(ticker)
        if fund_data:
            fund_results[ticker_symbol] = fund_data
            print(f"[{ticker}] Fundamental data fetch successful.")
        else:
            print(f"[{ticker}] Fundamental data fetch failed or returned None.")
            # Error logged in get_fundamentals

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(fetch_fund, result[0]) for result in technical_results]
        # Wait for all fundamental fetches to complete (or timeout)
        print(f"Waiting for {len(futures)} fundamental fetch jobs to complete...")
        concurrent.futures.wait(futures) # Removed timeout to ensure all complete
        print(f"Fundamental fetch jobs complete.")

    print(f"Fundamental data available for {len(fund_results)} stocks after fetch.")

    for result in technical_results:
        ticker_symbol, rsi, signal, rsi_history = result
        ticker = f"{ticker_symbol}.JK"

        if ticker_symbol not in fund_results:
            print(f"[{ticker}] Skipping fundamental filtering: Data not found in results dict.")
            continue # Skip if fundamental data fetch failed or timed out

        print(f"[{ticker}] Applying fundamental filters...")
        try:
            fund_data = fund_results[ticker_symbol]
            ni, prev_ni, pe, pb = fund_data

            # Calculate growth (handle division by zero or invalid prev_ni)
            growth = np.nan # Default to NaN
            if prev_ni is not None and isinstance(prev_ni, (int, float)) and not np.isnan(prev_ni):
                 if ni is not None and isinstance(ni, (int, float)) and not np.isnan(ni):
                     if prev_ni != 0:
                         growth = ((ni - prev_ni) / abs(prev_ni) * 100)
                     elif ni > 0: # Growth from zero
                         growth = np.inf
                     elif ni < 0:
                         growth = -np.inf
                     else: # ni = 0, prev_ni = 0
                         growth = 0.0
                 else:
                     print(f"[{ticker}] Growth Calc Warning: Current NI is invalid ({ni}).")
            else:
                 print(f"[{ticker}] Growth Calc Warning: Previous NI is invalid ({prev_ni}).")

            print(f"[{ticker}] Calculated Growth: {growth}")

            # Apply fundamental filters, logging reasons for exclusion
            reason = None
            if ni < min_ni:
                reason = f"NI {ni:.2f}T < {min_ni:.1f}T"
            # Only filter by PE if pe is valid and max_pe is restrictive
            elif pe is not None and max_pe < 50.0 and pe > max_pe:
                 reason = f"P/E {pe:.1f} > {max_pe:.1f}"
            # Only filter by PB if pb is valid and max_pb is restrictive
            elif pb is not None and max_pb < 5.0 and pb > max_pb:
                 reason = f"P/B {pb:.1f} > {max_pb:.1f}"
            # Only filter by growth if growth is valid and min_growth is restrictive
            elif not pd.isna(growth) and min_growth > -100.0 and growth < min_growth:
                 reason = f"Growth {growth:.1f}% < {min_growth:.1f}%"
            elif pd.isna(growth) and min_growth > -100.0:
                 reason = f"Growth calculation failed (NaN)"

            if reason:
                st.session_state.setdefault("filtered_out_fundamental", {})
                st.session_state.filtered_out_fundamental[ticker] = f"Filtered out: {reason}"
                print(f"[{ticker}] Filtering out (Fundamental): {reason}")
                continue # Skip this stock

            # Add to final results
            if growth == np.inf: growth_display = "+Inf"
            elif growth == -np.inf: growth_display = "-Inf"
            elif pd.isna(growth): growth_display = "N/A"
            else: growth_display = f"{growth:.1f}"

            final_results.append([
                ticker_symbol,
                f"{ni:.2f}", # Format NI
                growth_display, # Use display string for growth
                f"{pe:.1f}" if pe is not None else "N/A", # Format PE or N/A
                f"{pb:.1f}" if pb is not None else "N/A", # Format PB or N/A
                f"{rsi:.1f}", # Format RSI
                signal,
                rsi_history, # Keep history for charts
                growth # Keep original growth for potential sorting later if needed
            ])
            print(f"[{ticker}] Passed fundamental filters.")

        except Exception as e:
            error_msg = f"Fund. Apply Error: {e}\n{traceback.format_exc()}"
            print(f"[{ticker}] !!! EXCEPTION applying fundamental filters: {error_msg}")
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = error_msg

    print(f"Fundamental filtering complete. {len(final_results)} stocks passed.")
    return final_results


@st.cache_data(ttl=300)
def create_rsi_chart_image(rsi_values, current_rsi, ticker="N/A"):
    '''Create a matplotlib chart for RSI values and return as image bytes'''
    print(f"[{ticker}] Creating RSI chart image...")
    if isinstance(rsi_values, list):
        rsi_values = np.array(rsi_values)

    if rsi_values is None or len(rsi_values) == 0:
        print(f"[{ticker}] Chart Error: No RSI data provided.")
        fig, ax = plt.subplots(figsize=(3, 1.5))
        ax.text(0.5, 0.5, "No RSI Data", ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf

    try:
        fig, ax = plt.subplots(figsize=(3, 1.5))
        x = range(len(rsi_values))
        ax.plot(x, rsi_values, color='blue', linewidth=1.5)
        ax.axhline(y=OVERBOUGHT_THRESHOLD, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=OVERSOLD_THRESHOLD, color='green', linestyle='--', alpha=0.7, linewidth=1)
        ax.fill_between(x, OVERBOUGHT_THRESHOLD, 100, color='red', alpha=0.1)
        ax.fill_between(x, 0, OVERSOLD_THRESHOLD, color='green', alpha=0.1)
        ax.set_ylim(0, 100)

        num_ticks = min(5, len(rsi_values))
        tick_indices = np.linspace(0, len(rsi_values) - 1, num_ticks, dtype=int)
        tick_labels = [f"D-{len(rsi_values)-1-i}" for i in tick_indices]
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=30, ha='right')

        ax.set_yticks([0, OVERSOLD_THRESHOLD, 50, OVERBOUGHT_THRESHOLD, 100])
        ax.set_yticklabels(['0', str(OVERSOLD_THRESHOLD), '50', str(OVERBOUGHT_THRESHOLD), '100'], fontsize=8)

        text_x_pos = len(rsi_values) - 1
        text_y_pos = current_rsi + 5 if current_rsi < 95 else current_rsi - 5
        ax.text(text_x_pos, text_y_pos, f'{current_rsi:.1f}', verticalalignment='center', horizontalalignment='right', fontsize=9, color='black', fontweight='bold')
        ax.scatter(len(rsi_values)-1, current_rsi, color='blue', s=30, zorder=5)

        ax.set_title(f"RSI({RSI_PERIOD}) Chart", fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout(pad=0.2)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        print(f"[{ticker}] RSI chart image created successfully.")
        return buf
    except Exception as e:
        error_msg = f"Chart Creation Error: {e}"
        print(f"[{ticker}] !!! EXCEPTION creating chart: {error_msg}")
        st.session_state.setdefault("errors", {})
        st.session_state.errors[f"{ticker}_chart"] = error_msg
        # Return a placeholder image or re-raise?
        fig, ax = plt.subplots(figsize=(3, 1.5))
        ax.text(0.5, 0.5, "Chart Error", ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf

def process_batch_technical_first(batch_tickers, rsi_min, rsi_max, show_oversold, show_overbought, show_neutral):
    '''Process a batch of tickers with technical filters first (yfinance version).'''
    results = []
    print(f"--- Processing technical batch of {len(batch_tickers)} tickers ({batch_tickers[0]}...{batch_tickers[-1]}) ---")
    process_func = partial(
        process_ticker_technical_first,
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        show_oversold=show_oversold,
        show_overbought=show_overbought,
        show_neutral=show_neutral
    )
    # Using ThreadPoolExecutor for I/O bound yfinance calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {executor.submit(process_func, ticker): ticker for ticker in batch_tickers}
        print(f"Submitted {len(future_to_ticker)} technical jobs to executor.")
        processed_count = 0
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            processed_count += 1
            print(f"({processed_count}/{len(future_to_ticker)}) Future completed for {ticker}")
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    print(f"Result added for {ticker}")
            except Exception as exc:
                 error_msg = f'Tech. Batch Error processing future for {ticker}: {exc}'
                 print(f"!!! EXCEPTION {error_msg}")
                 st.session_state.setdefault("errors", {})
                 st.session_state.errors[ticker] = error_msg

    print(f"--- Technical batch processing complete. {len(results)} passed. ---")
    return results


def main():
    st.set_page_config(
        page_title="IDX Stock Screener V4 (Debug)", # Updated Title
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS (minor adjustments if needed)
    st.markdown('''
    <style>
    /* Make header sticky */
    .stDataFrame th {
        position: sticky;
        top: 0;
        background: white; /* Match background */
        z-index: 1;
    }
    /* Add border to expanders */
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .stSidebar {
            width: 100%;
        }
        .stDataFrame {
            width: 100%;
            overflow-x: auto;
        }
    }
    .stProgress > div > div {
        height: 10px;
    }
    .stButton > button {
        width: 100%;
    }
    </style>
    ''', unsafe_allow_html=True)

    # App header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("IDX Stock Screener V4 (Debug)") # Updated Title
        st.markdown(f"Screening **{len(IDX_ALL_TICKERS_YF)}** Indonesian stocks (Technical [yfinance] first, then Fundamental [yfinance])")
    with col2:
        st.metric("Total IDX Stocks", f"{len(IDX_ALL_TICKERS_YF)}")

    # Initialize session state
    if "errors" not in st.session_state: st.session_state.errors = {}
    if "warnings" not in st.session_state: st.session_state.warnings = {} # Added warnings log
    if "filtered_out_technical" not in st.session_state: st.session_state.filtered_out_technical = {}
    if "filtered_out_fundamental" not in st.session_state: st.session_state.filtered_out_fundamental = {}
    if "last_refresh" not in st.session_state: st.session_state.last_refresh = None
    if "results_cache" not in st.session_state: st.session_state.results_cache = None
    if "filter_settings" not in st.session_state:
        st.session_state.filter_settings = {
            "rsi_min": 0, "rsi_max": 100,
            "show_oversold": True, "show_overbought": True, "show_neutral": True,
            "min_ni": DEFAULT_MIN_NI, # Use updated default
            "max_pe": DEFAULT_MAX_PE, # Use updated default
            "max_pb": DEFAULT_MAX_PB, # Use updated default
            "min_growth": DEFAULT_MIN_GROWTH # Use updated default
        }

    # Sidebar filters
    with st.sidebar:
        st.header("Screening Filters")
        tab1, tab2, tab3, tab4 = st.tabs(["Technical", "Fundamental", "Performance", "Settings"])

        with tab1:
            st.subheader("Technical Filters (First Pass - yfinance)")
            st.caption(f"RSI Period: {RSI_PERIOD} days (Wilder's Smoothing)")
            rsi_range = st.slider("RSI Range", 0, 100, (st.session_state.filter_settings["rsi_min"], st.session_state.filter_settings["rsi_max"]), help="Filter stocks by RSI value range.")
            rsi_min, rsi_max = rsi_range
            show_oversold = st.checkbox("Show Oversold (RSI < 30)", st.session_state.filter_settings["show_oversold"])
            show_overbought = st.checkbox("Show Overbought (RSI > 70)", st.session_state.filter_settings["show_overbought"])
            show_neutral = st.checkbox("Show Neutral (30 <= RSI <= 70)", st.session_state.filter_settings["show_neutral"])

        with tab2:
            st.subheader("Fundamental Filters (Second Pass - yfinance)")
            min_ni = st.slider("Minimum Net Income (Trillion IDR)", 0.0, 10.0, st.session_state.filter_settings["min_ni"], 0.1, help="Minimum Net Income (most recent year). Set to 0 to disable.")
            max_pe = st.slider("Maximum P/E Ratio", 0.0, 50.0, st.session_state.filter_settings["max_pe"], 0.5, help="Maximum Price-to-Earnings ratio. Set to 50 to effectively disable.")
            max_pb = st.slider("Maximum P/B Ratio", 0.0, 5.0, st.session_state.filter_settings["max_pb"], 0.1, help="Maximum Price-to-Book ratio. Set to 5 to effectively disable.")
            min_growth = st.slider("Minimum YoY Growth (%)", -100.0, 100.0, st.session_state.filter_settings["min_growth"], 1.0, help="Minimum Year-over-Year Net Income growth. Set to -100 to disable.")

        with tab3:
            st.subheader("Performance Settings")
            batch_size = st.slider("Batch Size", 10, 100, BATCH_SIZE, 10, help="Tickers processed per batch.")
            max_workers = st.slider("Max Concurrent Workers", 1, 20, MAX_WORKERS, 1, help="Parallel threads for data fetching.")

        with tab4:
            st.subheader("Refresh & Debug")
            refresh = st.toggle("Auto-refresh", False, help="Automatically refresh data periodically.")
            refresh_interval = st.slider("Refresh Interval (minutes)", 1, 60, 10, help="Time between refreshes.") if refresh else 0
            show_logs = st.checkbox("Show Debug Logs", True, help="Display detailed error and warning logs.") # Default to True for debugging

        # Update session state with current filter settings if they changed
        current_filters = {
            "rsi_min": rsi_min, "rsi_max": rsi_max,
            "show_oversold": show_oversold, "show_overbought": show_overbought, "show_neutral": show_neutral,
            "min_ni": min_ni, "max_pe": max_pe, "max_pb": max_pb, "min_growth": min_growth
        }
        filters_changed = (current_filters != st.session_state.filter_settings)
        if filters_changed:
            print("Filters changed, clearing cache.")
            st.session_state.filter_settings = current_filters
            st.session_state.results_cache = None # Clear cache if filters change

    # Main content area
    main_tab1, main_tab2 = st.tabs(["Screener Results", "About & Logs"])

    with main_tab1:
        # Progress indicators
        progress_col1, progress_col2 = st.columns([3, 1])
        with progress_col1:
            progress_bar = st.progress(0, text="Status: Idle")
        with progress_col2:
            status_text = st.empty() # Placeholder for dynamic status text

        # Placeholders for results
        technical_results_placeholder = st.empty()
        final_results_placeholder = st.empty()

    with main_tab2:
        st.subheader("About IDX Stock Screener V4 (Debug)")
        st.markdown(f'''
        This application screens all **{len(IDX_ALL_TICKERS_YF)}** stocks listed on the Indonesia Stock Exchange (IDX).
        - **Technical Screening (First Pass):** Uses the **yfinance library** to fetch historical data and filters by RSI({RSI_PERIOD}) value and signal (Oversold < {OVERSOLD_THRESHOLD}, Overbought > {OVERBOUGHT_THRESHOLD}). Uses Wilder's Smoothing.
        - **Fundamental Screening (Second Pass):** Uses the **yfinance library** to fetch fundamental data and applies filters for Net Income, P/E Ratio, P/B Ratio, and YoY Growth to the stocks that passed the technical screen.
        - **Data:** Technical data cached for 5 mins, Fundamental data for 24 hours.
        - **Improvements (V4 Debug):** Reverted to yfinance for technical data. Added extensive logging to diagnose potential issues.
        ''')

        st.subheader("Current Filter Settings")
        filter_summary = st.session_state.filter_settings.copy()
        filter_summary["RSI Range"] = f"{filter_summary.pop('rsi_min')} - {filter_summary.pop('rsi_max')}"
        ordered_summary = {
            "RSI Range": filter_summary.pop("RSI Range"),
            "Show Oversold": filter_summary.pop("show_oversold"),
            "Show Overbought": filter_summary.pop("show_overbought"),
            "Show Neutral": filter_summary.pop("show_neutral"),
            "Min Net Income (T IDR)": filter_summary.pop("min_ni"),
            "Max P/E Ratio": filter_summary.pop("max_pe"),
            "Max P/B Ratio": filter_summary.pop("max_pb"),
            "Min Growth (%)": filter_summary.pop("min_growth")
        }
        st.dataframe(pd.Series(ordered_summary), use_container_width=True)

        # Always show logs in this debug version
        st.subheader("Debug Logs")
        with st.expander("Warnings (Data Issues)", expanded=False):
            if st.session_state.warnings: st.json(st.session_state.warnings)
            else: st.write("No warnings logged.")
        with st.expander("Errors (Failed Operations)", expanded=False):
             if st.session_state.errors: st.json(st.session_state.errors)
             else: st.write("No errors logged.")
        with st.expander("Filtered Out (Technical Stage)", expanded=False):
             if st.session_state.get("filtered_out_technical"): st.json(st.session_state.filtered_out_technical)
             else: st.write("No stocks logged as filtered out during technical screening.")
        with st.expander("Filtered Out (Fundamental Stage)", expanded=False):
             if st.session_state.filtered_out_fundamental: st.json(st.session_state.filtered_out_fundamental)
             else: st.write("No stocks logged as filtered out during fundamental screening.")


    # Check if refresh is needed
    current_time = datetime.now()
    force_refresh = st.sidebar.button("Run Screener Now", use_container_width=True)
    need_refresh = (
        force_refresh or
        st.session_state.results_cache is None or
        filters_changed or # Refresh if filters changed
        (refresh and refresh_interval > 0 and
         st.session_state.last_refresh is not None and
         (current_time - st.session_state.last_refresh).total_seconds() > refresh_interval * 60)
    )

    # --- Screening Logic ---
    def perform_screening():
        print("===== Starting Screening Process ====")
        technical_results = []
        final_results_data = [] # Store raw data before formatting for display/CSV
        start_time = time.time()

        # Clear previous logs
        print("Clearing previous logs...")
        st.session_state.errors = {}
        st.session_state.warnings = {}
        st.session_state.filtered_out_technical = {}
        st.session_state.filtered_out_fundamental = {}

        # --- FIRST PASS: Technical Screening (Using yfinance) ---
        num_batches = (len(IDX_ALL_TICKERS_YF) + batch_size - 1) // batch_size
        status_text.text("Starting Technical Screening (yfinance)...")
        progress_bar.progress(0, text="Technical Screening (yfinance): Batch 1")
        print(f"Starting Technical Screening: {len(IDX_ALL_TICKERS_YF)} tickers, {num_batches} batches.")

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(IDX_ALL_TICKERS_YF))
            batch_tickers = IDX_ALL_TICKERS_YF[batch_start:batch_end]

            progress = (batch_idx + 1) / (num_batches * 2) # Technical is first half
            progress_text = f"Technical Screening (yfinance): Batch {batch_idx + 1}/{num_batches} ({len(batch_tickers)} tickers)"
            print(f"=== Processing Batch {batch_idx + 1}/{num_batches} ===")
            progress_bar.progress(progress, text=progress_text)
            status_text.text(progress_text)

            batch_tech_results = process_batch_technical_first(
                batch_tickers,
                st.session_state.filter_settings["rsi_min"],
                st.session_state.filter_settings["rsi_max"],
                st.session_state.filter_settings["show_oversold"],
                st.session_state.filter_settings["show_overbought"],
                st.session_state.filter_settings["show_neutral"]
            )
            technical_results.extend(batch_tech_results)
            print(f"Batch {batch_idx + 1} complete. Current total technical results: {len(technical_results)}")

        # Remove duplicates after technical screening (if any)
        seen_tickers_tech = set()
        unique_technical_results = []
        for result in technical_results:
            if result[0] not in seen_tickers_tech:
                unique_technical_results.append(result)
                seen_tickers_tech.add(result[0])
        technical_results = unique_technical_results
        tech_count = len(technical_results)

        print(f"Technical Screening Complete. Total unique stocks passed: {tech_count}")
        progress_bar.progress(0.5, text=f"Technical Screening Complete ({tech_count} passed)")
        status_text.text(f"Technical Screening Complete: Found {tech_count} stocks")

        # Display intermediate technical results (optional, can be large)
        with technical_results_placeholder.container():
            if technical_results:
                 st.subheader(f"Technical Screening Results ({tech_count} Stocks)")
                 tech_df = pd.DataFrame([(t[0], f"{t[1]:.1f}", t[2]) for t in technical_results], columns=["Ticker", "RSI", "Signal"])
                 st.dataframe(tech_df, height=300, use_container_width=True)
            else:
                 st.info("No stocks passed the technical screening.")
                 print("No stocks passed technical screening phase.")


        # --- SECOND PASS: Fundamental Screening (Using yfinance) ---
        if technical_results:
            print(f"Starting Fundamental Screening for {tech_count} stocks...")
            status_text.text(f"Starting Fundamental Screening (yfinance) for {tech_count} stocks...")
            progress_bar.progress(0.5, text="Fundamental Screening (yfinance)...")

            # Apply fundamental filters (returns formatted data + raw growth)
            final_results_data = apply_fundamental_filters(
                technical_results,
                st.session_state.filter_settings["min_ni"],
                st.session_state.filter_settings["max_pe"],
                st.session_state.filter_settings["max_pb"],
                st.session_state.filter_settings["min_growth"]
            )

            # Remove duplicates after fundamental screening (shouldn't happen)
            seen_tickers_final = set()
            unique_final_results = []
            for result in final_results_data:
                if result[0] not in seen_tickers_final:
                    unique_final_results.append(result)
                    seen_tickers_final.add(result[0])
            final_results_data = unique_final_results
            final_count = len(final_results_data)

            print(f"Fundamental Screening Complete. Total stocks passed all filters: {final_count}")
            progress_bar.progress(1.0, text=f"Screening Complete ({final_count} passed all filters)")
            status_text.text(f"Screening Complete: Found {final_count} stocks matching all criteria")
        else:
             # Skip fundamental if no technical results
             final_count = 0
             print("Skipping Fundamental Screening as no stocks passed technical.")
             progress_bar.progress(1.0, text="Screening Complete (0 passed technical)")
             status_text.text("Screening Complete: No stocks passed technical filters.")


        # --- Store Results and Performance ---
        elapsed_time = time.time() - start_time
        print(f"Screening took {elapsed_time:.2f} seconds.")
        st.session_state.last_refresh = current_time
        st.session_state.results_cache = {
            "technical_results": technical_results,
            "final_results_data": final_results_data,
            "technical_count": tech_count,
            "final_count": final_count,
            "elapsed_time": elapsed_time,
            "errors_count": len(st.session_state.errors),
            "warnings_count": len(st.session_state.warnings),
            "filtered_out_tech_count": len(st.session_state.filtered_out_technical),
            "filtered_out_fund_count": len(st.session_state.filtered_out_fundamental)
        }
        print(f"Results cached. Errors: {len(st.session_state.errors)}, Warnings: {len(st.session_state.warnings)}, Filtered Tech: {len(st.session_state.filtered_out_technical)}, Filtered Fund: {len(st.session_state.filtered_out_fundamental)}")
        print("===== Screening Process Finished ====")

    # --- Run Screening or Load from Cache ---
    if need_refresh:
        perform_screening()
    elif st.session_state.results_cache:
        # Load from cache
        status_text.text(f"Loaded cached results from {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
        progress_bar.progress(1.0, text="Screening Complete (Cached)")
        print("Loaded results from cache.")
    else:
        # Initial state or cache cleared
        status_text.text("Ready to run screener.")
        progress_bar.progress(0, text="Status: Idle")
        print("App loaded, ready to run screener.")


    # --- Display Final Results ---
    with final_results_placeholder.container():
        if st.session_state.results_cache and st.session_state.results_cache["final_count"] > 0:
            final_results_data = st.session_state.results_cache["final_results_data"]
            final_count = st.session_state.results_cache["final_count"]

            st.subheader(f"Final Results ({final_count} Stocks)")

            # Create DataFrame for display (excluding raw RSI history and raw growth)
            display_df = pd.DataFrame(
                 [res[:7] for res in final_results_data], # Take first 7 columns (Ticker to Signal)
                 columns=["Ticker", "NI(T)", "Growth(%)", "P/E", "P/B", "RSI", "Signal"]
            )

            # Add RSI Chart column (using base64 encoded images)
            rsi_charts_html = []
            print(f"Generating {len(final_results_data)} RSI chart images...")
            for i, result in enumerate(final_results_data):
                 ticker, _, _, _, _, rsi_str, _, rsi_history, _ = result # Unpack
                 try:
                     rsi_float = float(rsi_str) # Convert formatted RSI back to float for chart function
                     chart_img_bytes = create_rsi_chart_image(rsi_history, rsi_float, ticker=ticker)
                     img_base64 = base64.b64encode(chart_img_bytes.getvalue()).decode()
                     rsi_charts_html.append(f'<img src="data:image/png;base64,{img_base64}" alt="RSI Chart for {ticker}" style="max-height: 50px;">')
                 except Exception as e:
                     print(f"[{ticker}] Error generating chart image: {e}")
                     rsi_charts_html.append("Chart Error")
                     st.session_state.setdefault("errors", {})
                     st.session_state.errors[f"{ticker}_chart"] = f"Chart Error: {e}"
            print("RSI chart image generation complete.")

            display_df["RSI Chart"] = rsi_charts_html

            # Display DataFrame with HTML for charts
            st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)

            # Prepare DataFrame for CSV download (without chart HTML, use raw data)
            csv_df = pd.DataFrame(
                [(res[0], res[1], res[2], res[3], res[4], res[5], res[6]) for res in final_results_data], # Use formatted growth (index 2)
                columns=["Ticker", "NI(T IDR)", "Growth(%)", "P/E", "P/B", "RSI", "Signal"]
            )
            csv = csv_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='idx_screener_results_v4_debug.csv',
                mime='text/csv',
            )

        elif st.session_state.results_cache: # Results exist but count is 0
            st.warning("⚠️ No stocks found matching all criteria.")
            st.info("Try relaxing the filters or check the Debug Logs in the 'About & Logs' tab for potential data issues (e.g., yfinance errors, missing data).")
            print("Displaying 'No stocks found' message.")
        else: # No results cache yet (initial load)
             st.info("Click 'Run Screener Now' in the sidebar to start.")

# Standard Python entry point
if __name__ == "__main__":
    # --- Enhanced Error Catching Wrapper ---
    try:
        print("===== Application Start =====")
        main()
        print("===== Application main() finished normally =====")
    except Exception as e:
        print(f"!!! TOP LEVEL ERROR: An exception occurred while running main(): {e}")
        # Display error in Streamlit app
        st.set_page_config(page_title="App Error", layout="wide")
        st.title("🚨 Application Error")
        st.error("An unexpected error occurred while running the application. Please see the details below:")
        st.exception(e)
        # Also print traceback to logs for debugging
        print("--- TOP LEVEL TRACEBACK START ---")
        traceback.print_exc()
        print("--- TOP LEVEL TRACEBACK END ---")
        print("===== Application finished due to top-level error =====")


