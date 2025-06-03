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

# Import the full list of IDX tickers
from idx_all_tickers import IDX_ALL_TICKERS_YF

# Constants
MAX_TICKERS = 950  # Increased from 50 to handle all IDX stocks
DEFAULT_MIN_NI = 1.0  # Default minimum Net Income in trillion IDR
DEFAULT_MAX_PE = 15.0  # Default maximum P/E ratio
DEFAULT_MAX_PB = 1.5  # Default maximum P/B ratio
RSI_PERIOD = 25  # Period for RSI calculation
OVERSOLD_THRESHOLD = 30  # RSI threshold for oversold condition
OVERBOUGHT_THRESHOLD = 70  # RSI threshold for overbought condition
MAX_WORKERS = 10  # Maximum number of concurrent workers for parallel processing
BATCH_SIZE = 50  # Number of tickers to process in each batch

# Cache technical data for 5 minutes (300 seconds)
@st.cache_data(ttl=300)
def get_rsi(ticker):
    """
    Calculate RSI for a given ticker.
    Returns: (rsi_value, signal) or None if data unavailable
    """
    try:
        # Get historical data for 30 days (to calculate RSI25)
        end_date = datetime.now()
        stock = yf.Ticker(ticker)
        hist = stock.history(period="30d")
        
        if len(hist) < RSI_PERIOD + 1:
            return None
        
        # Calculate price changes
        delta = hist['Close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss over RSI_PERIOD
        avg_gain = gain.rolling(window=RSI_PERIOD).mean()
        avg_loss = loss.rolling(window=RSI_PERIOD).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Get the latest RSI value
        latest_rsi = rsi.iloc[-1]
        
        # Determine signal based on RSI value
        if latest_rsi < OVERSOLD_THRESHOLD:
            signal = "Oversold"
        elif latest_rsi > OVERBOUGHT_THRESHOLD:
            signal = "Overbought"
        else:
            signal = "Neutral"
        
        return (latest_rsi, signal)
    
    except Exception as e:
        # Log error for debugging
        st.session_state.setdefault('errors', {})
        st.session_state.errors[ticker] = str(e)
        return None

# Cache fundamentals data for 24 hours (86400 seconds)
@st.cache_data(ttl=86400)
def get_fundamentals(ticker):
    """
    Retrieve fundamental financial data for a given ticker.
    Returns: (net_income, prev_net_income, pe_ratio, pb_ratio) or None if data unavailable
    """
    try:
        # Get ticker info
        stock = yf.Ticker(ticker)
        
        # Get financial data with timeout to prevent hanging
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        info = stock.info
        
        # Check if we have the necessary data
        if financials.empty or balance_sheet.empty:
            return None
        
        # Extract Net Income (convert to Trillion IDR)
        if 'Net Income' in financials.index:
            net_income = financials.loc['Net Income'].iloc[0] / 1e12
            prev_net_income = financials.loc['Net Income'].iloc[1] / 1e12 if len(financials.columns) > 1 else 0
        else:
            return None
        
        # Extract P/E and P/B ratios
        pe_ratio = info.get('trailingPE', None)
        pb_ratio = info.get('priceToBook', None)
        
        # If P/E or P/B is missing, try to calculate them
        if pe_ratio is None or pb_ratio is None:
            market_cap = info.get('marketCap', None)
            if market_cap is None:
                return None
            
            if pe_ratio is None and net_income != 0:
                pe_ratio = market_cap / (net_income * 1e12)
            
            if pb_ratio is None and 'Total Stockholder Equity' in balance_sheet.index:
                total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                if total_equity != 0:
                    pb_ratio = market_cap / total_equity
        
        # Return None if any value is still None or not a number
        if None in (net_income, prev_net_income, pe_ratio, pb_ratio) or \
           any(not isinstance(x, (int, float)) for x in (net_income, prev_net_income, pe_ratio, pb_ratio)):
            return None
        
        return (net_income, prev_net_income, pe_ratio, pb_ratio)
    
    except Exception as e:
        # Log error for debugging
        st.session_state.setdefault('errors', {})
        st.session_state.errors[ticker] = str(e)
        return None

def process_ticker_technical_first(ticker, rsi_min, rsi_max, show_oversold, show_overbought, show_neutral):
    """
    Process a single ticker with technical filters first.
    Returns: [ticker_symbol, rsi, signal] or None if not matching criteria
    """
    try:
        # Get RSI data first
        rsi_data = get_rsi(ticker)
        if not rsi_data:
            return None
        
        rsi, signal = rsi_data
        
        # Apply RSI range filter if specified
        if (rsi_min > 0 and rsi < rsi_min) or (rsi_max < 100 and rsi > rsi_max):
            return None
        
        # Apply RSI signal filters
        if (signal == "Oversold" and not show_oversold) or \
           (signal == "Overbought" and not show_overbought) or \
           (signal == "Neutral" and not show_neutral):
            return None
        
        # Return result with technical data
        ticker_symbol = ticker.replace('.JK', '')
        return [ticker_symbol, rsi, signal]
    
    except Exception as e:
        # Log error for debugging
        st.session_state.setdefault('errors', {})
        st.session_state.errors[ticker] = str(e) + "\n" + traceback.format_exc()
        return None

def apply_fundamental_filters(technical_results, min_ni, max_pe, max_pb, min_growth):
    """
    Apply fundamental filters to stocks that passed technical screening.
    Returns: List of stocks with both technical and fundamental data
    """
    final_results = []
    
    for result in technical_results:
        ticker_symbol, rsi, signal = result
        ticker = f"{ticker_symbol}.JK"
        
        try:
            # Get fundamental data
            fund_data = get_fundamentals(ticker)
            if not fund_data:
                continue
            
            ni, prev_ni, pe, pb = fund_data
            
            # Calculate growth
            growth = ((ni - prev_ni) / abs(prev_ni) * 100) if prev_ni != 0 else 0
            
            # Apply fundamental filters
            if ni < min_ni or pe > max_pe or pb > max_pb or growth < min_growth:
                continue
            
            # Add to final results with both technical and fundamental data
            final_results.append([ticker_symbol, ni, growth, pe, pb, rsi, signal])
        
        except Exception as e:
            # Log error for debugging
            st.session_state.setdefault('errors', {})
            st.session_state.errors[ticker] = str(e) + "\n" + traceback.format_exc()
    
    return final_results

def highlight_oversold(df):
    """
    Apply conditional formatting to highlight oversold stocks.
    """
    # Create a copy of the dataframe
    styled_df = pd.DataFrame('', index=df.index, columns=df.columns)
    
    # Highlight oversold stocks
    mask = df['Signal'] == 'Oversold'
    styled_df.loc[mask, :] = 'background-color: #FFCCCB'
    
    # Highlight overbought stocks
    mask = df['Signal'] == 'Overbought'
    styled_df.loc[mask, :] = 'background-color: #CCFFCC'
    
    return styled_df

def format_progress_bar(value, min_val, max_val, color_scheme):
    """
    Create HTML for a progress bar with the given value.
    """
    # Normalize the value between 0 and 100
    normalized = max(0, min(100, (value - min_val) / (max_val - min_val) * 100))
    
    # Determine color based on the scheme
    if color_scheme == 'rsi':
        if value < OVERSOLD_THRESHOLD:
            color = 'red'
        elif value > OVERBOUGHT_THRESHOLD:
            color = 'green'
        else:
            color = 'blue'
    else:
        color = 'blue'
    
    # Create the HTML for the progress bar
    html = f"""
    <div style="width:100%; background-color:#f0f0f0; height:20px; border-radius:5px; margin-bottom:5px;">
        <div style="width:{normalized}%; background-color:{color}; height:20px; border-radius:5px;"></div>
    </div>
    <div style="text-align:center;">{value:.2f}</div>
    """
    return html

def process_batch_technical_first(batch_tickers, rsi_min, rsi_max, show_oversold, show_overbought, show_neutral):
    """
    Process a batch of tickers with technical filters first.
    """
    results = []
    
    # Create a partial function with filter parameters
    process_func = partial(
        process_ticker_technical_first,
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        show_oversold=show_oversold,
        show_overbought=show_overbought,
        show_neutral=show_neutral
    )
    
    # Process tickers in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        batch_results = list(executor.map(process_func, batch_tickers))
    
    # Filter out None results
    return [r for r in batch_results if r is not None]

def main():
    st.set_page_config(
        page_title="IDX Stock Screener",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better mobile responsiveness
    st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
    }
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
    """, unsafe_allow_html=True)
    
    # App header with stats
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("IDX Stock Screener")
        st.markdown(f"Screening **{len(IDX_ALL_TICKERS_YF)}** Indonesian stocks based on technical analysis first, then fundamental criteria")
    with col2:
        st.metric("Total IDX Stocks", f"{len(IDX_ALL_TICKERS_YF)}")
    
    # Initialize session state for errors and results cache
    if 'errors' not in st.session_state:
        st.session_state.errors = {}
    
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = None
    
    if 'results_cache' not in st.session_state:
        st.session_state.results_cache = None
    
    if 'filter_settings' not in st.session_state:
        st.session_state.filter_settings = {
            'rsi_min': 0,
            'rsi_max': 100,
            'show_oversold': True,
            'show_overbought': True,
            'show_neutral': True,
            'min_ni': DEFAULT_MIN_NI,
            'max_pe': DEFAULT_MAX_PE,
            'max_pb': DEFAULT_MAX_PB,
            'min_growth': 0.0
        }
    
    # Sidebar filters with tabs for better organization
    with st.sidebar:
        st.header("IDX Stock Screener Filters")
        
        # Create tabs for filter categories
        tab1, tab2, tab3, tab4 = st.tabs(["Technical", "Fundamental", "Performance", "Settings"])
        
        with tab1:
            st.subheader("Technical Filters (First Pass)")
            st.write(f"RSI Period: {RSI_PERIOD}")
            
            # RSI range sliders
            rsi_min, rsi_max = st.slider(
                "RSI Range", 
                0, 100, 
                (0, 100),
                help="Filter stocks by RSI range"
            )
            
            # RSI signal checkboxes
            show_oversold = st.checkbox(
                "Show Oversold Stocks (RSI < 30)", 
                st.session_state.filter_settings['show_oversold'],
                help="Include stocks with RSI below 30 (potentially undervalued)"
            )
            show_overbought = st.checkbox(
                "Show Overbought Stocks (RSI > 70)", 
                st.session_state.filter_settings['show_overbought'],
                help="Include stocks with RSI above 70 (potentially overvalued)"
            )
            show_neutral = st.checkbox(
                "Show Neutral Stocks", 
                st.session_state.filter_settings['show_neutral'],
                help="Include stocks with RSI between 30 and 70"
            )
        
        with tab2:
            st.subheader("Fundamental Filters (Second Pass)")
            min_ni = st.slider(
                "Minimum Net Income (T IDR)", 
                0.1, 10.0, 
                st.session_state.filter_settings['min_ni'], 
                0.1,
                help="Minimum Net Income in trillion IDR"
            )
            max_pe = st.slider(
                "Maximum P/E Ratio", 
                5.0, 50.0, 
                st.session_state.filter_settings['max_pe'], 
                0.5,
                help="Maximum Price-to-Earnings ratio"
            )
            max_pb = st.slider(
                "Maximum P/B Ratio", 
                0.5, 5.0, 
                st.session_state.filter_settings['max_pb'], 
                0.1,
                help="Maximum Price-to-Book ratio"
            )
            min_growth = st.slider(
                "Minimum YoY Growth (%)", 
                -50.0, 100.0, 
                st.session_state.filter_settings['min_growth'], 
                5.0,
                help="Minimum Year-over-Year growth percentage"
            )
        
        with tab3:
            st.subheader("Performance Settings")
            batch_size = st.slider(
                "Batch Size", 
                10, 100, BATCH_SIZE, 10,
                help="Number of stocks to process in each batch"
            )
            max_workers = st.slider(
                "Max Concurrent Workers", 
                1, 20, MAX_WORKERS, 1,
                help="Maximum number of parallel processing threads"
            )
        
        with tab4:
            st.subheader("Refresh Settings")
            refresh = st.toggle(
                "Auto-refresh", 
                True,
                help="Automatically refresh data at specified intervals"
            )
            refresh_interval = st.slider(
                "Refresh Interval (minutes)", 
                1, 60, 5, 
                help="Time between automatic refreshes"
            ) if refresh else 0
            
            # Debug options
            st.subheader("Advanced Options")
            show_errors = st.checkbox(
                "Show Error Log", 
                False,
                help="Display errors encountered during data retrieval"
            )
            
            if st.button("Clear Cache", help="Clear all cached data and force refresh"):
                st.cache_data.clear()
                st.session_state.last_refresh = None
                st.session_state.results_cache = None
                st.success("Cache cleared!")
        
        # Save filter settings to session state
        st.session_state.filter_settings = {
            'rsi_min': rsi_min,
            'rsi_max': rsi_max,
            'show_oversold': show_oversold,
            'show_overbought': show_overbought,
            'show_neutral': show_neutral,
            'min_ni': min_ni,
            'max_pe': max_pe,
            'max_pb': max_pb,
            'min_growth': min_growth
        }
        
        # Manual refresh button (outside tabs for visibility)
        if st.button("🔄 Refresh Now", help="Force refresh data now"):
            st.session_state.last_refresh = None
            st.session_state.results_cache = None
    
    # Main content area
    # Create tabs for results and statistics
    main_tab1, main_tab2 = st.tabs(["Screening Results", "Statistics & Info"])
    
    with main_tab1:
        # Progress indicators
        col1, col2 = st.columns([3, 1])
        with col1:
            progress_bar = st.progress(0)
        with col2:
            status_text = st.empty()
        
        # Create placeholders for intermediate and final results
        technical_results_placeholder = st.empty()
        results_placeholder = st.empty()
    
    with main_tab2:
        st.subheader("About IDX Stock Screener")
        st.write("""
        This application screens all stocks listed on the Indonesia Stock Exchange (IDX) using a two-pass approach:
        
        **First Pass - Technical Screening:**
        - RSI(25) with signals for oversold (RSI<30) and overbought (RSI>70) conditions
        - Custom RSI range filtering
        
        **Second Pass - Fundamental Screening:**
        - Net Income > specified threshold (in trillion IDR)
        - Positive YoY growth (or as specified)
        - P/E ratio < specified threshold
        - P/B ratio < specified threshold
        
        **Data Sources:**
        - Financial data from Yahoo Finance API
        - Data is cached to minimize API calls (fundamentals: 24h, RSI: 5min)
        """)
        
        # Display current filter summary
        st.subheader("Current Filter Settings")
        filter_df = pd.DataFrame({
            'Filter': ['RSI Min', 'RSI Max', 'Show Oversold', 'Show Overbought', 'Show Neutral',
                      'Min Net Income (T IDR)', 'Max P/E Ratio', 'Max P/B Ratio', 'Min Growth (%)'],
            'Value': [rsi_min, rsi_max, show_oversold, show_overbought, show_neutral,
                     min_ni, max_pe, max_pb, min_growth]
        })
        st.dataframe(filter_df, use_container_width=True)
        
        # Display performance metrics if available
        if st.session_state.results_cache:
            st.subheader("Performance Metrics")
            perf_df = pd.DataFrame({
                'Metric': ['Total Stocks Screened', 'Technical Pass', 'Final Results', 'Processing Time (s)', 'Errors'],
                'Value': [
                    len(IDX_ALL_TICKERS_YF),
                    st.session_state.results_cache.get('technical_count', 0),
                    st.session_state.results_cache.get('final_count', 0),
                    f"{st.session_state.results_cache.get('elapsed_time', 0):.2f}",
                    st.session_state.results_cache.get('errors', 0)
                ]
            })
            st.dataframe(perf_df, use_container_width=True)
    
    # Check if we need to refresh
    current_time = datetime.now()
    need_refresh = (
        st.session_state.last_refresh is None or
        st.session_state.results_cache is None or
        (refresh and refresh_interval > 0 and
         st.session_state.last_refresh is not None and
         (current_time - st.session_state.last_refresh).total_seconds() > refresh_interval * 60)
    )
    
    # Function to perform screening
    def perform_screening():
        technical_results = []
        final_results = []
        errors = 0
        start_time = time.time()
        
        # Clear previous errors
        st.session_state.errors = {}
        
        # FIRST PASS: Technical Screening
        # Process tickers in batches to avoid timeouts
        num_batches = (len(IDX_ALL_TICKERS_YF) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(IDX_ALL_TICKERS_YF))
            batch_tickers = IDX_ALL_TICKERS_YF[batch_start:batch_end]
            
            # Update progress
            progress_bar.progress((batch_idx) / (num_batches * 2))  # First half of progress bar for technical screening
            status_text.text(f"Technical Screening: Batch {batch_idx + 1}/{num_batches} (tickers {batch_start + 1}-{batch_end})")
            
            # Process batch with technical filters first
            batch_results = process_batch_technical_first(
                batch_tickers,
                rsi_min,
                rsi_max,
                show_oversold,
                show_overbought,
                show_neutral
            )
            
            # Add batch results to technical results
            technical_results.extend(batch_results)
            
            # Count errors
            errors = len(st.session_state.errors)
        
        # Update progress after technical screening
        progress_bar.progress(0.5)  # 50% complete after technical screening
        status_text.text(f"Technical Screening Complete: Found {len(technical_results)} stocks")
        
        # Display intermediate technical results
        if technical_results:
            tech_df = pd.DataFrame(technical_results, columns=["Ticker", "RSI", "Signal"])
            tech_df['RSI_Display'] = tech_df['RSI'].apply(lambda x: format_progress_bar(x, 0, 100, 'rsi'))
            tech_styled_df = tech_df[["Ticker", "RSI_Display", "Signal"]].style.apply(
                lambda _: highlight_oversold(tech_df), axis=None
            )
            
            with technical_results_placeholder.container():
                st.subheader("Technical Screening Results")
                st.write(f"Found {len(technical_results)} stocks matching technical criteria")
                st.dataframe(tech_styled_df, height=200, use_container_width=True)
                st.write("Applying fundamental filters...")
        
        # SECOND PASS: Fundamental Screening
        status_text.text(f"Fundamental Screening: Processing {len(technical_results)} stocks")
        
        # Apply fundamental filters to stocks that passed technical screening
        final_results = apply_fundamental_filters(
            technical_results,
            min_ni,
            max_pe,
            max_pb,
            min_growth
        )
        
        # Complete the progress bar
        progress_bar.progress(1.0)
        status_text.text(f"Screening Complete: Found {len(final_results)} stocks matching all criteria")
        
        # Update last refresh time
        st.session_state.last_refresh = current_time
        
        # Create DataFrame from results
        if final_results:
            df = pd.DataFrame(final_results, columns=["Ticker", "NI(T)", "Growth(%)", "P/E", "P/B", "RSI", "Signal"])
            
            # Sort by Net Income (descending)
            df = df.sort_values(by="NI(T)", ascending=False)
            
            # Format RSI column with progress bars
            df['RSI_Display'] = df['RSI'].apply(lambda x: format_progress_bar(x, 0, 100, 'rsi'))
            
            # Format numeric columns
            df['NI(T)'] = df['NI(T)'].map('{:.2f}'.format)
            df['Growth(%)'] = df['Growth(%)'].map('{:.2f}'.format)
            df['P/E'] = df['P/E'].map('{:.2f}'.format)
            df['P/B'] = df['P/B'].map('{:.2f}'.format)
            
            # Reorder columns for display
            display_df = df[["Ticker", "NI(T)", "Growth(%)", "P/E", "P/B", "RSI_Display", "Signal"]]
            
            # Apply styling
            styled_df = display_df.style.apply(lambda _: highlight_oversold(df), axis=None)
            
            # Cache the results
            st.session_state.results_cache = {
                'df': df,
                'display_df': display_df,
                'styled_df': styled_df,
                'technical_count': len(technical_results),
                'final_count': len(final_results),
                'elapsed_time': time.time() - start_time,
                'errors': errors
            }
            
            return st.session_state.results_cache
        else:
            st.session_state.results_cache = {
                'df': None,
                'display_df': None,
                'styled_df': None,
                'technical_count': len(technical_results),
                'final_count': 0,
                'elapsed_time': time.time() - start_time,
                'errors': errors
            }
            
            return st.session_state.results_cache
    
    # Perform screening if needed
    if need_refresh:
        results = perform_screening()
    else:
        results = st.session_state.results_cache
        # Show cached status
        status_text.text(f"Using cached results from {st.session_state.last_refresh.strftime('%H:%M:%S')}")
        progress_bar.progress(1.0)
    
    # Display results in the first tab
    with main_tab1:
        if results.get('final_count', 0) > 0:
            # Create result header with metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Technical Pass", results.get('technical_count', 0))
            with col2:
                st.metric("Final Results", results.get('final_count', 0))
            with col3:
                st.metric("Processing Time", f"{results.get('elapsed_time', 0):.2f}s")
            
            # Display results table
            with results_placeholder.container():
                st.subheader("Final Results (Technical + Fundamental)")
                st.dataframe(
                    results['styled_df'], 
                    height=500, 
                    use_container_width=True
                )
                
                # Add download button for CSV export
                if results['df'] is not None:
                    csv = results['df'].to_csv(index=False)
                    st.download_button(
                        label="📥 Download results as CSV",
                        data=csv,
                        file_name=f"idx_screener_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )
        else:
            results_placeholder.info("No stocks found matching all criteria. Try adjusting your filters.")
    
    # Show error log if requested
    if show_errors and st.session_state.errors:
        with st.expander("Error Log"):
            st.write(f"Total errors: {len(st.session_state.errors)}")
            
            # Show first 10 errors to avoid cluttering the UI
            for i, (ticker, error) in enumerate(list(st.session_state.errors.items())[:10]):
                st.error(f"{ticker}: {error}")
            
            if len(st.session_state.errors) > 10:
                st.write(f"... and {len(st.session_state.errors) - 10} more errors")
    
    # Set up auto-refresh
    if refresh and refresh_interval > 0:
        next_refresh = st.session_state.last_refresh + pd.Timedelta(minutes=refresh_interval)
        st.write(f"Auto-refreshing every {refresh_interval} minutes. Next update: {next_refresh.strftime('%H:%M:%S')}")
        
        # Check if it's time to refresh
        if datetime.now() >= next_refresh:
            time.sleep(1)  # Small delay
            st.experimental_rerun()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 10px; border-top: 1px solid #ddd;">
        <p style="color: #666; font-size: 0.8em;">
            IDX Stock Screener | Data from Yahoo Finance | Updated: {update_time}
        </p>
    </div>
    """.format(update_time=st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.last_refresh else "Never"), 
    unsafe_allow_html=True)

if __name__ == "__main__":
    main()
