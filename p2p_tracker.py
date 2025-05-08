import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Set page config
st.set_page_config(
    page_title="P2P & E-Voucher Tracker",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1E88E5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
    .currency-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .currency-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
    }
    .currency-value {
        font-size: 24px;
        font-weight: bold;
        color: #1E88E5;
    }
    .currency-subtitle {
        font-size: 14px;
        color: #666;
        margin-top: 5px;
    }
    .total-value-card {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        border-left: 5px solid #1565C0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .profit-card {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #2E7D32;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .profit-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #2E7D32;
    }
    .profit-value {
        font-size: 24px;
        font-weight: bold;
        color: #2E7D32;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("💱 P2P & E-Voucher Transaction Tracker")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("Initial Settings")

# Function to load and process data
def process_data(file, initial_balances, initial_usdt_rate):
    try:
        # Read Excel file
        df = pd.read_excel(file)

        # Check if required columns exist
        required_columns = ['Date', 'Type', 'TradeType', 'Currency', 'Real Amount', 'USDT', 'Price', 'Status']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None

        # Add AED/EGP column if it doesn't exist
        if 'AED/EGP' not in df.columns:
            df['AED/EGP'] = 13.75  # Default value

        # Filter completed transactions
        df = df[df['Status'] == 'COMPLETED'].copy()

        # Convert date to datetime
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y, %H:%M:%S')
        except:
            try:
                # Try alternative format
                df['Date'] = pd.to_datetime(df['Date'])
            except:
                st.error("Could not parse date format. Please ensure dates are in format 'DD/MM/YYYY, HH:MM:SS'")
                return None

        # Sort by date
        df = df.sort_values('Date')

        # Initialize balances dictionary with initial balances
        balances = initial_balances.copy()

        # Initialize lists to store balance history
        balance_history = []
        profit_history = []
        cash_flow = []

        # Initialize FIFO queue for USDT purchases with initial USDT
        usdt_queue = []

        # Add initial USDT to the queue with the initial rate
        if 'USDT' in initial_balances and initial_balances['USDT'] > 0:
            usdt_queue.append((initial_balances['USDT'], initial_usdt_rate))

        # Initialize profit tracking
        total_p2p_profit = 0
        total_evoucher_profit = 0

        # Calculate average USDT rate
        def calculate_avg_usdt_rate():
            if not usdt_queue or sum(amount for amount, _ in usdt_queue) == 0:
                return initial_usdt_rate

            total_usdt = sum(amount for amount, _ in usdt_queue)
            total_value = sum(amount * rate for amount, rate in usdt_queue)
            return total_value / total_usdt if total_usdt > 0 else initial_usdt_rate

        # Process each transaction
        for idx, row in df.iterrows():
            date = row['Date']
            trade_type = row['TradeType']
            transaction_type = row['Type']
            currency = row['Currency']
            real_amount = row['Real Amount']
            usdt_amount = row['USDT']
            price = row['Price']
            aed_egp_rate = row['AED/EGP']

            # Ensure currency exists in balances
            if currency not in balances:
                balances[currency] = 0

            # Calculate transaction profit
            transaction_profit = 0

            # Process based on trade type
            if trade_type == 'P2P':
                # P2P transactions
                if transaction_type == 'Buy':
                    # Buy USDT: decrease local currency, increase USDT
                    balances[currency] -= real_amount
                    balances['USDT'] += usdt_amount

                    # Add to USDT queue with purchase price in AED
                    if currency == 'AED':
                        usdt_queue.append((usdt_amount, price))
                    else:
                        # Convert price to AED equivalent
                        aed_price = price / aed_egp_rate if currency == 'EGP' else price
                        usdt_queue.append((usdt_amount, aed_price))

                elif transaction_type == 'Sell':
                    # Sell USDT: increase local currency, decrease USDT
                    balances[currency] += real_amount
                    balances['USDT'] -= usdt_amount

                    # Calculate profit using FIFO method
                    remaining_usdt = usdt_amount
                    sell_price_in_aed = price

                    # For debugging
                    debug_info = []

                    # Convert sell price to AED if needed
                    if currency != 'AED':
                        sell_price_in_aed = price * aed_egp_rate if currency == 'EGP' else price

                    # Calculate total sell value
                    total_sell_value = sell_price_in_aed * usdt_amount
                    total_cost_value = 0

                    # Track original purchase rate for this sell transaction
                    original_purchase_rate = 0

                    # Track USDT used for detailed calculation
                    usdt_used_details = []

                    while remaining_usdt > 0 and usdt_queue:
                        bought_usdt, bought_price = usdt_queue[0]

                        if bought_usdt <= remaining_usdt:
                            # Use entire bought batch
                            batch_cost = bought_usdt * bought_price
                            total_cost_value += batch_cost

                            # Add to debug info
                            debug_info.append(f"Used {bought_usdt} USDT @ {bought_price} = {batch_cost} AED")
                            usdt_used_details.append((bought_usdt, bought_price))

                            remaining_usdt -= bought_usdt
                            usdt_queue.pop(0)
                        else:
                            # Use partial bought batch
                            batch_cost = remaining_usdt * bought_price
                            total_cost_value += batch_cost

                            # Add to debug info
                            debug_info.append(f"Used {remaining_usdt} USDT @ {bought_price} = {batch_cost} AED")
                            usdt_used_details.append((remaining_usdt, bought_price))

                            usdt_queue[0] = (bought_usdt - remaining_usdt, bought_price)
                            remaining_usdt = 0

                    # Calculate profit as the difference between sell value and cost
                    transaction_profit = total_sell_value - total_cost_value

                    # Calculate original purchase rate (weighted average of used USDT)
                    if usdt_amount > 0:
                        original_purchase_rate = total_cost_value / usdt_amount

                    # Add detailed calculation to debug info
                    debug_info.append(f"Sell value: {usdt_amount} USDT @ {sell_price_in_aed} = {total_sell_value} AED")
                    debug_info.append(f"Cost value: {total_cost_value} AED")
                    debug_info.append(f"Original purchase rate: {original_purchase_rate} AED")
                    debug_info.append(f"Profit: {transaction_profit} AED")

                    # Store debug info for this transaction
                    if 'debug_transactions' not in globals():
                        globals()['debug_transactions'] = []

                    globals()['debug_transactions'].append({
                        'date': date,
                        'type': 'Sell',
                        'usdt_amount': usdt_amount,
                        'sell_price': sell_price_in_aed,
                        'total_sell_value': total_sell_value,
                        'usdt_used': usdt_used_details,
                        'total_cost_value': total_cost_value,
                        'original_purchase_rate': original_purchase_rate,
                        'profit': transaction_profit,
                        'debug_info': debug_info
                    })

                    total_p2p_profit += transaction_profit

            elif trade_type == 'E-Voucher':
                # E-Voucher transactions
                if transaction_type == 'Buy':
                    # Receiving AED from workers
                    balances['AED'] += real_amount

                elif transaction_type == 'Sell':
                    # Selling USDT for EGP to families
                    # Note: We don't add EGP to balances as it's sent to families, not kept
                    balances['USDT'] -= usdt_amount

                    # Calculate cost of USDT in AED
                    usdt_cost_in_aed = 0
                    remaining_usdt = usdt_amount

                    while remaining_usdt > 0 and usdt_queue:
                        bought_usdt, bought_price = usdt_queue[0]

                        if bought_usdt <= remaining_usdt:
                            # Use entire bought batch
                            usdt_cost_in_aed += bought_usdt * bought_price
                            remaining_usdt -= bought_usdt
                            usdt_queue.pop(0)
                        else:
                            # Use partial bought batch
                            usdt_cost_in_aed += remaining_usdt * bought_price
                            usdt_queue[0] = (bought_usdt - remaining_usdt, bought_price)
                            remaining_usdt = 0

                    # For E-Voucher, profit is calculated separately
                    # We don't add it to the P2P profit calculation
                    # The profit is the AED received (from Buy transactions) minus the cost of USDT sold
                    # This will be calculated separately in the E-Voucher analysis
                    transaction_profit = 0  # Don't calculate profit here

            # Calculate current average USDT rate
            current_avg_usdt_rate = calculate_avg_usdt_rate()

            # Calculate USDT value in AED
            usdt_value_in_aed = balances.get('USDT', 0) * current_avg_usdt_rate

            # Record balance history
            balance_record = {
                'Date': date,
                'TradeType': trade_type,
                'Type': transaction_type,
                'Currency': currency,
                'Real Amount': real_amount,
                'USDT': usdt_amount,
                'Price': price,
                'Profit': transaction_profit if trade_type == 'P2P' else 0,  # Only show profit for P2P transactions
                'Current USDT Rate': current_avg_usdt_rate,
                'USDT Value (AED)': usdt_value_in_aed
            }

            # Add current balances to record
            for curr, bal in balances.items():
                balance_record[f'{curr} Balance'] = bal

            balance_history.append(balance_record)

            # Record profit history
            profit_history.append({
                'Date': date,
                'P2P Profit': total_p2p_profit,
                'E-Voucher Profit': total_evoucher_profit,
                'Total Profit': total_p2p_profit + total_evoucher_profit
            })

            # Record cash flow
            cash_flow_record = {
                'Date': date,
                'Transaction': f"{transaction_type} {usdt_amount} USDT @ {price} ({trade_type})",
                'AED Balance': balances.get('AED', 0),
                'USDT Balance': balances.get('USDT', 0),
                'USDT Rate': current_avg_usdt_rate,
                'USDT Value (AED)': usdt_value_in_aed,
                'Total Value (AED)': balances.get('AED', 0) + usdt_value_in_aed,  # Only count AED and USDT value
                'Transaction Profit': transaction_profit if trade_type == 'P2P' else 0,  # Only show profit for P2P transactions
                'Cumulative P2P Profit': total_p2p_profit
            }

            # Add original purchase rate for sell transactions
            if transaction_type == 'Sell' and trade_type == 'P2P':
                cash_flow_record['Original Purchase Rate'] = original_purchase_rate
            cash_flow.append(cash_flow_record)

        # Create DataFrame from balance history
        balance_df = pd.DataFrame(balance_history)
        profit_df = pd.DataFrame(profit_history)
        cash_flow_df = pd.DataFrame(cash_flow)

        # Calculate final USDT rate
        final_usdt_rate = calculate_avg_usdt_rate()

        # Calculate final USDT value in AED
        final_usdt_value = balances.get('USDT', 0) * final_usdt_rate

        # Create debug transactions DataFrame if available
        debug_df = None
        if 'debug_transactions' in globals() and globals()['debug_transactions']:
            debug_df = pd.DataFrame(globals()['debug_transactions'])

        # Calculate E-Voucher profit separately
        evoucher_profit = 0
        evoucher_df = df[df['TradeType'] == 'E-Voucher'].copy()

        if not evoucher_df.empty:
            # Get all Buy transactions (AED received)
            buy_transactions = evoucher_df[evoucher_df['Type'] == 'Buy']
            total_aed_received = buy_transactions['Real Amount'].sum()

            # Get all Sell transactions (USDT sold)
            sell_transactions = evoucher_df[evoucher_df['Type'] == 'Sell']

            # Calculate the cost of USDT sold based on the current USDT rate
            total_usdt_cost = 0
            for _, row in sell_transactions.iterrows():
                usdt_amount = row['USDT']
                # Use the average USDT rate at the time of the transaction
                idx = balance_df[balance_df['Date'] == row['Date']].index[0]
                usdt_rate = balance_df.loc[idx, 'Current USDT Rate']
                total_usdt_cost += usdt_amount * usdt_rate

            # Calculate profit
            evoucher_profit = total_aed_received - total_usdt_cost

        return {
            'transactions': df,
            'balance_history': balance_df,
            'profit_history': profit_df,
            'cash_flow': cash_flow_df,
            'final_balances': balances,
            'total_p2p_profit': total_p2p_profit,
            'total_evoucher_profit': evoucher_profit,  # Use the separately calculated E-Voucher profit
            'final_usdt_rate': final_usdt_rate,
            'final_usdt_value': final_usdt_value,
            'debug_transactions': debug_df  # Add debug transactions
        }

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# Function to create balance chart
def create_balance_chart(balance_df, currencies):
    fig = go.Figure()

    for currency in currencies:
        fig.add_trace(
            go.Scatter(
                x=balance_df['Date'],
                y=balance_df[f'{currency} Balance'],
                mode='lines+markers',
                name=currency
            )
        )

    fig.update_layout(
        title='Balance History',
        xaxis_title='Date',
        yaxis_title='Balance',
        legend_title='Currency',
        height=500
    )

    return fig

# Function to create profit chart
def create_profit_chart(profit_df):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=profit_df['Date'],
            y=profit_df['P2P Profit'],
            mode='lines+markers',
            name='P2P Profit'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=profit_df['Date'],
            y=profit_df['E-Voucher Profit'],
            mode='lines+markers',
            name='E-Voucher Profit'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=profit_df['Date'],
            y=profit_df['Total Profit'],
            mode='lines+markers',
            name='Total Profit',
            line=dict(width=3)
        )
    )

    fig.update_layout(
        title='Profit History',
        xaxis_title='Date',
        yaxis_title='Profit (AED)',
        legend_title='Profit Type',
        height=500
    )

    return fig

# Function to analyze E-Voucher transactions
def analyze_evoucher(df, balance_history):
    evoucher_df = df[df['TradeType'] == 'E-Voucher'].copy()

    # Get E-Voucher transactions from balance history for profit calculation
    evoucher_history = balance_history[balance_history['TradeType'] == 'E-Voucher'].copy()

    # Calculate metrics
    total_aed_received = evoucher_df[evoucher_df['Type'] == 'Buy']['Real Amount'].sum()
    total_usdt_sold = evoucher_df[evoucher_df['Type'] == 'Sell']['USDT'].sum()
    avg_aed_egp_rate = evoucher_df['AED/EGP'].mean()

    # Calculate E-Voucher profit
    # For each USDT sold in E-Voucher, calculate its cost in AED
    evoucher_profit = 0

    if not evoucher_history.empty:
        # Get all Buy transactions (AED received)
        buy_transactions = evoucher_history[evoucher_history['Type'] == 'Buy']
        total_aed_received = buy_transactions['Real Amount'].sum()

        # Get all Sell transactions (USDT sold)
        sell_transactions = evoucher_history[evoucher_history['Type'] == 'Sell']

        # Calculate the cost of USDT sold
        total_usdt_cost = 0
        for _, row in sell_transactions.iterrows():
            usdt_amount = row['USDT']
            usdt_rate = row['Current USDT Rate']
            total_usdt_cost += usdt_amount * usdt_rate

        # Calculate profit
        evoucher_profit = total_aed_received - total_usdt_cost

    # Create summary
    summary = {
        'Total AED Received': total_aed_received,
        'Total USDT Sold': total_usdt_sold,
        'Average AED/EGP Rate': avg_aed_egp_rate,
        'E-Voucher Profit': evoucher_profit
    }

    # Create rate chart
    rate_fig = px.line(
        evoucher_df,
        x='Date',
        y='AED/EGP',
        title='AED/EGP Exchange Rate Over Time'
    )

    return summary, rate_fig, evoucher_profit

# Main application
def main():
    # Sidebar for initial balances
    st.sidebar.subheader("Enter Initial Balances")

    # Default currencies
    default_currencies = ['AED', 'EGP', 'USDT']

    # Initial balances
    initial_balances = {}
    for currency in default_currencies:
        initial_balances[currency] = st.sidebar.number_input(f"Initial {currency} Balance", value=0.0, step=0.01, format="%.15f")

    # Initial USDT rate
    st.sidebar.subheader("Initial USDT Rate")
    initial_usdt_rate = st.sidebar.number_input("Initial USDT Rate (AED)", value=3.67, step=0.01, format="%.15f",
                                               help="This is the rate at which your initial USDT balance was acquired")

    # E-Voucher AED received
    st.sidebar.subheader("E-Voucher Settings")
    evoucher_aed_received = st.sidebar.number_input("Total AED Received from Workers", value=0.0, step=100.0, format="%.15f",
                                                  help="Enter the total AED amount received from workers for E-Voucher transactions")

    # Option to add more currencies
    st.sidebar.subheader("Add Additional Currency")
    new_currency = st.sidebar.text_input("Currency Code (e.g., USD)")

    if new_currency and new_currency not in initial_balances:
        initial_balances[new_currency] = st.sidebar.number_input(f"Initial {new_currency} Balance", value=0.0, step=0.01, format="%.15f")

    # File uploader
    st.sidebar.subheader("Upload Transaction File")
    uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx", "csv"])

    if uploaded_file is not None:
        # Process data
        results = process_data(uploaded_file, initial_balances, initial_usdt_rate)

        # Update E-Voucher profit with manually entered AED amount
        if results and evoucher_aed_received > 0:
            # Get all E-Voucher sell transactions
            evoucher_df = results['transactions'][
                (results['transactions']['TradeType'] == 'E-Voucher') &
                (results['transactions']['Type'] == 'Sell')
            ]

            if not evoucher_df.empty:
                # Calculate the cost of USDT sold
                total_usdt_cost = 0
                for _, row in evoucher_df.iterrows():
                    usdt_amount = row['USDT']
                    # Find the corresponding row in balance history
                    idx = results['balance_history'][results['balance_history']['Date'] == row['Date']].index[0]
                    usdt_rate = results['balance_history'].loc[idx, 'Current USDT Rate']
                    total_usdt_cost += usdt_amount * usdt_rate

                # Calculate profit
                results['total_evoucher_profit'] = evoucher_aed_received - total_usdt_cost

        if results:
            # Create tabs
            tabs = st.tabs(["Dashboard", "Cash Flow", "Transactions", "P2P Analysis", "E-Voucher Analysis", "Debug"])

            # Dashboard tab
            with tabs[0]:
                # Summary metrics - using custom styled cards
                st.markdown(f"""
                <div class="profit-card">
                    <div class="profit-title">Total P2P Profit (AED)</div>
                    <div class="profit-value">{results['total_p2p_profit']:.5f}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="profit-card">
                    <div class="profit-title">Total E-Voucher Profit (AED)</div>
                    <div class="profit-value">{results['total_evoucher_profit']:.5f}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="profit-card">
                    <div class="profit-title">Total Profit (AED)</div>
                    <div class="profit-value">{results['total_p2p_profit'] + results['total_evoucher_profit']:.5f}</div>
                </div>
                """, unsafe_allow_html=True)

                # Explanation of profit calculation
                st.info("""
                **Profit Calculation:**
                - **P2P Profit**: Calculated from P2P transactions only, using the FIFO method (first in, first out)
                - **E-Voucher Profit**: Calculated separately as (AED received from workers - cost of USDT sold)
                - **Total Profit**: Sum of P2P and E-Voucher profits
                """)

                # Final balances
                st.subheader("Final Balances")

                # Calculate total value in AED (only AED and USDT)
                total_value_aed = (
                    results['final_balances'].get('AED', 0) +
                    results['final_usdt_value']
                )

                # Display each currency in a separate card
                for currency, balance in results['final_balances'].items():
                    if currency == 'USDT':
                        st.markdown(f"""
                        <div class="currency-card">
                            <div class="currency-title">{currency}</div>
                            <div class="currency-value">{balance:.15f}</div>
                            <div class="currency-subtitle">≈ {results['final_usdt_value']:.5f} AED</div>
                        </div>
                        """, unsafe_allow_html=True)
                    elif currency == 'EGP':
                        st.markdown(f"""
                        <div class="currency-card">
                            <div class="currency-title">{currency}</div>
                            <div class="currency-value">{balance:.15f}</div>
                            <div class="currency-subtitle">Not included in total value</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="currency-card">
                            <div class="currency-title">{currency}</div>
                            <div class="currency-value">{balance:.15f}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # Display USDT rate
                st.markdown(f"""
                <div class="currency-card">
                    <div class="currency-title">USDT Rate</div>
                    <div class="currency-value">{results['final_usdt_rate']:.5f} AED</div>
                </div>
                """, unsafe_allow_html=True)

                # Display total value
                st.markdown(f"""
                <div class="total-value-card">
                    <div class="currency-title">Total Value (AED)</div>
                    <div class="currency-value">{total_value_aed:.5f}</div>
                    <div class="currency-subtitle">Sum of AED and USDT value</div>
                </div>
                """, unsafe_allow_html=True)

                # Charts
                st.subheader("Balance History")
                balance_chart = create_balance_chart(results['balance_history'], results['final_balances'].keys())
                st.plotly_chart(balance_chart, use_container_width=True)

                st.subheader("Profit History")
                profit_chart = create_profit_chart(results['profit_history'])
                st.plotly_chart(profit_chart, use_container_width=True)

            # Cash Flow tab
            with tabs[1]:
                st.subheader("Cash Flow Analysis")

                # Display cash flow table
                st.dataframe(results['cash_flow'], use_container_width=True)

                # Create cash flow chart
                st.subheader("Total Value Over Time")

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=results['cash_flow']['Date'],
                        y=results['cash_flow']['Total Value (AED)'],
                        mode='lines+markers',
                        name='Total Value (AED)',
                        line=dict(width=3, color='green')
                    )
                )

                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Value (AED)',
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # Download button
                csv = results['cash_flow'].to_csv(index=False)
                st.download_button(
                    label="Download Cash Flow as CSV",
                    data=csv,
                    file_name="cash_flow.csv",
                    mime="text/csv"
                )

            # Transactions tab
            with tabs[2]:
                st.subheader("All Transactions")
                st.dataframe(results['balance_history'], use_container_width=True)

                # Download button
                csv = results['balance_history'].to_csv(index=False)
                st.download_button(
                    label="Download Transactions as CSV",
                    data=csv,
                    file_name="transactions_with_balances.csv",
                    mime="text/csv"
                )

            # P2P Analysis tab
            with tabs[3]:
                st.subheader("P2P Transactions Analysis")

                # Filter P2P transactions
                p2p_df = results['balance_history'][results['balance_history']['TradeType'] == 'P2P'].copy()

                # P2P metrics
                p2p_buy = p2p_df[p2p_df['Type'] == 'Buy']
                p2p_sell = p2p_df[p2p_df['Type'] == 'Sell']

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total USDT Bought", f"{p2p_buy['USDT'].sum():.15f}")

                with col2:
                    st.metric("Total USDT Sold", f"{p2p_sell['USDT'].sum():.15f}")

                with col3:
                    st.metric("P2P Profit (AED)", f"{results['total_p2p_profit']:.15f}")

                # P2P transactions by currency
                st.subheader("P2P Transactions by Currency")

                currency_counts = p2p_df['Currency'].value_counts().reset_index()
                currency_counts.columns = ['Currency', 'Count']

                fig = px.pie(
                    currency_counts,
                    values='Count',
                    names='Currency',
                    title='Distribution of P2P Transactions by Currency'
                )
                st.plotly_chart(fig, use_container_width=True)

                # P2P transactions table
                st.subheader("P2P Transactions")
                st.dataframe(p2p_df, use_container_width=True)

            # E-Voucher Analysis tab
            with tabs[4]:
                st.subheader("E-Voucher Transactions Analysis")

                # Filter E-Voucher transactions
                evoucher_df = results['balance_history'][results['balance_history']['TradeType'] == 'E-Voucher'].copy()

                if not evoucher_df.empty:
                    # Get E-Voucher analysis
                    evoucher_summary, rate_chart, _ = analyze_evoucher(results['transactions'], results['balance_history'])

                    # Display metrics
                    col1, col2, col3 = st.columns(3)

                    # If manual AED amount was entered, use that instead
                    aed_received = evoucher_aed_received if evoucher_aed_received > 0 else evoucher_summary['Total AED Received']

                    with col1:
                        st.metric("Total AED Received", f"{aed_received:.15f}",
                                 f"{'(Manual Entry)' if evoucher_aed_received > 0 else ''}")

                    with col2:
                        st.metric("Total USDT Sold", f"{evoucher_summary['Total USDT Sold']:.15f}")

                    with col3:
                        st.metric("E-Voucher Profit (AED)", f"{results['total_evoucher_profit']:.15f}")

                    # Explanation of E-Voucher profit calculation
                    st.info(f"""
                    **E-Voucher Profit Calculation:**
                    1. Total AED received from workers: {aed_received:.15f} AED {' (Manual Entry)' if evoucher_aed_received > 0 else ''}
                    2. Minus the cost of USDT sold: {evoucher_summary['Total USDT Sold']:.15f} USDT at average rate
                    3. Total cost of USDT: {(results['total_evoucher_profit'] - aed_received) * -1:.15f} AED
                    4. Profit: {results['total_evoucher_profit']:.15f} AED

                    Note: The EGP sent to families is not counted as an asset since it's transferred out
                    """)

                    # Display exchange rate chart
                    st.subheader("AED/EGP Exchange Rate")
                    st.plotly_chart(rate_chart, use_container_width=True)

                    # E-Voucher transactions table
                    st.subheader("E-Voucher Transactions")
                    st.dataframe(evoucher_df, use_container_width=True)
                else:
                    st.info("No E-Voucher transactions found in the uploaded file.")

            # Debug tab
            with tabs[5]:
                st.subheader("Debug Information")

                if results['debug_transactions'] is not None:
                    st.write("This tab shows detailed calculations for each sell transaction.")

                    # Display debug transactions
                    for i, row in results['debug_transactions'].iterrows():
                        with st.expander(f"Transaction {i+1}: Sell {row['usdt_amount']:.2f} USDT on {row['date']}"):
                            st.write(f"**Sell Details:**")
                            st.write(f"- Amount: {row['usdt_amount']:.15f} USDT")
                            st.write(f"- Sell Price: {row['sell_price']:.15f} AED")
                            st.write(f"- Total Sell Value: {row['total_sell_value']:.15f} AED")

                            st.write("**USDT Used (FIFO method):**")
                            for j, (amount, price) in enumerate(row['usdt_used']):
                                st.write(f"- Batch {j+1}: {amount:.15f} USDT @ {price:.15f} AED = {amount * price:.15f} AED")

                            st.write(f"**Total Cost Value:** {row['total_cost_value']:.15f} AED")
                            st.write(f"**Original Purchase Rate:** {row['original_purchase_rate']:.15f} AED")
                            st.write(f"**Profit:** {row['profit']:.15f} AED")

                            st.write("**Calculation Details:**")
                            for line in row['debug_info']:
                                st.write(f"- {line}")
                else:
                    st.info("No debug information available.")

if __name__ == "__main__":
    main()
