import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Create sample data
def create_sample_data():
    # Start date
    start_date = datetime(2023, 1, 1, 10, 0, 0)
    
    # Create empty lists for data
    data = []
    
    # Generate 50 transactions
    for i in range(50):
        # Random date within the last year
        date = start_date + timedelta(days=i*3, hours=random.randint(0, 12))
        date_str = date.strftime("%d/%m/%Y, %H:%M:%S")
        
        # Determine trade type
        trade_type = random.choice(['P2P', 'E-Voucher'])
        
        # Determine transaction type
        transaction_type = random.choice(['Buy', 'Sell'])
        
        # Determine currency based on trade type
        if trade_type == 'P2P':
            currency = random.choice(['AED', 'EGP'])
        else:
            # E-Voucher: Buy is always AED, Sell is always EGP
            currency = 'AED' if transaction_type == 'Buy' else 'EGP'
        
        # USDT amount
        usdt_amount = round(random.uniform(50, 500), 2)
        
        # Price per USDT
        if currency == 'AED':
            price = round(random.uniform(3.65, 3.75), 2)  # AED/USDT
        else:
            price = round(random.uniform(45, 50), 2)  # EGP/USDT
        
        # Real amount (after fees)
        real_amount = round(usdt_amount * price * 0.99, 2)  # 1% fee
        
        # AED/EGP exchange rate
        aed_egp_rate = round(random.uniform(12.5, 13.5), 2)
        
        # Status (mostly completed)
        status = 'COMPLETED' if random.random() < 0.9 else 'CANCELLED'
        
        # Create transaction record
        transaction = {
            'Reference': f'TX{i+1:04d}',
            'Type': transaction_type,
            'Currency': currency,
            'Amount': round(usdt_amount * price, 2),  # Amount before fees
            'Real Amount': real_amount,
            'AED/EGP': aed_egp_rate,
            'Usdt B': usdt_amount,
            'USDT': round(usdt_amount * 0.99, 2) if transaction_type == 'Buy' else usdt_amount,  # Apply fee for Buy
            'Price': price,
            'Fees': round(usdt_amount * price * 0.01, 2),  # 1% fee
            'Status': status,
            'Date': date_str,
            'TradeType': trade_type,
            'Source': random.choice(['Binance', 'LocalBitcoins', 'OTC'])
        }
        
        data.append(transaction)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to Excel
    df.to_excel('sample_data.xlsx', index=False)
    print("Sample data created and saved to 'sample_data.xlsx'")

if __name__ == "__main__":
    create_sample_data()
