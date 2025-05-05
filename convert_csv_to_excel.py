import pandas as pd

# Read CSV files
df1 = pd.read_csv('sample_data.csv')
df2 = pd.read_csv('sample_data_new.csv')

# Save to Excel
df1.to_excel('sample_data.xlsx', index=False)
df2.to_excel('sample_data_new.xlsx', index=False)

print("CSV files converted to Excel successfully!")
