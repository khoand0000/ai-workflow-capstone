import pandas as pd

# Read the Excel file
print("Reading Excel file...")
data = pd.read_excel('Online Retail.xlsx')

# Calculate the date range
print("Analyzing date range...")
min_date = data['InvoiceDate'].min()
max_date = data['InvoiceDate'].max()

# Calculate the difference in days
date_range_days = (max_date.date() - min_date.date()).days

print(f"First transaction date: {min_date.date()}")
print(f"Last transaction date: {max_date.date()}")
print(f"The dataset spans {date_range_days} days ({date_range_days/365.25:.2f} years)") 