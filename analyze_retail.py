import pandas as pd
from datetime import timedelta

# Read the Excel file
print("Reading Excel file...")
data = pd.read_excel('Online Retail.xlsx')

# Calculate the date range
print("\nAnalyzing date range...")
min_date = data['InvoiceDate'].min()
max_date = data['InvoiceDate'].max()

# Convert to pandas Timestamp objects if they aren't already
if not isinstance(min_date, pd.Timestamp):
    min_date = pd.Timestamp(min_date)
if not isinstance(max_date, pd.Timestamp):
    max_date = pd.Timestamp(max_date)

# Calculate the difference in days
date_range = (max_date.date() - min_date.date())
days = date_range.days

print(f"First transaction date: {min_date.date()}")
print(f"Last transaction date: {max_date.date()}")
print(f"The dataset spans {days} days ({days/365.25:.2f} years)")

# Clean the data
print("\nCleaning data...")
# Remove rows with missing values
data = data.dropna(subset=['CustomerID'])
# Remove rows with negative or zero quantities
data = data[data['Quantity'] > 0]
# Remove rows with negative or zero unit prices
data = data[data['UnitPrice'] > 0]

# Calculate revenue (Quantity * UnitPrice)
print("Calculating revenue...")
data['Revenue'] = data['Quantity'] * data['UnitPrice']

# Group by country and sum the revenue
print("Calculating total revenue by country...")
revenue_by_country = data.groupby('Country')['Revenue'].sum().sort_values(ascending=False)

# Print the top 5 countries by revenue
print("\nTop 5 countries by revenue:")
print(revenue_by_country.head(5))

# Print the country with the most revenue
top_country = revenue_by_country.index[0]
top_revenue = revenue_by_country.iloc[0]
print(f"\nThe country with the most total revenue is {top_country} with £{top_revenue:.2f}") 