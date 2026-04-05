import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Revenue'].fillna(df['Revenue'].median(), inplace=True)
df.drop_duplicates(inplace=True)

df['Year']       = df['Date'].dt.year
df['Month']      = df['Date'].dt.month
df['Quarter']    = df['Date'].dt.quarter
df['Month_Name'] = df['Date'].dt.strftime('%b')
df['Month_Year'] = df['Date'].dt.to_period('M')

print('=== SALES SUMMARY ===')
print(f'Total Revenue : Rs.{df["Revenue"].sum():,.0f}')
print(f'Total Profit  : Rs.{df["Profit"].sum():,.0f}')
print(f'Total Units   : {df["Units_Sold"].sum():,}')
print(f'Avg Order     : Rs.{df["Revenue"].mean():,.0f}')

print('\nRevenue by Region:')
print(df.groupby('Region')['Revenue'].sum().sort_values(ascending=False))

print('\nRevenue by Product:')
print(df.groupby('Product')['Revenue'].sum().sort_values(ascending=False))

print('\nRevenue by Sales Rep:')
print(df.groupby('Sales_Rep')['Revenue'].sum().sort_values(ascending=False))

df.to_csv('sales_data_clean.csv', index=False)
print('\nClean data saved as sales_data_clean.csv')