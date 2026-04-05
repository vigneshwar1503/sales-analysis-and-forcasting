import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os, warnings
warnings.filterwarnings('ignore')

os.makedirs('fig', exist_ok=True)
sns.set_theme(style='whitegrid')

df = pd.read_csv('sales_data_clean.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Month_Year'] = df['Date'].dt.to_period('M')

def save(name):
    plt.savefig(f'fig/{name}', bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Saved: fig/{name}')

# Chart 1 - Revenue by Region
reg = df.groupby('Region')['Revenue'].sum().sort_values(ascending=False)
plt.figure(figsize=(9,5))
sns.barplot(x=reg.index, y=reg.values/1e6, palette='Blues_d')
plt.title('Total Revenue by Region')
plt.ylabel('Revenue (Rs. Millions)')
plt.xlabel('Region')
for i,v in enumerate(reg.values):
    plt.text(i, v/1e6+0.5, f'Rs.{v/1e6:.1f}M', ha='center', fontsize=10)
plt.tight_layout()
save('01_revenue_by_region.png')

# Chart 2 - Revenue by Product
prod = df.groupby('Product')['Revenue'].sum().sort_values()
plt.figure(figsize=(9,5))
sns.barplot(y=prod.index, x=prod.values/1e6, palette='Greens_d')
plt.title('Revenue by Product')
plt.xlabel('Revenue (Rs. Millions)')
plt.tight_layout()
save('02_revenue_by_product.png')

# Chart 3 - Monthly Trend
monthly = df.groupby('Month_Year')['Revenue'].sum().reset_index()
monthly['Label'] = monthly['Month_Year'].astype(str)
plt.figure(figsize=(14,5))
plt.plot(monthly['Label'], monthly['Revenue']/1e6,
         marker='o', linewidth=2.5, color='#2196F3')
plt.fill_between(monthly['Label'], monthly['Revenue']/1e6,
                 alpha=0.1, color='#2196F3')
plt.title('Monthly Revenue Trend')
plt.ylabel('Revenue (Rs. Millions)')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.tight_layout()
save('03_monthly_trend.png')

# Chart 4 - Revenue vs Profit
mon2 = df.groupby('Month_Year')[['Revenue','Profit']].sum().reset_index()
mon2['Label'] = mon2['Month_Year'].astype(str)
x = range(len(mon2))
w = 0.4
plt.figure(figsize=(14,5))
plt.bar([i-w/2 for i in x], mon2['Revenue']/1e6, w, label='Revenue', color='#42A5F5')
plt.bar([i+w/2 for i in x], mon2['Profit']/1e6,  w, label='Profit',  color='#66BB6A')
plt.xticks(list(x), mon2['Label'], rotation=45)
plt.title('Monthly Revenue vs Profit')
plt.ylabel('Amount (Rs. Millions)')
plt.legend()
plt.tight_layout()
save('04_revenue_vs_profit.png')

# Chart 5 - Sales Rep
rep = df.groupby('Sales_Rep')['Revenue'].sum().sort_values(ascending=False)
plt.figure(figsize=(9,5))
colors = ['#FF7043','#42A5F5','#66BB6A','#AB47BC','#FFA726']
plt.bar(rep.index, rep.values/1e6, color=colors)
plt.title('Sales Rep Performance')
plt.ylabel('Revenue (Rs. Millions)')
for i,v in enumerate(rep.values):
    plt.text(i, v/1e6+0.2, f'Rs.{v/1e6:.1f}M', ha='center', fontsize=10)
plt.tight_layout()
save('05_salesrep_performance.png')

# Chart 6 - Category Pie
cat = df.groupby('Category')['Revenue'].sum()
plt.figure(figsize=(7,7))
plt.pie(cat.values, labels=cat.index, autopct='%1.1f%%',
        colors=sns.color_palette('pastel'), startangle=140)
plt.title('Revenue by Category')
plt.tight_layout()
save('06_category_pie.png')

# Chart 7 - Quarterly
df['Year']    = df['Date'].dt.year
df['Quarter'] = df['Date'].dt.quarter
q = df.groupby(['Year','Quarter'])['Revenue'].sum().reset_index()
q['Label'] = q['Year'].astype(str)+' Q'+q['Quarter'].astype(str)
plt.figure(figsize=(10,5))
sns.barplot(x='Label', y='Revenue', data=q, palette='coolwarm')
plt.title('Quarterly Revenue')
plt.ylabel('Revenue (Rs.)')
plt.xticks(rotation=30)
plt.tight_layout()
save('07_quarterly_revenue.png')

# Chart 8 - Heatmap
pivot = df.pivot_table(values='Revenue', index='Region',
                       columns='Quarter', aggfunc='sum')
plt.figure(figsize=(9,5))
sns.heatmap(pivot/1e6, annot=True, fmt='.1f', cmap='YlOrRd')
plt.title('Revenue Heatmap: Region vs Quarter (Rs. Millions)')
plt.tight_layout()
save('08_heatmap.png')

# Chart 9 - Units Histogram
plt.figure(figsize=(9,5))
plt.hist(df['Units_Sold'], bins=20, color='#7E57C2', edgecolor='white')
plt.title('Units Sold Distribution')
plt.xlabel('Units Sold')
plt.ylabel('Frequency')
plt.tight_layout()
save('09_units_distribution.png')

# Chart 10 - Forecast
monthly2 = df.groupby('Month_Year')['Revenue'].sum().reset_index()
monthly2['Month_Index'] = range(1, len(monthly2)+1)
X = monthly2[['Month_Index']].values
y = monthly2['Revenue'].values
model = LinearRegression()
model.fit(X, y)
future_idx  = np.array([[len(monthly2)+i] for i in range(1,7)])
future_pred = model.predict(future_idx)
future_labels = pd.date_range(
    start=monthly2['Month_Year'].iloc[-1].to_timestamp()+pd.DateOffset(months=1),
    periods=6, freq='MS').strftime('%b %Y')
monthly2['Label'] = monthly2['Month_Year'].astype(str)
plt.figure(figsize=(14,5))
plt.plot(monthly2['Label'], monthly2['Revenue']/1e6,
         marker='o', label='Actual', color='#2196F3')
fx = list(monthly2['Label'].iloc[-1:]) + list(future_labels)
fy = [monthly2['Revenue'].iloc[-1]/1e6]+[v/1e6 for v in future_pred]
plt.plot(fx, fy, marker='s', linestyle='--',
         label='Forecast (next 6 months)', color='#E53935')
plt.title('Sales Forecast - Next 6 Months')
plt.ylabel('Revenue (Rs. Millions)')
plt.xlabel('Month')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
save('10_forecast.png')

print('\nAll 10 charts saved in fig/ folder!')