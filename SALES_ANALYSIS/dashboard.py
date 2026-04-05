import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Sales Dashboard', page_icon='📊', layout='wide')
st.title('📊 Sales Data Analysis and Forecasting')
st.markdown('---')

df = pd.read_csv('sales_data_clean.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Month_Year'] = df['Date'].dt.to_period('M')
df['Year']    = df['Date'].dt.year
df['Month']   = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter

st.success(f'✅ Data loaded successfully — {len(df)} rows')

# ── Sidebar ──────────────────────────────────
st.sidebar.header('🔍 Filters')
region  = st.sidebar.multiselect('Region',
    sorted(df['Region'].unique()),
    default=sorted(df['Region'].unique()))
product = st.sidebar.multiselect('Product',
    sorted(df['Product'].unique()),
    default=sorted(df['Product'].unique()))
year = st.sidebar.multiselect('Year',
    sorted(df['Year'].unique()),
    default=sorted(df['Year'].unique()))

fdf = df[
    df['Region'].isin(region) &
    df['Product'].isin(product) &
    df['Year'].isin(year)
]

# ── KPI Metrics ──────────────────────────────
st.subheader('📌 Key Metrics')
k1,k2,k3,k4 = st.columns(4)
k1.metric('💰 Total Revenue',   f"Rs.{fdf['Revenue'].sum()/1e6:.2f}M")
k2.metric('📈 Total Profit',    f"Rs.{fdf['Profit'].sum()/1e6:.2f}M")
k3.metric('📦 Units Sold',      f"{int(fdf['Units_Sold'].sum()):,}")
k4.metric('🛒 Avg Order Value', f"Rs.{fdf['Revenue'].mean():,.0f}")

st.markdown('---')

# ── Chart 1 & 2 ──────────────────────────────
st.subheader('📊 Revenue Analysis')
c1, c2 = st.columns(2)

with c1:
    st.markdown('**Monthly Revenue Trend**')
    m = fdf.groupby('Month_Year')['Revenue'].sum().reset_index()
    m['Label'] = m['Month_Year'].astype(str)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(m['Label'], m['Revenue']/1e6, marker='o', color='#2196F3', linewidth=2)
    ax.fill_between(m['Label'], m['Revenue']/1e6, alpha=0.1, color='#2196F3')
    ax.set_ylabel('Revenue (Rs. Millions)')
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with c2:
    st.markdown('**Revenue by Region**')
    r = fdf.groupby('Region')['Revenue'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8,4))
    bars = ax.bar(r.index, r.values/1e6,
                  color=['#42A5F5','#66BB6A','#FF7043','#AB47BC','#FFA726'])
    ax.set_ylabel('Revenue (Rs. Millions)')
    for bar, val in zip(bars, r.values):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.2,
                f'Rs.{val/1e6:.1f}M', ha='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Chart 3 & 4 ──────────────────────────────
c3, c4 = st.columns(2)

with c3:
    st.markdown('**Revenue by Product**')
    p = fdf.groupby('Product')['Revenue'].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(p.index, p.values/1e6, color='#26C6DA')
    ax.set_xlabel('Revenue (Rs. Millions)')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with c4:
    st.markdown('**Sales Rep Performance**')
    s = fdf.groupby('Sales_Rep')['Revenue'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8,4))
    colors = ['#FF7043','#42A5F5','#66BB6A','#AB47BC','#FFA726']
    ax.bar(s.index, s.values/1e6, color=colors[:len(s)])
    ax.set_ylabel('Revenue (Rs. Millions)')
    for i,v in enumerate(s.values):
        ax.text(i, v/1e6+0.1, f'Rs.{v/1e6:.1f}M', ha='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown('---')

# ── Chart 5 & 6 ──────────────────────────────
c5, c6 = st.columns(2)

with c5:
    st.markdown('**Quarterly Revenue**')
    q = fdf.groupby(['Year','Quarter'])['Revenue'].sum().reset_index()
    q['Label'] = q['Year'].astype(str)+' Q'+q['Quarter'].astype(str)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(q['Label'], q['Revenue']/1e6,
           color=['#42A5F5','#66BB6A','#FF7043','#AB47BC']*3)
    ax.set_ylabel('Revenue (Rs. Millions)')
    plt.xticks(rotation=30, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with c6:
    st.markdown('**Revenue by Category**')
    cat = fdf.groupby('Category')['Revenue'].sum()
    fig, ax = plt.subplots(figsize=(7,4))
    ax.pie(cat.values, labels=cat.index,
           autopct='%1.1f%%',
           colors=sns.color_palette('pastel'),
           startangle=140)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown('---')

# ── Revenue vs Profit ─────────────────────────
st.markdown('**Monthly Revenue vs Profit**')
mon = fdf.groupby('Month_Year')[['Revenue','Profit']].sum().reset_index()
mon['Label'] = mon['Month_Year'].astype(str)
x = range(len(mon))
w = 0.4
fig, ax = plt.subplots(figsize=(14,4))
ax.bar([i-w/2 for i in x], mon['Revenue']/1e6, w,
       label='Revenue', color='#42A5F5')
ax.bar([i+w/2 for i in x], mon['Profit']/1e6, w,
       label='Profit', color='#66BB6A')
ax.set_xticks(list(x))
ax.set_xticklabels(mon['Label'], rotation=45, fontsize=8)
ax.set_ylabel('Rs. Millions')
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown('---')

# ── Heatmap ───────────────────────────────────
st.markdown('**Revenue Heatmap — Region vs Quarter**')
pivot = fdf.pivot_table(values='Revenue',
                        index='Region',
                        columns='Quarter',
                        aggfunc='sum')
fig, ax = plt.subplots(figsize=(10,4))
sns.heatmap(pivot/1e6, annot=True, fmt='.1f',
            cmap='YlOrRd', ax=ax)
ax.set_title('Revenue (Rs. Millions)')
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown('---')

# ── Forecasting ───────────────────────────────
st.subheader('📈 Sales Forecast — Next 6 Months')
m2 = df.groupby('Month_Year')['Revenue'].sum().reset_index()
m2['Month_Index'] = range(1, len(m2)+1)
model = LinearRegression()
model.fit(m2[['Month_Index']].values, m2['Revenue'].values)

fi = np.array([[len(m2)+i] for i in range(1,7)])
fp = model.predict(fi)
fl = pd.date_range(
    start=m2['Month_Year'].iloc[-1].to_timestamp()+pd.DateOffset(months=1),
    periods=6, freq='MS').strftime('%b %Y')

m2['Label'] = m2['Month_Year'].astype(str)
fig, ax = plt.subplots(figsize=(14,5))
ax.plot(m2['Label'], m2['Revenue']/1e6,
        marker='o', label='Actual', color='#2196F3', linewidth=2)
fx2 = list(m2['Label'].iloc[-1:]) + list(fl)
fy2 = [m2['Revenue'].iloc[-1]/1e6]+[v/1e6 for v in fp]
ax.plot(fx2, fy2, marker='s', linestyle='--',
        label='Forecast', color='#E53935', linewidth=2)
ax.set_ylabel('Revenue (Rs. Millions)')
ax.set_xlabel('Month')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
plt.close()

fc = pd.DataFrame({
    'Month': list(fl),
    'Forecasted Revenue': [f'Rs.{v:,.0f}' for v in fp]
})
st.dataframe(fc, use_container_width=True)

st.markdown('---')
st.subheader('📋 Raw Data')
st.dataframe(fdf.reset_index(drop=True), use_container_width=True)
st.caption(f'Total rows: {len(fdf):,} | Model: Linear Regression')