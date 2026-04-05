import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000
regions   = ['North','South','East','West','Central']
products  = ['Product A','Product B','Product C','Product D','Product E']
salesreps = ['Alice','Bob','Charlie','Diana','Evan']
categories= ['Electronics','Clothing','Furniture','Kitchen','Sports']
dates = pd.date_range('2023-01-01','2024-12-31',freq='D')

df = pd.DataFrame({
    'Date':       pd.to_datetime(np.random.choice(dates,n)),
    'Region':     np.random.choice(regions,n),
    'Product':    np.random.choice(products,n),
    'Category':   np.random.choice(categories,n),
    'Sales_Rep':  np.random.choice(salesreps,n),
    'Units_Sold': np.random.randint(1,50,n),
    'Unit_Price': np.round(np.random.uniform(100,5000,n),2),
    'Discount_Pct': np.round(np.random.uniform(0,30,n),1),
})
df['Revenue'] = np.round(df['Units_Sold'] * df['Unit_Price'],2)
df['Profit']  = np.round(df['Revenue'] * (1 - df['Discount_Pct']/100),2)
df.sort_values('Date',inplace=True)
df.reset_index(drop=True,inplace=True)
df.to_csv('sales_data.csv',index=False)
print('Done! sales_data.csv created with',len(df),'rows')
print(df.head())
