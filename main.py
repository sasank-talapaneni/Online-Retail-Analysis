import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load data
customers = pd.read_csv('customers.csv')
products = pd.read_csv('products.csv')
transactions = pd.read_csv('transactions.csv')

df = pd.merge(transactions, customers, on='CustomerID')
df = pd.merge(df, products, on='ProductID')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Visualizations :
# Monthly sales trends
monthly_sales = df.resample('ME', on='Timestamp')['Quantity'].sum()
plt.figure(figsize=(10, 6))
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Total Quantity Sold')
plt.grid(True)
plt.show()

# Top-selling products by category
top_products_by_category = df.groupby(['Category', 'Product'])['Quantity'].sum().reset_index()
top_products_by_category = top_products_by_category.sort_values(by='Quantity', ascending=False)
plt.figure(figsize=(12, 8))
for category in top_products_by_category['Category'].unique():
    data = top_products_by_category[top_products_by_category['Category'] == category].head(2)
    plt.barh(data['Product'], data['Quantity'], label=category)
plt.xlabel('Total Quantity Sold')
plt.ylabel('Product')
plt.title('Top-Selling Products by Category')
plt.legend()
plt.show()

# Top-selling products overall
top_products = df.groupby('Product')['Quantity'].sum().reset_index()
top_products = top_products.sort_values(by='Quantity', ascending=False).head(5)
plt.figure(figsize=(10, 6))
plt.barh(top_products['Product'], top_products['Quantity'], color='skyblue')
plt.xlabel('Total Quantity Sold')
plt.ylabel('Product')
plt.title('Top-Selling Products')
plt.gca().invert_yaxis()
plt.show()

# Analysis
# Print top-selling products overall
print("Top-Selling Products:")
print(top_products)

# Print top-selling products by category
top_products_by_category = df.groupby(['Category', 'Product'])['Quantity'].sum().reset_index()
top_products_by_category = top_products_by_category.sort_values(by=['Category', 'Quantity'], ascending=[True, False])
print("Top-Selling Products by Category:")
for category, data in top_products_by_category.groupby('Category'):
    print(f"\nCategory: {category}")
    print(data.head(3))

# Customer Segmentation

X = df.groupby('CustomerID')['Quantity'].sum().reset_index()
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['Quantity']])

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
X['Cluster'] = kmeans.fit_predict(X_scaled)
# Plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=X, x='CustomerID', y='Quantity', hue='Cluster', palette='Set1', legend='full')
plt.title('Customer Segmentation based on Purchasing Behavior')
plt.xlabel('Customer ID')
plt.ylabel('Total Quantity Purchased')
plt.show()

# Predictive Analysis

# Calculate monthly sales
monthly_sales = df.resample('ME', on='Timestamp')['Quantity'].sum()
monthly_sales = monthly_sales.reset_index()
# Convert 'Timestamp' to months
monthly_sales['Months'] = (monthly_sales['Timestamp'].dt.year - monthly_sales['Timestamp'].dt.year.min()) * 12 + \
                           (monthly_sales['Timestamp'].dt.month - monthly_sales['Timestamp'].dt.month.min())

# Split data into features and target
X = monthly_sales[['Months']]
y = monthly_sales['Quantity']
# Train linear regression model
model = LinearRegression()
model.fit(X, y)
# Predict future sales
future_months = np.arange(X['Months'].max() + 1, X['Months'].max() + 13).reshape(-1, 1)
future_sales = model.predict(future_months)
# Plot actual and predicted sales trends
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression line')
plt.plot(future_months, future_sales, color='green', linestyle='--', label='Predicted')
plt.title('Monthly Sales Trends')
plt.xlabel('Months')
plt.ylabel('Total Sales')
plt.legend()
plt.grid(True)
plt.show()
