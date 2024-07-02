import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('car_sales.csv')

# Display the first few rows of the dataset
print(df.head())

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Add a new column for the year and month
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Check for any missing values
print(df.isnull().sum())

# Descriptive statistics
print(df.describe())

# Total sales per car model
total_sales_per_model = df.groupby('Model')['Total_Sale'].sum().reset_index()
print(total_sales_per_model)

# Plot total sales per car model
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Total_Sale', data=total_sales_per_model)
plt.title('Total Sales per Car Model')
plt.xlabel('Car Model')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()

# Monthly sales trend
monthly_sales = df.groupby(['Year', 'Month'])['Total_Sale'].sum().reset_index()
print(monthly_sales)

# Plot monthly sales trend
plt.figure(figsize=(14, 7))
sns.lineplot(x='Month', y='Total_Sale', hue='Year', data=monthly_sales, marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.show()

# Distribution of car prices
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], bins=20, kde=True)
plt.title('Distribution of Car Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Sales quantity per model
sales_quantity_per_model = df.groupby('Model')['Quantity'].sum().reset_index()
print(sales_quantity_per_model)

# Plot sales quantity per model
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Quantity', data=sales_quantity_per_model)
plt.title('Sales Quantity per Car Model')
plt.xlabel('Car Model')
plt.ylabel('Quantity Sold')
plt.xticks(rotation=45)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
