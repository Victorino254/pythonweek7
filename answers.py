# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Set file path 
file_path = 'Iris.csv'  

# Load the dataset
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()

# Display the first few rows
print("First few rows of the dataset:")
print(df.head())

# Check the structure of the dataset
print("\nDataset Information:")
print(df.info())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Display summary statistics for numerical columns
print("\nSummary statistics:")
print(df.describe())

# Group data by 'Species' and compute the mean for numerical columns
grouped_means = df.groupby('Species').mean()
print("\nMean values grouped by Species:")
print(grouped_means)

# Fix for correlation matrix
numerical_df = df.select_dtypes(include=['float64', 'int64'])  # Select only numerical columns
correlation_matrix = numerical_df.corr()
print("\nCorrelation matrix:")
print(correlation_matrix)

# Line Chart: Comparing Petal Length Across Rows
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['PetalLengthCm'], label='Petal Length', color='blue')
plt.title('Petal Length Across Rows')
plt.xlabel('Index')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.show()

# Bar Chart: Average Sepal Width by Species
species_avg_sepal_width = df.groupby('Species')['SepalWidthCm'].mean()
plt.figure(figsize=(10, 6))
species_avg_sepal_width.plot(kind='bar', color='green')
plt.title('Average Sepal Width by Species')
plt.xlabel('Species')
plt.ylabel('Average Sepal Width (cm)')
plt.xticks(rotation=45)
plt.show()

# Histogram: Distribution of Sepal Length
plt.figure(figsize=(10, 6))
plt.hist(df['SepalLengthCm'], bins=20, edgecolor='black', color='orange')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot: Sepal Length vs. Petal Length
plt.figure(figsize=(10, 6))
plt.scatter(df['SepalLengthCm'], df['PetalLengthCm'], color='purple', alpha=0.7)
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()
