import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('k12_students_data.csv')

# 1. Summary Statistics
print(df.describe())  # Summary of numerical features
print(df['Gender'].value_counts())  # Frequency of categorical data (e.g., gender)

# 2. Missing Data Analysis
print(df.isnull().sum())  # Count missing values per column

# 3. Visualization: Histogram of Assessment Scores
sns.histplot(df['Assessment Score'], kde=True)
plt.title('Assessment Score Distribution')
plt.show()

# 4. Boxplot for IQ scores by Promotion Status
sns.boxplot(x='Promoted', y='IQ of Student', data=df)
plt.title('IQ Scores by Promotion Status')
plt.show()

# 5. Correlation Heatmap
corr_matrix = df.select_dtypes(include='number').corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=1, linecolor='black')
plt.title('Correlation Matrix')
plt.show()

# 6. Pairplot for Numerical Relationships
sns.pairplot(df[['Assessment Score', 'IQ of Student', 'Time per Day (min)']])
plt.suptitle('Pairplot of Assessment Score, IQ, and Time Spent', y=1.02)
plt.show()
print(df.select_dtypes(include='number').columns)
