import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('AmesHousing.csv')

print(data.dtypes)

#num_col = data.select_dtypes(include=['number'])
for col in data.select_dtypes(include=['object', 'bool']).columns:
    data[col] = LabelEncoder().fit_transform(data[col])

corr_matrix = data.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(data=corr_matrix, annot=False, linewidths=1.0, vmin=-1.0, vmax=1.0)
plt.title("Feature Correlation Heatmap")
plt.show()