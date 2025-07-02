import pandas as pd

df = pd.read_csv('titanic_dataset.csv')

#print(df.isnull().sum())

# Removing irrelevant columns
df.drop(['PassengerId'], inplace=True, axis=1)
df.drop(['Name'], inplace=True, axis=1)
df.drop(['Cabin'], inplace=True, axis=1)

# Filling missing age(s) with mean age
df['Age'] = df['Age'].fillna(df['Age'].mean())

# replacing symbols for Embarked Port with names of city
df['Embarked'] = df['Embarked'].replace({'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'})
df.dropna(subset=['Embarked'], inplace=True)

# dropping duplicate records
df.drop_duplicates(inplace=True)

#print(df.head())
#print(df.tail())

#print(df.isnull().sum())
# Saving the cleaned data to a new CSV file
df.to_csv('titanic_dataset_cleaned.csv', index=False)
print("Cleaned data saved to 'titanic_dataset_cleaned.csv'")