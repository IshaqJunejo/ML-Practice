import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the Dataset
data = pd.read_csv('../Day-03/student-scores.csv')

data['total_score'] = data['math_score'] + data['history_score'] + data['physics_score'] + data['chemistry_score'] + data['biology_score'] + data['english_score'] + data['geography_score']

data = data.drop(columns=['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score'])

data = data.drop(columns=['id', 'first_name', 'last_name', 'email'])

#print(data.head())
#print(data.describe())
#print(data.isnull().sum())
print(data.dtypes)
#print(data.shape)
#print(data.columns)
#print(data.isnull().sum())

data['career_aspiration'] = LabelEncoder().fit_transform(data['career_aspiration'])
data['extracurricular_activities'] = LabelEncoder().fit_transform(data['extracurricular_activities'])
data['part_time_job'] = LabelEncoder().fit_transform(data['part_time_job'])

sns.histplot(data['total_score'], kde=True)
plt.title("Score of Students")
plt.show()

#sns.scatterplot(x='weekly_self_study_hours', y='absence_days', hue='total_score', data=data)
#plt.show()

#sns.pairplot(data=data, vars=['weekly_self_study_hours', 'absence_days', 'career_aspiration', 'part_time_job', 'extracurricular_activities', 'total_score'], hue='gender')
#plt.show()