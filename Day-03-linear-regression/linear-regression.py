from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('student-scores.csv')

# Viewing information about the dataset
#print(df.head())
#print(df.tail())

#print(df.isnull().sum())

df.drop_duplicates(inplace=True)

# Calculating the total score
df['total_score'] = df[['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']].sum(axis=1)

# dropping irrelevant columns
df.drop(columns=['id', 'first_name', 'last_name', 'email', 'career_aspiration', 'gender', 'part_time_job', 'extracurricular_activities'], inplace=True)

df.drop(columns=['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score'], inplace=True)

#df.replace({'True': 1, 'False': 0}, inplace=True)
#df['part_time_job'] = df['part_time_job'].replace({'True': 1, 'False': 0})

# describe the dataset
print(df.info())

print(df.head())

# Matplotlib
plt.title('Student Scores')
plt.xlabel('Study Hours (Weekly)')
plt.ylabel('Total Score')
plt.scatter(df['weekly_self_study_hours'], df['total_score'], color='blue', label='Weekly Self Study Hours')

# Splitting the dataset into features and label variable
X = df[['weekly_self_study_hours', 'absence_days']]
y = df['total_score']

# Creating the model
model = LinearRegression()
# Fitting the model
model.fit(X, y)

coefficients = model.coef_
intercept = model.intercept_
print(f'Coefficients: {coefficients}')
print(f'Intercept: {intercept}')


# Calculating the regression line
x_min = df['weekly_self_study_hours'].min()
x_max = df['weekly_self_study_hours'].max()
y_min = (x_min * coefficients[0]) + (df['absence_days'].mean() * coefficients[1]) + intercept
y_max = (x_max * coefficients[0]) + (df['absence_days'].mean() * coefficients[1]) + intercept

# plotting the regression line
plt.plot([x_min, x_max], [y_min, y_max], color='red', label='Regression Line')
plt.legend()
plt.show()