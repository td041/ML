import pandas as pd 
# from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

data= pd.read_csv("/home/td041/Python/ML/datasets/StudentScore/StudentScore.xls")
# profile = ProfileReport(data, title="Student Score", explorative=True)
# profile.to_file("/home/td041/Python/ML/src/student_score/score.html")

target = "writing score"

x = data.drop(target, axis=1) # axix = 1 : cột, axis = 0 hàng 
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Numerical : reading score, math score 
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

education_values= ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", 
                   "master's degree"]
gender = ['male', 'female']
lunch = x_train["lunch"].unique()
race = x_train["race/ethnicity"].unique()
test_values = x_train["test preparation course"].unique()

# Ordinal: parental level of education, gender, lunch, test preparation course
ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[education_values, gender, lunch, test_values]))
])

# Nominal: race/ethnicity
nom_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])

preprocessor = ColumnTransformer(transformers=[
    ('num_features', num_transformer, ["reading score", "math score"]),
    ('ord_features', ordinal_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ('nom_features', nom_transformer, ["race/ethnicity"])
])

reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

print(reg.score(x_test, y_test))


for i,j in zip(y_pred, y_test):
    print("Predicted value: {}, Actual value: {}".format(i,j))