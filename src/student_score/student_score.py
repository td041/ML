import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
# from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

data = pd.read_csv("/home/td041/Python/ML/datasets/StudentScore/StudentScore.xls")
# profile = ProfileReport(data, title="Student Score", explorative=True)
# profile.to_file("/home/td041/Python/ML/src/student_score/score.html")

target = "writing score"

x = data.drop(target, axis=1)  # axix = 1 : cột, axis = 0 hàng
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Numerical : reading score, math score 
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

education_values = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree",
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
    ('ord_features', ordinal_transformer,
     ["parental level of education", "gender", "lunch", "test preparation course"]),
    ('nom_features', nom_transformer, ["race/ethnicity"])
])

reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

params = {
    'preprocessor__num_features__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'regressor__n_estimators': [100, 200, 300],
    'regressor__criterion': ['squared_error', 'absolute_error', 'poisson'],
    'regressor__max_depth': [None, 2, 5],

}

# clf = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = clf.fit(x_train, x_test, y_train, y_test)
# print(models)

grid_search = RandomizedSearchCV(estimator=reg, param_distributions=params, cv=5, scoring='r2', verbose=2, n_jobs=-1,
                                 n_iter=20)
grid_search.fit(x_train, y_train)

print("Best score: ", grid_search.best_score_)
print("Best parameters: ", grid_search.best_params_)

y_pred = grid_search.predict(x_test)

print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
print("R2 Score: ", r2_score(y_test, y_pred))

# for i,j in zip(y_pred, y_test):
#     print("Predicted value: {}, Actual value: {}".format(i,j))
