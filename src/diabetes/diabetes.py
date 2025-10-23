import pandas as pd
# from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from  sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from lazypredict.Supervised import LazyClassifier



data = pd.read_csv('/home/td041/Python/ML/datasets/diabetes/diabetes.csv')
print(data.head())
# profile = ProfileReport(data, title="Diabetes Dataset Report", explorative=True)
# profile.to_file('diabetes_dataset_report.html')

# Data Split
target = 'Outcome'
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Data processing
scaler = StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

params= {
    "n_estimators": [100, 200, 300],
    "criterion": ["gini", "entropy", "log_loss"],

}

# Train Model
# grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=params, cv=5, verbose=2, scoring="recall")
# grid_search.fit(x_train, y_train)

# y_predict = grid_search.predict(x_test)
# print(classification_report(y_test, y_predict))

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
print(models)