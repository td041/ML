import pandas as pd
# from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from  sklearn.metrics import classification_report

data = pd.read_csv('datasets/Datasets/diabetes/diabetes.csv')
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

# Train Model
model = SVC()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

print(classification_report(y_test, y_predict))

