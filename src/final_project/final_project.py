import re

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTEN


def filter_location(location):
    filtering = re.findall(r",\s[A-Z]{2}$", location)
    if len(filtering) > 0:
        return filtering[0][2:]
    return location


data = pd.read_excel('/home/td041/Python/ML/datasets/final_project/final_project.ods', engine='odf', dtype=str)
data = data.dropna(axis=0)
data["location"] = data["location"].apply(filter_location)

target = "career_level"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)

ros = SMOTEN(random_state=42, k_neighbors=2, sampling_strategy={
    'director_business_unit_leader': 500,
    'specialist': 500,
    'managing_director_small_medium_company': 500,
    'bereichsleiter': 1000
})

x_train, y_train = ros.fit_resample(x_train, y_train)

preprocessor = ColumnTransformer(transformers=[
    ('title_ft', TfidfVectorizer(stop_words='english', ngram_range=(1, 1)), 'title'),
    ('description_ft', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=0.01, max_df=0.95),
     'description'),
    ('industry_tf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2)), 'industry'),
    ('location_ft', OneHotEncoder(handle_unknown="ignore"), ['location']),
    ('function_ft', OneHotEncoder(handle_unknown="ignore"), ['function']),
])

cls = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selector', SelectPercentile(chi2, percentile=5)),
    ('model', RandomForestClassifier())
])

# result = cls.fit_transform(x_train, y_train)
# print(result.shape)

params = {
    # 'model_n_estimators': [100, 200, 300],
    'model__criterion': ['gini', 'entropy', 'log_loss'],
    'feature_selector__percentile': [1, 5, 10]
}

grid_search = GridSearchCV(estimator=cls, param_grid=params, cv=4, scoring='recall_weighted', verbose=2)
grid_search.fit(x_train, y_train)
y_predicted = grid_search.predict(x_test)
print(classification_report(y_test, y_predicted))
