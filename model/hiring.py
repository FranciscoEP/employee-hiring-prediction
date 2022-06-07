import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.stats import diagnostic

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pickle


def boolean_parser(row, first_pattern, second_pattern):
    if row == first_pattern:
        return 1
    elif row == second_pattern:
        return 0
    else:
        row

def downscale(score):
    return score/10/2

def clean_dataset(data):
    data_cleaned = data[['gender', 'degree_p', 'degree_t', 'workex','etest_p', 'specialisation', 'mba_p', 'status']]
    data_cleaned['degree_t'] = data_cleaned['degree_t'].replace({"Comm&Mgmt":1, "Sci&Tech":2,"Others":3})
    data_cleaned['specialisation'] = data_cleaned['specialisation'].replace({"Mkt&Fin":1, "Mkt&HR":2})
    data_cleaned['gender'] = data_cleaned['gender'].apply(lambda x: boolean_parser(x, 'M', 'F'))
    data_cleaned['workex'] = data_cleaned['workex'].apply(lambda x: boolean_parser(x, 'Yes', 'No'))
    data_cleaned['status'] = data_cleaned['status'].apply(lambda x: boolean_parser(x, 'Placed', 'Not Placed'))
    data_cleaned[['degree_p']] = data_cleaned[['degree_p']].apply(downscale)
    data_cleaned[['mba_p']] = data_cleaned[['mba_p']].apply(downscale)
    return data_cleaned

def model_creation(data):
    X = data.drop(['status', 'degree_t', 'specialisation'], axis=1)
    y = data['status']
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train,y_train)
    pickle.dump(model, open('applicants_hireable.pkl','wb'))

def run(): 
    # Load the dataset
    applicants = pd.read_csv('employees.csv')
    applicants_cleaned = clean_dataset(applicants)
    applicants_model = model_creation(applicants_cleaned)


if __name__ == '__main__':
    run()