import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_extraction import DictVectorizer
def read_iris_data():
    iris = load_iris()
    data = pd.DataFrame(iris.data[:], columns=iris.feature_names)
    data['class'] = [iris.target_names[i] for i in iris.target]
    y_column='class'
    x_columns=iris.feature_names
    return data,x_columns,y_column
def read_winery_data():
    x_columns=['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium',
                                     'Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins',
                                    'Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
    y_column='class'
    data = pd.read_csv("datasets/wine.data",names=[y_column]+x_columns)
    return data, x_columns, y_column
def read_breast_cancer_data():
    x_columns = ['code_number', 'Clump_thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
             'Marginal Adhesion', 'Single Epithelial Cell Size'
        , 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
    y_column='class'
    data = pd.read_csv('datasets/breast-cancer-wisconsin.data',names=x_columns+[y_column])
    data = data[data['Bare Nuclei'] != '?']
    data['Bare Nuclei'] = [int(i) for i in data['Bare Nuclei']]
    return data, x_columns, y_column
def read_tic_tac_toe_dataset():
    x_columns=['top - left - square','top - middle - square','top - right - square','middle - left - square',
               'middle - middle - square','middle - right - square','bottom - left - square','bottom - middle - square','bottom - right - square']
    y_column='class'
    data=pd.read_csv('datasets/tic-tac-toe.data',names=x_columns+[y_column])
    dv=DictVectorizer()
    dv_data=dv.fit_transform([dict(row) for index,row in data[x_columns].iterrows()])
    dv_data=pd.DataFrame(dv_data.toarray(),columns=dv.feature_names_)
    dv_data[y_column]=data[y_column]
    data=dv_data
    return data, dv.feature_names_, y_column
def read_australian():
    x_columns = ["A" + str(i) for i in range(14)]
    y_column='class'
    data = pd.read_csv("datasets/australian.dat", sep=" ", names=x_columns+['class'])
    return data,x_columns,y_column
def read_nurse():
    x_columns = ['x' + str(i) for i in range(8)]
    y_column='class'
    data = pd.read_csv("datasets/post-operative.data", names=x_columns+[y_column])
    dv = DictVectorizer()
    dv_data = dv.fit_transform([dict(row) for index, row in data[x_columns].iterrows()])
    dv_data = pd.DataFrame(dv_data.toarray(), columns=dv.feature_names_)
    dv_data[y_column] = data[y_column]
    data = dv_data
    return data, dv.feature_names_, y_column
def get_dataset_by_string(s):
    if s=='iris':
        return read_iris_data()
    elif s=='winery':
        return read_winery_data()
    elif s=='breast cancer':
        return read_breast_cancer_data()
    elif s == 'aust_credit':
        return read_australian()
    elif s == 'nurse':
        return read_nurse()
    elif s=='tic-tac-toe':
        return read_tic_tac_toe_dataset()