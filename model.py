"""
@authors Madeleine Comtois and Ciara Gilsenan
@version 2/12/2020
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, mean_squared_error
from extract_features import get_feature_matrix

feature_matrix = get_feature_matrix()
print(feature_matrix)