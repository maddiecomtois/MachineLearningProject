"""
@authors Madeleine Comtois and Ciara Gilsenan
@version 2/12/2020
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score, roc_curve
from sklearn.dummy import DummyClassifier
from extract_features import get_feature_matrix
import matplotlib.pyplot as plt
import numpy as np


# Normalize Data???
def normalize(X1, X2):
    # x1
    s = max(X1) - min(X1)
    for i in range(len(X1)):
        X1[i] = (X1[i] - min(X1)) / s

    # x2
    s = max(X2) - min(X2)
    for i in range(len(X2)):
        X2[i] = (X2[i] - min(X2)) / s

    print("Norm")
    print(np.column_stack((X1, X2, X3)))
    return np.column_stack((X1, X2, X3))


# Get training and testing data
feature_matrix = get_feature_matrix()
X1 = np.array(feature_matrix[:, 0:1])
X2 = np.array(feature_matrix[:, 1:2])
X3 = np.array(feature_matrix[:, 2:len(feature_matrix[0])-1])
x = np.column_stack((X1, X2, X3))
print("Not Norm")
print(x)
y = feature_matrix[:, len(feature_matrix[0])-1:].flatten()

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, stratify=None)

"""
# Graph Data
plt.title("initial data")
plt.xlabel('formal')
plt.ylabel('familiar')
colours = ['red' if y_train[i] == -1 else 'green' for i in range(len(y_train))]
plt.scatter(x, y, label='Training', c=colours, marker='+')
plt.legend()
# plt.show()
"""

x = normalize(X1, X2, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, stratify=None)


# Cross validation for hyperparameters gamma and C for kernalised svm
kf = KFold(n_splits=5)
gammas = [1, 5, 10, 25]
c_values = [0.1, 1, 10, 50, 100]
mse_vals = []
std_vals = []
for g in gammas:
    for c in c_values:
        hyp_pair = (g, c)
        tmp_errors = []
        for train, test in kf.split(x_train):
            model = SVC(kernel='rbf', gamma=g, C=c)
            model.fit(x_train[train], y_train[train])
            ypred = model.predict(x_train[test])
            tmp_errors.append(mean_squared_error(y_train[test], ypred))
        mse_vals.append((hyp_pair, np.mean(tmp_errors)))
        std_vals.append((hyp_pair, np.std(tmp_errors)))

# ax = plt.figure().add_subplot(projection='3d')
# print("\nMSE CV for hyperparameter pairs:\n", mse_vals)
g, c = min(mse_vals, key=lambda x: x[1])[0]
print("Min MSE pair:", (g, c))


# Train Kernelised SVM with chosen gamma and C
model2 = SVC(kernel='rbf', gamma=g, C=c)
model2.fit(x_train, y_train)
predictions = model2.predict(x_test)
predictions = predictions.reshape(-1, 1)

# Evalutation
print("\nMSE Kernel_SVC on test data:", mean_squared_error(y_test, predictions))
print("ACC Kernel_SVC on test data:", accuracy_score(y_test, predictions))
print("Confusion Matrix Kernel_SVC:\n", confusion_matrix(y_test, predictions))

# Plot ROC Curve
plt.cla()
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(x_train, y_train)
dummy_scores = dummy.predict_proba(x_test)
fpr, tpr, _ = roc_curve(y_test, dummy_scores[:, 1])
plt.plot(fpr, tpr, color='red', linestyle='--', label='Baseline')

fpr, tpr, _ = roc_curve(y_test, model2.decision_function(x_test))
plt.plot(fpr, tpr, label="SVC")

plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Other CV method --- Able to create graphs
mse_vals = []
std_vals = []
for g in gammas:
    tmp_errors = []
    for train, test in kf.split(x_train):
        model = SVC(kernel='rbf', gamma=g)
        model.fit(x_train[train], y_train[train])
        ypred = model.predict(x_train[test])
        tmp_errors.append(mean_squared_error(y_train[test], ypred))
    mse_vals.append(np.mean(tmp_errors))
    std_vals.append(np.std(tmp_errors))

print(mse_vals)
label = "CV For Gamma"
plt.cla()
plt.title(label)
plt.xlabel("gamma")
plt.ylabel("MSE, STD")
plt.errorbar(gammas, mse_vals, std_vals, ecolor='red')
plt.show()

mse_vals = []
std_vals = []
for c in c_values:
    tmp_errors = []
    for train, test in kf.split(x_train):
        model = SVC(kernel='rbf', gamma=1, C=c)
        model.fit(x_train[train], y_train[train])
        ypred = model.predict(x_train[test])
        tmp_errors.append(mean_squared_error(y_train[test], ypred))
    mse_vals.append(np.mean(tmp_errors))
    std_vals.append(np.std(tmp_errors))

print(mse_vals)
plt.cla()
label = "CV For C"
plt.cla()
plt.title(label)
plt.xlabel("C")
plt.ylabel("MSE, STD")
plt.errorbar(c_values, mse_vals, std_vals, ecolor='red')
plt.show()