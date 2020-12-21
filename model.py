"""
@authors Madeleine Comtois and Ciara Gilsenan
@version 2/12/2020
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score, roc_curve, \
    precision_score, recall_score, auc
from sklearn.dummy import DummyClassifier
from extract_features import get_feature_matrix
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
# mpl.use('Qt5Agg')


# Normalize Data???
def normalize(X1, X2):
    """
    :param X1: Formal freq vector
    :param X2: Familiar freq vector
    :return: normalized np array
    """
    # x1
    s = max(X1) - min(X1)
    for i in range(len(X1)):
        X1[i] = (X1[i] - min(X1)) / s

    # x2
    s = max(X2) - min(X2)
    for i in range(len(X2)):
        X2[i] = (X2[i] - min(X2)) / s

    # print("Norm")
    normalized_data = np.column_stack((X1, X2, X3))
    # print(normalized_data)
    return normalized_data


def choose_gamma():
    gammas = [1, 5, 10, 25]
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

    g = gammas[mse_vals.index(min(mse_vals))]

    label = "Kernelized SVM Cross-Validation For Gamma"
    plt.cla()
    plt.title(label)
    plt.xlabel("gamma")
    plt.ylabel("Mean Square Error")
    plt.errorbar(gammas, mse_vals, std_vals, ecolor='red')
    plt.show()
    print(g, min(mse_vals))
    return g


# Get training and testing data
feature_matrix = get_feature_matrix()
X1 = np.array(feature_matrix[:, 0:1])
X2 = np.array(feature_matrix[:, 1:2])
X3 = np.array(feature_matrix[:, 2:len(feature_matrix[0])-1])
x = np.column_stack((X1, X2, X3))
y = feature_matrix[:, len(feature_matrix[0])-1:].flatten()
# print("Not Norm")
# print(x)

# Train - Test Split
x = normalize(X1, X2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, stratify=None)

# Graph Data 2d -- Use Function in Log branch
plt.title("initial data")
plt.xlabel('formal')
plt.ylabel('familiar')
plt.scatter(x_train[:, 0], x_train[:, 1], label='Training', c='red', marker='+')
plt.legend()
plt.show()

# Graph Data 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('Initial Data 3D')
ax.set_xlabel('Formal')
ax.set_ylabel('Familiar')
ax.set_zlabel('Label')
ax.scatter(x_train[:, 0], x_train[:, 1], y_train, color='purple', marker='+', label='training')
plt.legend()
plt.show()


# Cross validation for hyperparameters gamma and C for kernalised svm
# Other CV method --- Able to create graphs
kf = KFold(n_splits=5)
g = choose_gamma()
c_values = [0.1, 1, 10, 50, 100]
mse_vals = []
std_vals = []
for c in c_values:
    tmp_errors = []
    for train, test in kf.split(x_train):
        model = SVC(kernel='rbf', gamma=g, C=c)
        model.fit(x_train[train], y_train[train])
        ypred = model.predict(x_train[test])
        tmp_errors.append(mean_squared_error(y_train[test], ypred))
    mse_vals.append(np.mean(tmp_errors))
    std_vals.append(np.std(tmp_errors))

c = c_values[mse_vals.index(min(mse_vals))]
print(c, min(mse_vals))

plt.cla()
label = "Kernelized SVM Cross-Validation For C"
plt.cla()
plt.title(label)
plt.xlabel("C")
plt.ylabel("Mean Square Error")
plt.errorbar(c_values, mse_vals, std_vals, ecolor='red')
plt.show()


# Train Kernelised SVC with chosen gamma and C
SVC = SVC(kernel='rbf', gamma=g, C=c)
SVC.fit(x_train, y_train)
predictions = SVC.predict(x_test)
predictions = predictions.reshape(-1, 1)

print("Kernel_SVC Intercept:", SVC.intercept_)
# coef_ is only for linear kernels -- Gaussian is a curve

# Evaluation
print("\nEVALUATION OF KERNEL_SVC\nMSE on test data:", mean_squared_error(y_test, predictions))
print("Accuracy on test data:", accuracy_score(y_test, predictions))
print("Recall Score:", recall_score(y_test, predictions))
print("Precision Score:", precision_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

# Plot ROC Curve against logistic and baseline
plt.cla()
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(x_train, y_train)
dummy_scores = dummy.predict_proba(x_test)
fpr, tpr, _ = roc_curve(y_test, dummy_scores[:, 1])
plt.plot(fpr, tpr, color='red', linestyle='--', label='Baseline')

fpr, tpr, _ = roc_curve(y_test, SVC.decision_function(x_test))
plt.plot(fpr, tpr, label="SVC")
dummy_pred = dummy.predict(x_test)
print("\nDummy MSE:", mean_squared_error(y_test, dummy_pred))

print("AUC:", auc(fpr, tpr))

plt.title("ROC Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# Plot Predictions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('Initial Data 3D')
ax.set_xlabel('Formal')
ax.set_ylabel('Familiar')
ax.set_zlabel('Label')
ax.scatter(x_test[:, 0], x_test[:, 1], predictions, color='green', marker='D', label='predictions')
ax.scatter(x_test[:, 0], x_test[:, 1], y_test, color='red', marker='+', label='test data')
plt.legend()
plt.show()

# trying surface plot of testing and prediction points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('Test Points and their predictions')
ax.set_xlabel('Formal')
ax.set_ylabel('Familiar')
ax.set_zlabel('Label')
ax.scatter(x_test[:, 0], x_test[:, 1], y_test, color='red', marker='+', label='test data')
ax.plot_surface(x_test[:, 0], x_test[:, 1], predictions, color='green', alpha=0.05)
# ax.plot_trisurf(x_test[:, 0].flatten(), x_test[:, 1].flatten(), predictions.flatten(), color='green')
plt.legend()
plt.show()
