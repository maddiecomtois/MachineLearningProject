"""
@authors Madeleine Comtois and Ciara Gilsenan
@version 2/12/2020
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score, roc_curve, auc, classification_report
from sklearn.dummy import DummyClassifier
from extract_features import get_feature_matrix
import matplotlib.pyplot as plt
import numpy as np
# mpl.use('Qt5Agg')

# Function to normalize formal and familiar frequencies in the feature matrix
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

    normalized_data = np.column_stack((X1, X2, X3))
    return normalized_data


# Get training and testing data
feature_matrix = get_feature_matrix()
X1 = np.array(feature_matrix[:, 0:1])
X2 = np.array(feature_matrix[:, 1:2])
X3 = np.array(feature_matrix[:, 2:len(feature_matrix[0])-1])
x = np.column_stack((X1, X2, X3))
y = feature_matrix[:, len(feature_matrix[0])-1:].flatten()

# Train - Test Split
x = normalize(X1, X2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, stratify=None)

# initialise KFold
kf = KFold(n_splits=5)


# fuction for graphing normalized data on a 3D graph
def graph_data():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1, X2, y, c='red')
    plt.title("Normalized Data")
    plt.xlabel("Formal count")
    plt.ylabel("Familiar count")
    plt.show()
    plt.cla()


# function for choosing gamma value for Kernel_SVC model
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

    label = "Kernelized SVC Cross-Validation For Gamma"
    plt.cla()
    plt.title(label)
    plt.xlabel("gamma")
    plt.ylabel("Mean Square Error")
    plt.errorbar(gammas, mse_vals, std_vals, ecolor='red')
    plt.show()
    return g


# function for choosing C for SVC model
def choose_c_svc():
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

    plt.cla()
    label = "Kernelized SVC Cross-Validation For C"
    plt.cla()
    plt.title(label)
    plt.xlabel("C")
    plt.ylabel("Mean Square Error")
    plt.errorbar(c_values, mse_vals, std_vals, ecolor='red')
    plt.show()
    return c


# function for choosing penalty weight C for Logistic Regression Model
def choose_c_log():
    mean_error = []
    std_error = []
    Ci_range = [0.1, 0.5, 1, 5, 10, 50]
    for Ci in Ci_range:
        model = LogisticRegression(penalty='l2', solver='lbfgs', C=Ci)
        temp = []
        for train, test in kf.split(x):
            model.fit(x[train], y[train])
            ypred = model.predict(x[test])
            temp.append(mean_squared_error(y[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())

    print("Logistic C mean square error: ", mean_error)
    c = Ci_range[mean_error.index(min(mean_error))]
    plt.cla()
    plt.title('Logistic Cross-Validation For C')
    plt.errorbar(Ci_range, mean_error, yerr=std_error, linewidth=3, ecolor='red')
    plt.xlabel('C')
    plt.ylabel('Mean square error')
    plt.show()
    return c


# function for graphing logistic regression predictions
def graph_logistic_predictions():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Predictions & Test Data - Logistic Regression')
    ax.set_xlabel('Formal')
    ax.set_ylabel('Familiar')
    ax.set_zlabel('Label')
    ax.scatter(x_test[:, 0], x_test[:, 1], logistic_predictions, color='green', marker='D', label='predictions')
    ax.scatter(x_test[:, 0], x_test[:, 1], y_test, color='red', marker='+', label='test data')
    plt.legend()
    plt.show()


# function for plotting the ROC curves for the logistic, SVC, and baseline models
def plot_ROC():
    # logistic regression
    fpr, tpr, _ = roc_curve(y_test, logistic_model.decision_function(x_test))
    plt.plot(fpr, tpr, label="Logistic")
    print("\nAUC Logistic:", auc(fpr, tpr))

    # Kernel_SVC
    fpr, tpr, _ = roc_curve(y_test, SVC.decision_function(x_test))
    plt.plot(fpr, tpr, label="SVC")
    print("AUC Kernel_SVC:", auc(fpr, tpr))

    # baseline classifier
    fpr, tpr, _ = roc_curve(y_test, ydummy)
    plt.plot(fpr, tpr, label="Dummy Classifier")

    plt.title('ROC - SVC, Logistic, and Baseline')
    plt.legend()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='dashed', linewidth=3)
    plt.show()


graph_data()

# TRAIN MODELS
# Train a logistic regression classifier
c_log = choose_c_log()
logistic_model = LogisticRegression(penalty='l2', solver='lbfgs', C=c_log)
logistic_model.fit(x_train, y_train)

# Predict target values for logistic regression
logistic_predictions = logistic_model.predict(x_test)

# Evaluation
print("\nLOGISTIC REGRESSION:\nLogistic Regression Coefficients: ", logistic_model.coef_)
print("Logistic Regression Intercept", logistic_model.intercept_)
print("Accuracy score: ", accuracy_score(y_test, logistic_predictions))
print("Logistic Regression Confusion Matrix\n", confusion_matrix(y_test, logistic_predictions))
print(classification_report(y_test, logistic_predictions))

# Plot Predictions & Test Data for Logistic Regression
graph_logistic_predictions()


# Train Kernelised SVC with chosen gamma and C
g = choose_gamma()
c = choose_c_svc()
SVC = SVC(kernel='rbf', gamma=g, C=c)
SVC.fit(x_train, y_train)
svc_predictions = SVC.predict(x_test).reshape(-1, 1)

# Plot Predictions & Test Data for Kernel_SVC
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('Predictions & Test Data - Kernel_SVC')
ax.set_xlabel('Formal')
ax.set_ylabel('Familiar')
ax.set_zlabel('Label')
ax.scatter(x_test[:, 0], x_test[:, 1], svc_predictions, color='green', marker='D', label='predictions')
ax.scatter(x_test[:, 0], x_test[:, 1], y_test, color='red', marker='+', label='test data')
plt.legend()
plt.show()

# Evaluation
print("\nKERNEL_SVC\n", mean_squared_error(y_test, svc_predictions))
print("Accuracy on test data:", accuracy_score(y_test, svc_predictions))
print("Kernel_SVC Intercept:", SVC.intercept_)
print("Confusion Matrix:\n", confusion_matrix(y_test, svc_predictions))
print(classification_report(y_test, svc_predictions))
# coef_ is only for linear kernels -- Gaussian is a curve

# create a baseline model and print its confusion matrix
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(x_train, y_train)
ydummy = dummy.predict(x_test)
print("\nBaseline Accuracy score: ", accuracy_score(y_test, ydummy))

# Calculate the confusion matrices for Logistic Regression and baseline model
print("Baseline Model Confusion Matrix", confusion_matrix(y_test, ydummy))

# show the ROC graph
plot_ROC()
