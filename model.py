"""
@authors Madeleine Comtois and Ciara Gilsenan
@version 2/12/2020
"""
from matplotlib.lines import Line2D
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

# fuction to graph the data on a 3D graph
def graph_data():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1, X2, y)
    plt.title("3D Graph")
    plt.xlabel("Formal count")
    plt.ylabel("Familiar count")
    plt.show()
    plt.cla()


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

# choose a penalty weight C for the Logistic Regression Model using k-fold cross validation
def choose_c():
    mean_error = []
    std_error = []
    Ci_range = [0.1, 0.5, 1, 5, 10, 50]
    for Ci in Ci_range:
        model = LogisticRegression(penalty='l2', solver='lbfgs', C=Ci)
        temp = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(x):
            model.fit(x[train], y[train])
            ypred = model.predict(x[test])
            temp.append(mean_squared_error(y[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())

    print("C mean square error: ", mean_error)
    plt.title('C vs Mean Square Error')
    plt.errorbar(Ci_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel('C')
    plt.ylabel('Mean square error')
    plt.show()

def graph_predictions():
    plt.title('Logistic Regression Predictions')
    plt.xlabel("Formal count")
    plt.ylabel("Familiar count")
    legend_elements = [Line2D([0], [0], marker='+', color='red', label='Formal'), Line2D([0], [0], marker='_', color='blue', label='Familiar')]
    plt.legend(handles=legend_elements, loc='upper right')
    for i in range(len(logistic_preds)):
        if logistic_preds[i] == 1:
            plt.scatter(X1[i], X2[i], color="red", marker='+')
        if logistic_preds[i] == -1:
            plt.scatter(X1[i], X2[i], color="blue", marker='_')

    plt.show()
    plt.cla()



# Get training and testing data
feature_matrix = get_feature_matrix()
X1 = np.array(feature_matrix[:, 0:1])
X2 = np.array(feature_matrix[:, 1:2])
X3 = np.array(feature_matrix[:, 2:len(feature_matrix[0])-1])
x = np.column_stack((X1, X2, X3))
y = feature_matrix[:, len(feature_matrix[0])-1:].flatten()

# Train - Test Split
# x = normalize(X1, X2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, stratify=None)


plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
graph_data()

# plot the data on a 2D graph
plt.title('2D Graph')
plt.xlabel("Formal count")
plt.ylabel("Familiar count")
legend_elements = [Line2D([0], [0], marker='+', color='g', label='Formal'), Line2D([0], [0], marker='_', color='b', label='Familiar')]
plt.legend(handles=legend_elements, loc='upper right')

# label the data for the graph as either '+' for target values of +1 or 'o' for values of -1
for i in range(len(y)):
    if y[i] == 1:
        plt.scatter(X1[i], X2[i], color="green", marker='+')
    if y[i] == -1:
        plt.scatter(X1[i], X2[i], color="blue", marker='_')


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

# Train a logistic regression classifier
logistic_model = LogisticRegression(penalty='none', solver='lbfgs')
logistic_model.fit(x_train, y_train)

print("Logistic Regression Coefficients: ", logistic_model.coef_)
print("Logistic Regression Intercept", logistic_model.intercept_)
print("Accuracy score: ", logistic_model.score(x_train, y_train))

# Predict target values with sklearn logistic model
logistic_preds = logistic_model.predict(x_test)

# create a baseline model and print its confusion matrix
dummy = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
print("Baseline Accuracy score: ", dummy.score(x_train, y_train))
ydummy = dummy.predict(x_test)

# Calculate the confusion matrices for Logistic Regression and baseline model
print("\nLogistic Regression Confusion Matrix", confusion_matrix(y_test, logistic_preds))
print("Baseline Model Confusion Matrix", confusion_matrix(y_test, ydummy))


graph_predictions()

# Plot the ROC curve for the models
def plot_ROC():
    # logistic regression
    fpr, tpr, _ = roc_curve(y_test, logistic_model.decision_function(x_test))
    plt.plot(fpr, tpr, label="Logistic")
    print("AUC:", auc(fpr, tpr))

    # Kernel_SVC
    fpr, tpr, _ = roc_curve(y_test, SVC.decision_function(x_test))
    plt.plot(fpr, tpr, label="SVC")
    print("AUC:", auc(fpr, tpr))

    # baseline classifier
    fpr, tpr, _ = roc_curve(y_test, ydummy)
    plt.plot(fpr, tpr, label="Dummy Classifier")

    plt.title('Model ROC')
    plt.legend()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='dashed', linewidth=3)
    plt.show()


plot_ROC()


# Evaluation
print("\nEVALUATION OF KERNEL_SVC\nMSE on test data:", mean_squared_error(y_test, predictions))
print("Accuracy on test data:", accuracy_score(y_test, predictions))
print("Recall Score:", recall_score(y_test, predictions))
print("Precision Score:", precision_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))


# Plot Predictions & Training for Kernel_SVC
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('3D Predictions & Test Data')
ax.set_xlabel('Formal')
ax.set_ylabel('Familiar')
ax.set_zlabel('Label')
ax.scatter(x_test[:, 0], x_test[:, 1], predictions, color='green', marker='D', label='predictions')
ax.scatter(x_test[:, 0], x_test[:, 1], y_test, color='red', marker='+', label='test data')
plt.legend()
plt.show()