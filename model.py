"""
@authors Madeleine Comtois and Ciara Gilsenan
@version 2/12/2020
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_curve, classification_report
from extract_features import get_feature_matrix

feature_matrix = get_feature_matrix()
X1 = np.array(feature_matrix[:, 0:1])
X2 = np.array(feature_matrix[:, 1:2])
X3 = np.array(feature_matrix[:, 2:len(feature_matrix[0])-1])
X = np.column_stack((X1, X2, X3))
y = feature_matrix[:, len(feature_matrix[0])-1:].flatten()
print(len(X))
print(len(y))

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

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

plt.show()
plt.cla()

# choose a penalty weight C for the Logistic Regression Model using k-fold cross validation
def choose_c():
    mean_error = []
    std_error = []
    Ci_range = [0.1, 0.5, 1, 5, 10, 50]
    for Ci in Ci_range:
        model = LogisticRegression(penalty='l2', solver='lbfgs', C=Ci)
        temp = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            temp.append(mean_squared_error(y[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())

    print("C mean square error: ", mean_error)
    plt.title('C vs Mean Square Error')
    plt.errorbar(Ci_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel('C')
    plt.ylabel('Mean square error')
    plt.show()

choose_c()

# Train a logistic regression classifier
logistic_model = LogisticRegression(penalty='l2', solver='lbfgs', C=5)
logistic_model.fit(X, y)

print("Logistic Regression Coefficients: ", logistic_model.coef_)
print("Logistic Regression Intercept", logistic_model.intercept_)
print("Accuracy score: ", logistic_model.score(X, y))

# Predict target values with sklearn logistic model
logistic_preds = logistic_model.predict(X)

# create a baseline model and print its confusion matrix
dummy = DummyClassifier(strategy="most_frequent").fit(X, y)
print("Baseline Accuracy score: ", dummy.score(X, y))
ydummy = dummy.predict(X)

# Calculate the confusion matrices for Logistic Regression and baseline model
print("\nLogistic Regression Confusion Matrix", confusion_matrix(y, logistic_preds))
print("Baseline Model Confusion Matrix", confusion_matrix(y, ydummy))


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


graph_predictions()

# Plot the ROC curve for the models
def plot_ROC():
    # logistic regression
    fpr, tpr, _ = roc_curve(y, logistic_model.decision_function(X))
    plt.plot(fpr, tpr)

    # baseline classifier
    fpr, tpr, _ = roc_curve(y, ydummy)
    plt.plot(fpr, tpr)

    plt.title('Logistic Regression ROC')
    plt.legend(["logistic", "baseline"])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='dashed', linewidth=3)
    plt.show()


plot_ROC()

print(classification_report(y, logistic_preds))





