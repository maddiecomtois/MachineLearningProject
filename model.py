"""
@authors Madeleine Comtois and Ciara Gilsenan
@version 2/12/2020
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, mean_squared_error
from extract_features import get_feature_matrix

feature_matrix = get_feature_matrix()
print(feature_matrix)
X1 = np.array(feature_matrix[:, 0:1])
X2 = np.array(feature_matrix[:, 1:2])
X3 = np.array(feature_matrix[:, 2:3])
X = np.column_stack((X1, X2, X3))
y = feature_matrix[:, 3:4]
y = list(map(int, y))
print(X)
print(len(X))
print(len(y))

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

# fuction to graph the data on a 3D graph
def graph_data():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1, X2, y)
    plt.title("Data - 3D Graph")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    plt.cla()

graph_data()

# plot the data on a 2D graph
plt.title('Dataset - 2D Graph')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
legend_elements = [Line2D([0], [0], marker='+', color='g', label='Class 1'), Line2D([0], [0], marker='_', color='b', label='Class -1')]
plt.legend(handles=legend_elements, loc='upper right')

# label the data for the graph as either '+' for target values of +1 or 'o' for values of -1
for i in range(len(y)):
    if y[i] == 1:
        plt.scatter(X1[i], X2[i], color="green", marker='+')
    if y[i] == -1:
        plt.scatter(X1[i], X2[i], color="blue", marker='_')

plt.show()
plt.cla()

# choose a penalty weight C for the Logistic Regression Model
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


