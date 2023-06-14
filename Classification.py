import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# load the data from a CSV file
data = pd.read_csv('final_data_MS2.csv', low_memory=False)

#extract the feature matrix and target vector
X = data[["speed_limit","light_conditions","number_of_vehicles","number_of_casualties","weather_conditions","road_surface_conditions"]].values
y = data["accident_severity"].values

# create the classifiers
classifiers = {
 'KNN': KNeighborsClassifier(),
 'Naive Bayes': GaussianNB(),
 'Logistic Regression': LogisticRegression(max_iter=1000),
 'Neural Networks': MLPClassifier()
}

# split the data into training, cross-validation and test sets using stratified sampling
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(sss.split(X, y))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
cv_index, test_index = next(sss.split(X_test, y_test))
X_cv, X_test = X_test[cv_index], X_test[test_index]
y_cv, y_test = y_test[cv_index], y_test[test_index]

# generate polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_cv_poly = poly.transform(X_cv)
X_test_poly = poly.transform(X_test)

# standardize the data
scaler = StandardScaler()
X_train_poly = scaler.fit_transform(X_train_poly)
X_cv_poly = scaler.transform(X_cv_poly)
X_test_poly = scaler.transform(X_test_poly)

# evaluate the classifiers using cross-validation with polynomial features
print('With cross-validation and polynomial features:')
for name, clf in classifiers.items():
    scores = cross_val_score(clf, X_cv_poly, y_cv, cv=5)
    mean_score = np.mean(scores)
    print(f'{name} mean cross-validation score: {mean_score:.2f}')
    
    clf.fit(X_train_poly, y_train)
    y_pred = clf.predict(X_test_poly)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} test set accuracy: {accuracy:.2f}')
    
    # compute and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    print(f'{name} confusion matrix:\n{cm_percentage}')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage)
    disp.plot()
    plt.title(f'{name} confusion matrix')
    plt.show()
    
    # compute and plot ROC curve if possible using one-vs-rest approach
    if hasattr(clf, "predict_proba"):
        probas = clf.predict_proba(X_test_poly)
        n_classes = len(np.unique(y))
        fpr = dict()
        tpr = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test == i, probas[:, i])
        print(f'{name} ROC curve:')
        for i in range(n_classes):
            print(f'  Class {i}: fpr={fpr[i]}, tpr={tpr[i]}')
            plt.plot(fpr[i], tpr[i], label=f'Class {i}')
        plt.title(f'{name} ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()
    else:
        print(f'{name} does not support computation of ROC curve')

# evaluate the classifiers without cross-validation with polynomial features
print('Without cross-validation and polynomial features:')
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} test set accuracy: {accuracy:.2f}')
    
    # compute and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    print(f'{name} confusion matrix:\n{cm_percentage}')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage)
    disp.plot()
    plt.title(f'{name} confusion matrix')
    plt.show()
    
    # compute and plot ROC curve if possible using one-vs-rest approach
    if hasattr(clf, "predict_proba"):
        probas = clf.predict_proba(X_test)
        n_classes = len(np.unique(y))
        fpr = dict()
        tpr = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test == i, probas[:, i])
        print(f'{name} ROC curve:')
        for i in range(n_classes):
            print(f'  Class {i}: fpr={fpr[i]}, tpr={tpr[i]}')
            plt.plot(fpr[i], tpr[i], label=f'Class {i}')
        plt.title(f'{name} ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()
    else:
        print(f'{name} does not support computation of ROC curve')
