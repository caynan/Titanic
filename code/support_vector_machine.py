import numpy as np
import pandas as pd
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
from patsy import dmatrices
from sklearn import datasets, svm

# Load the data as a Panda DataFrame
df = pd.read_csv('..data/train.csv')

# ---- PREPROCESSING --- #

# Remove Ticket and Cabin feature, since they have a lot of NaN and will not bring much to our models
df = df.drop(['Ticket', 'Cabin'], axis=1)
# remove NaN values (any line with a NaN will be droped)
df = df.dropna()

# Declare a formula for our SVM algorithm
formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'

# create a regression friendly data frame
y, x = dmatrices(formula, data=df, return_type='matrix')

# ---- SUPPORT VECTOR MACHINE ALGORITHM ---- #

results = []

for i in xrange(100):
    # Select which features we could like to analyze
    feature_1 = 2
    feature_2 = 3

    # create a regression friendly data frame
    y, x = dmatrices(formula_ml, data=df, return_type='matrix')

    X = np.asarray(x)
    X = X[:,[feature_1, feature_2]]

    y = np.asarray(y)
    # needs to be 1 dimenstional so we flatten. it comes out of dmatrices with a shape.
    y = y.flatten()

    n_sample = len(X)

    # shuffle data
    order = np.random.permutation(n_sample)
    X = X[order]
    y = y[order].astype(np.float)

    # do a cross validation with 70% to train and 30% to test
    seventy_precent_of_sample = int(.7 * n_sample)
    X_train = X[:seventy_precent_of_sample]
    y_train = y[:seventy_precent_of_sample]
    X_test = X[seventy_precent_of_sample:]
    y_test = y[seventy_precent_of_sample:]

    # Here you can output which ever result you would like by changing the Kernel and clf.predict lines
    # Change kernel here to poly, rbf or linear
    # adjusting the gamma level also changes the degree to which the model is fitted
    clf = svm.SVC(kernel='rbf', gamma=3).fit(X_train, y_train)
    y,x = dmatrices(formula_ml, data=test_data, return_type='dataframe')

    # Change the interger values within x.ix[:,[6,3]].dropna() explore the relationships between other
    # features. the ints are column postions. ie. [6,3] 6th column and the third column are evaluated.
    res_svm = clf.predict(x.ix[:,[6,3]].dropna())

    res_svm = DataFrame(res_svm,columns=['Survived'])
    res_svm.to_csv("../data/output/svm_linear_63_g10.csv") # saves the results for you, change the name as you please.

    # add score to results
    results.append(clf.score(X_test, y_test))

# Calculate mean of results and print result
results_mean = np.array(results).mean()
print "Mean accuracy of Support Vector Machine Predictions on the data was: {0}".format(results_mean)
