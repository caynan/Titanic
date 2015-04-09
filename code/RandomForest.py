import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from scipy.stats import mode
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold

#This path should be changed to whatever path is on your machine to train.csv
df = pd.read_csv('../data/train.csv', header=0)

#We drop these attributes as we feel they have very little importance to our model
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

#Fill in age missing values with the mean.
#This is definitely something we can improve on
mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)


#Fill in embarked missing values with the mode of embarked category
mode_embarked = mode(df['Embarked'])[0][0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)

#This was used to create dummy variables for the embarked field. First, the data in the Embarked
#category were strings, and we needed to make them into numerical values. Then, instead of giving
#them values that could suggest rank, we simply create three extra rows, one for each port. Then
#each column is filled with a 0 or 1 indicating where that person left from.
df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)


#This changes the Sex category to an integer field which is required to run the Random Forest later
df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

#Drop columns we don't need
df = df.drop(['Sex', 'Embarked'], axis=1)
df = df.drop(['SibSp', 'Parch'], axis=1)

#This just switches up columns so the survived column is the left most column
cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]

#This transforms the dataframe into a numpy array.
#This is required to use the sklearn library
train_data = df.values

'''
If you would like to see what the data frame looks like with the modified attributes,
simply uncomment the line below. This will show the attributes and the first 5 entries
as an example.
'''
#print df.head(5)

columns = list(df.columns.values)

#This assigns survived column to y and the rest to X
X = train_data[:, 2:]
y = train_data[:, 0]

#Provides train/test indices to split data in train test sets. Split dataset into k consecutive folds (In this case 10 folds)
cv = KFold(n=len(train_data), n_folds=10)
ave_prediction = []
counter = 1
for training_set, test_set in cv:
    X_train = X[training_set]
    y_train = y[training_set]
    X_test = X[test_set]
    y_test = y[test_set]

    #This is used to specify the parameters we are trying to optimize using grid search. We are reviewing the number
    #of features considered at each step a branch is made: 50% or 100% of features and the maximum number of
    #branches: 5 levels or no limit.
    parameter_grid = {
    'max_features': [0.5, 1.],
    'max_depth': [5., None]
    }
    '''
    A hyperparameter optimization algorithm that is simply an exhaustive search through a manually specified subset of data
    The model used is Random Forest using 100 estimators. This comes from the sci-kit learn module. This allows us to test
    the desired range of input parameters and review the performance of each set of values on a cross-validation process.
    '''
    grid_search = GridSearchCV(RandomForestClassifier(n_estimators=100), parameter_grid, cv=10)

    #We use the result to fit our training data
    grid_search = grid_search.fit(X_train, y_train)

    #If you would like to see the output of the grid search scores for each iteration, uncomment the line below.
    #print grid_search.grid_scores_

    #This sorts the results from grid search to pick out the best performing parameters
    sorted(grid_search.grid_scores_, key=lambda x: x.mean_validation_score)

    #If you would like to see the best performing parameters, uncomment the two lines of code below
    #print grid_search.best_score_
    #print grid_search.best_params_

    #We use the result from the grid search (best params and max depth of search) to train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, criterion='entropy', \
                                   max_features=grid_search.best_params_['max_features'], \
                                   max_depth=grid_search.best_params_['max_depth'])

    #Fit the training data
    model.fit(X_train, y_train)


    important_features = []
    for x,i in enumerate(model.feature_importances_):
        if i>np.average(model.feature_importances_):
            important_features.append(columns[x+2])
    print 'Most important features for fold ' + str(counter) + ':',', '.join(important_features)
    counter += 1

    #Use this to predict the outcome of the test data
    y_prediction = model.predict(X_test)

    #Use an array to later calculate the average prediction accuracy of the model
    ave_prediction.append(np.sum(y_test == y_prediction)*1./len(y_test))

print
print "ave prediction accuracy:", np.mean(ave_prediction)



#This is a method that is used to print out a graph comparing cross-validation and training values
#This method was found in the sci-kit learn documentation and I utilized it to show a visual representation of
#the models results
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects
    n_jobs : integer, optional
        Number of jobs to run in parallel.
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


title = "Learning Curve (Random Forests)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = cross_validation.ShuffleSplit(len(X), n_iter=50,
                                  test_size=0.2, random_state=0)
plot_learning_curve(model, title, X, y, ylim=(0.7, 1.05), cv=cv, n_jobs=4)

plt.show()
