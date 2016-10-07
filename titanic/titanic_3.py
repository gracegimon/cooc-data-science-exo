
import pandas
titanic = pandas.read_csv('C:\Users\Grace\Documents\Kaggle\\train.csv')
# Prints the first 5 lines of the file
print(titanic.head(5))

# Prints the univariate statistics for each numeric column of the file
print(titanic.describe())

print(titanic.info())

####################################################
####### CLEANSING DATA ##############################
####################################################

# We can see in the data that there are 891 individuals but, the data is not
# cleansed, for example the Age column has 714 values only.

# To clean the Age column we can execute the following:

# Replace all the "NA" with the median of the column
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

# We should convert the Sex column to a numeric column to be able to train 
# with it
print (titanic['Sex'])


# To check all the unique values in the column we use, 
# here we notice we only have female and male.
print(titanic["Sex"].unique())

 
# We select the male and code it with 0
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# Now we transform the Embarked column

titanic["Embarked"] = titanic["Embarked"].fillna("S")

print(titanic["Embarked"].unique())

titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2



# Cleaning test data

titanic_test = pandas.read_csv('C:\Users\Grace\Documents\Kaggle\\test.csv')

# Replace missing values with training median

titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

# Now we transform the Embarked column

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

print(titanic_test["Embarked"].unique())

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic["Fare"].median())




#######################################
#### GENERATING NEW FEATURES ##########
#######################################

####TODO
# * Try using features related to the cabins. -- CHECK
# * See if any family size features might help -- do the number of women in a 
# family make the whole family more likely to survive?
# * Does the national origin of the passenger's name have anything to do with 
# survival?


# Generating a familysize column
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

# The .apply method generates a new series
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))


# Now we can set a new feature with the titles of the passangers
# that will inform us of the class he was in

import re
import numpy as np

# A function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


# We'll extract those who where located near the boat deck,
# mainly decks A to E, we give a higher punctuation for A.

def extract_deck(cabin):
    if pandas.isnull(cabin):
        return "Error"
    else:
        return cabin[0]
        
def punctuate_deck(deck):
    if deck == 'A':
        return 7
    elif deck == 'B':
        return 6
    elif deck == 'C':
        return 5
    elif deck == 'D':
        return 4
    elif deck == 'E':
        return 3
    elif deck== 'F':
        return 2
    elif deck == 'G':
        return 1
    else:
        return 0
        

# Get all the titles and print how often each one occurs.
titles = titanic["Name"].apply(get_title)
print(pandas.value_counts(titles))

# Map each title to an integer.  Some titles are very rare, and are compressed 
# into the same codes as other titles.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, 
"Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, 
"Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, 
"Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

for k,v in title_mapping.items():
    titles[titles == k] = v

# Verify that we converted everything.
print(pandas.value_counts(titles))

# Add in the title column.
titanic["Title"] = titles

titanic["CabinPunct"] =  titanic["Cabin"].apply(lambda x: extract_deck(x)).apply(punctuate_deck)

# Also for the test set
titles = titanic_test["Name"].apply(get_title)
# We're adding the Dona title to the mapping, because it's in the test set, but not the training set
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, 
"Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, 
"Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
for k,v in title_mapping.items():
    titles[titles == k] = v
titanic_test["Title"] = titles

titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]

titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))

titanic_test["CabinPunct"] = titanic_test["Cabin"].apply(lambda x: extract_deck(x)).apply(punctuate_deck)

print(pandas.value_counts(titanic_test["CabinPunct"]))

# Let's add cabin punctuation

#### FEATURE SELECTION #########
from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib.pyplot as plt

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", 
 "CabinPunct"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()


predictors = ["Pclass", "Sex", "Fare", "Title", "CabinPunct"]

########## APPLY SUPPORT VECTOR MACHINES #############################
from sklearn import svm
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
 
import numpy as np


# Let's try their ensemble
alg = RandomForestClassifier(n_estimators=10, random_state = 1)
# Different types of evaluation
scores = cross_validation.cross_val_score(alg, titanic[predictors], 
titanic["Survived"], cv=3)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

kf = KFold(titanic.shape[0], n_folds=3, random_state=1)


##TODO

# Random forest classifier in the ensemble.

# Try neural networks.
# Boosting with a different base classifier might work better
predictors = ["Pclass", "Sex", "CabinPunct", "Title"]

from sklearn.ensemble import GradientBoostingClassifier

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

#### TRAINING  AND EVALUATION #####
predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold.  
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)


# Compute accuracy by comparing to the training data.
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)
#
#
##### TESTING FOR SUBMISSION #####
#
full_predictions = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(titanic[predictors], titanic["Survived"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

predictions = (full_predictions[0] + full_predictions[1]) / 2

predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0

submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions.astype(int)
    })
    
submission.to_csv("WithCabins.csv", index=False)

print titanic["Survived"].loc[1:]


# Could majority voting be a better ensembling method than averaging probabilities?
# (Substitute predict_proba) with majority voting