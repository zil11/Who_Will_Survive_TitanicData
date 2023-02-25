# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:57:59 2023

@author: jeffe
"""

# Import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Import data

df = pd.read_csv(r'C:\Users\jeffe\Desktop\PITT\ECON 2824 Big Data and Forcasting in Econ\Assignments\Assignment3\ass3df.csv')

# Summary statistics

stat = df.describe()

# Check for missing values

print(df.isna().sum())

# Drop 'cabin' due to 1014 missing values, also drop 'name' and 'ticket'

df = df.drop(['cabin', 'ticket', 'name', 'id'], axis=1)

# Check the type of each feature

df.info()

# Remove missing values in 'age'

df = df.dropna()

# Convert 'gender' to a dummy variable

df.loc[df['gender'] == 'male', 'gender'] = 1
df.loc[df['gender'] == 'female', 'gender'] = 0

# Convert 'embarked' to numerical group indicator

## C = Cherbourg, Q = Queenstown, S = Southampton
df.loc[df['embarked'] == 'C', 'embarked'] = 1
df.loc[df['embarked'] == 'Q', 'embarked'] = 2
df.loc[df['embarked'] == 'S', 'embarked'] = 3

# Look at 'embarked' distribution

sns.countplot(df['embarked'])
plt.xlabel('Port')
plt.ylabel('Count')
plt.title('Location of Embarked Port')
plt.show()

# Assign group indicators to 'age'

## Children: 0-12 years
## Adolescents: 13-19 years
## Young Adults: 20-39 years
## Adults: 40-56 years
## Seniors: 57 years and above

df.loc[df['age'] <= 12, 'age'] = 1
df['age'] = np.where(df['age'].between(13,19), 2, df['age']) 
df['age'] = np.where(df['age'].between(20,39), 3, df['age'])
df['age'] = np.where(df['age'].between(40,56), 4, df['age'])
df.loc[df['age'] >= 57, 'age'] = 5

# Re-look at 'age' distribution

sns.countplot(df['age'])
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Age Group Distribution')
plt.show()

# Assign group indicators to 'fare'

## Economy: < $13
## Third-class: $13-$49.99
## Second-class: $50-$100
## First-class: > $100

df.loc[df['fare'] < 13, 'fare'] = 1
df['fare'] = np.where(df['fare'].between(13,49.99), 2, df['fare']) 
df['fare'] = np.where(df['fare'].between(50,100), 3, df['fare'])
df.loc[df['fare'] > 100, 'fare'] = 4

# Re-look at 'fare' distribution

sns.countplot(df['fare'])
plt.xlabel('Fare group')
plt.ylabel('Count')
plt.title('Fare Group Distribution')
plt.show()

# Re-check summary statistics after data cleaning

stat = df.describe()
print(stat)

# Now, check correlations

colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

# Check correlations between features and survival

df[['class', 'survived']].groupby(['class'], as_index=False).mean().sort_values(by='survived', ascending=False)
df[['gender', 'survived']].groupby(['gender'], as_index=False).mean().sort_values(by='survived', ascending=False)
df[['age', 'survived']].groupby(['age'], as_index=False).mean().sort_values(by='survived', ascending=False)
df[['sib_sp', 'survived']].groupby(['sib_sp'], as_index=False).mean().sort_values(by='survived', ascending=False)
df[['par_chil', 'survived']].groupby(['par_chil'], as_index=False).mean().sort_values(by='survived', ascending=False)
df[['fare', 'survived']].groupby(['fare'], as_index=False).mean().sort_values(by='survived', ascending=False)
df[['embarked', 'survived']].groupby(['embarked'], as_index=False).mean().sort_values(by='survived', ascending=False)

# Given that 'fare' and 'class' have -0.74 correlation, I decided to drop 'fare'

df = df.drop('fare', axis=1)

# Data visualization

g = sns.FacetGrid(df, col='survived')
g.map(plt.hist, 'age', bins=5)

# Split dataframe into train/test dataset

features = ['gender','age','class', 'embarked']
x = df.loc[:,features]
y = df.loc[:,['survived']]

# Split data into test and training

X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, train_size=0.68, random_state=1)

# Define decision tree model

tree = DecisionTreeClassifier(random_state=0)

# Fit the model

tree.fit(X_train, y_train)

# Compute accuracy in the test data

print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# Plot the tree
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["Death", "Survival"],
    feature_names=features, impurity=True, filled=True)

import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))

# apply cost complexity pruning

# call the cost complexity command
path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# for each alpha, estimate the tree
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# drop the last model because that only has 1 node
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

# plot accuracy (in test and training) over alpha; first compute accuracy for each alpha
train_acc = []
test_acc = []
for c in clfs:
    y_train_pred = c.predict(X_train)
    y_test_pred = c.predict(X_test)
    train_acc.append(accuracy_score(y_train_pred,y_train))
    test_acc.append(accuracy_score(y_test_pred,y_test))
    
# second, plot it
plt.scatter(ccp_alphas,train_acc)
plt.scatter(ccp_alphas,test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Accuracy vs alpha')
plt.show()


# estimate the tree with the optimal alpha and display accuracy
clf_ = DecisionTreeClassifier(random_state=0 ,ccp_alpha=0.005)
clf_.fit(X_train,y_train)

print("Accuracy on test set: {:.3f}".format(clf_.score(X_test, y_test)))

# plot the pruned tree
export_graphviz(clf_, out_file="tree.dot", class_names=["Death", "Survival"], feature_names=features, impurity=True, filled=True)

with open("tree.dot") as f:
    dot_graph = f.read()
    display(graphviz.Source(dot_graph))
    
# Predict the probability of each class for a new observation

probabilities = clf_.predict_proba(X_test)

# Print the probabilities
print(probabilities)

# Using probit model

import statsmodels.api as sm

# fit the Probit Regression model

features = ['gender','age','class','embarked']
x = df.loc[:,features]
y = df.loc[:,['survived']]

model = sm.Probit(y,x.astype(float)).fit()
print(model.summary())

# Make predictions on the test set
y_pred = model.predict(X_test.astype(float))
print(y_pred)

# Round the predicted values to get binary predictions
y_pred_binary = [1 if y >= 0.5 else 0 for y in y_pred]

# Evaluate the model performance on the test set
accuracy = sum(y_pred_binary) / len(y_test)
print('Accuracy:', accuracy)

