#To get input file from system arguments
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter
train_file = sys.argv[1]
test_file = sys.argv[2]
target = sys.argv[3]

#To get train & test dataframe to dataframe and split the target
import pandas as pd
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
xtrain = train_df.drop(target,axis=1)
ytrain = pd.DataFrame(train_df[target])
xtest = test_df.drop(target,axis=1)
ytest = pd.DataFrame(test_df[target])

#creating node class to buildtree
class Node:
    def __init__(self, current):
        self.data = current
        self.branch = list()
        self.child = list()

# decisiontree class to fit, predict and compute accuracy score
class decisiontree():
    x = pd.DataFrame()
    y = pd.DataFrame()
    df = pd.DataFrame()

    #Root node of the tree
    def __init__(self):
        self.root = None

    # Fitting the train data
    def fit(self, xdata, ydata):
        self.x = self.x.append(xdata, ignore_index=True)
        self.y = self.y.append(ydata, ignore_index=True)
        self.df = xdata.join(ydata)
        self.columns = self.x.columns.tolist()
        print("ID3 Decision Tree Algorithm")
        self.root = self.buildtree(self.df, self.columns, self.root)

    def buildtree(self, df, columns, thisnode):
        myList = df[target].tolist()
        # Exit condition to check if the target values are all either +ve/ -ve
        if myList.count(myList[0]) == len(myList):
            print(": ", myList[0])
            return myList[0]
        # Exit condition to find frequency if one column is present
        if len(columns) == 1:
            result = Counter(myList)
            result.most_common()
            freq, count = result.most_common()[0]
            print(": ", freq)
            return freq
        print("")
        targetentropy = self.findentropy(df, target)
        idx = self.findnode(targetentropy, df, columns)
        selected = columns[idx]
        if (thisnode is not None):
            print("   ", end="")
        # Each attribute is selected and node is created
        thisnode = Node(selected)
        columns.remove(selected)
        unique = list(df[selected].unique())
        for each in unique:
            print(selected,end = " = ")
            print(each,end="")
            others = columns[:]
            currentdf = df.loc[df[selected] == each]
            # Recursively finding the node
            thisnode.branch.append(each)
            thisnode.child.append(self.buildtree(currentdf, others, thisnode))
        return thisnode

    # To find entropy
    def findentropy(self, df, feature):
        total = 0
        param = list(df[feature].unique())
        X = (len(df[feature]))
        for one in param:
            inthis = ((df[feature] == one).sum())
            value = round((inthis / X), 3)
            total += round(value * (np.log2(value)), 3)
        return (-total)

    def findnode(self, thisentropy, thisDf, thiscolumns):
        compare = []
        for each in thiscolumns:
            addin = thisentropy - self.entropy(thisDf, each)
            compare.append(addin)
        maximum_gain = max(compare)
        thisentropy = maximum_gain
        index = compare.index(maximum_gain)
        return index

    def entropy(self, df, feature):
        total = 0
        param = list(df[feature].unique())
        count = (len(df[feature]))
        for each in param:
            inthis = ((df[feature] == each).sum())
            value = round((inthis / count), 3)
            newdf = pd.DataFrame(df.loc[df[feature] == each])
            total += value * self.findentropy(newdf, target)
        return total

    # Predict for test data using created tree
    def predict(self, testdata):
        foundlist = []
        for index, row in testdata.iterrows():
            foundlist.append(self.predictedlist(self.root, row))
        return foundlist

    def predictedlist(self, thisnode, row):
        answer = None
        value = row[thisnode.data]
        index = thisnode.branch.index(value)
        newnode = thisnode.child[index]
        if (type(newnode) is not Node):
            answer = newnode
        else:
            answer = self.predictedlist(newnode, row)
        return answer

    # Finding accuracy
    def accuracy(self, actual, predicted):
        score = accuracy_score(actual, predicted)
        return score

model = decisiontree()
model.fit(xtrain,ytrain)

Ypred = model.predict(xtest)

accuracy = model.accuracy(ytest[target].tolist(),Ypred)
print("      ")
print("The accuracy is:",accuracy)