# Import necessary modules
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
from classifySpam import predictTest
from evaluateClassifierTest import tprAtFPR


trainData = np.loadtxt('spamTrain1.csv',delimiter=',')
testData = np.loadtxt('spamTrain2.csv',delimiter=',')


# Randomly shuffle rows of training and test sets then separate labels (last column)
shuffleIndex = np.arange(np.shape(trainData)[0])
np.random.shuffle(shuffleIndex)
trainData = trainData[shuffleIndex,:]
trainFeatures = trainData[:,:-1]
trainLabels = trainData[:,-1]

shuffleIndex = np.arange(np.shape(testData)[0])
np.random.shuffle(shuffleIndex)
testData = testData[shuffleIndex,:]
testFeatures = testData[:,:-1]
testLabels = testData[:,-1]

# Step 1: Fit RandomForestClassifier to the training data to get feature importance
classifier = RandomForestClassifier(criterion='entropy', max_depth=10, max_features='log2', 
                                    min_samples_leaf=2, min_samples_split=5, n_estimators=100, random_state=1)
classifier.fit(trainFeatures, trainLabels)

# Step 2: Perform feature selection based on importance
threshold = 0.001  # Define the importance threshold, can adjust based on needs
importances = classifier.feature_importances_
important_indices = np.where(importances >= threshold)[0]

# Step 3: Select important features for training and test sets
trainFeatures_important = trainFeatures[:, important_indices]
testFeatures_important = testFeatures[:, important_indices]

# Step 4: Use the reduced feature set for predictions
testOutputs = predictTest(trainFeatures_important, trainLabels, testFeatures_important)

# Step 5: Evaluate model performance
aucTestRun = roc_auc_score(testLabels, testOutputs)
tprAtDesiredFPR, fpr, tpr = tprAtFPR(testLabels, testOutputs, 0.01)

# Output results
print(f"AUC Test Run: {aucTestRun}")
print(f"TPR at Desired FPR: {tprAtDesiredFPR}")

'''this makes both auc and tpr at fpr worse LMAOOOOOO'''