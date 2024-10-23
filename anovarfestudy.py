import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_classif

# Evaluate true positive rate at a specific false positive rate
def tprAtFPR(labels, outputs, desiredFPR):
    fpr, tpr, _ = roc_curve(labels, outputs)
    maxFprIndex = np.where(fpr <= desiredFPR)[0][-1]
    fprBelow = fpr[maxFprIndex]
    fprAbove = fpr[maxFprIndex + 1]
    tprBelow = tpr[maxFprIndex]
    tprAbove = tpr[maxFprIndex + 1]
    tprAt = ((tprAbove - tprBelow) / (fprAbove - fprBelow) * (desiredFPR - fprBelow) + tprBelow)
    return tprAt, fpr, tpr

# Cross-validate the AUC score of the model
def aucCV(features, labels):
    model = make_pipeline(SimpleImputer(missing_values=-1, strategy='median'),
                          RandomForestClassifier(criterion='entropy', max_depth=10, max_features='log2', min_samples_leaf=2, min_samples_split=5, n_estimators=100, random_state=1))
    scores = cross_val_score(model, features, labels, cv=10, scoring='roc_auc')
    return scores

# Preprocessing pipeline: imputation and normalization
def pre_pipe(trainFeatures, trainLabels, testFeatures):
    # Imputation
    median_imputer = SimpleImputer(missing_values=-1, strategy='median')
    trainFeatures_imputed = median_imputer.fit_transform(trainFeatures)
    testFeatures_imputed = median_imputer.transform(testFeatures)

    return trainFeatures_imputed, testFeatures_imputed

# Feature selection using RFE and ANOVA
def feature_selection(trainFeatures, trainLabels):
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    rfe = RFE(estimator=model, n_features_to_select=5)  # Adjust the number of features to select
    rfe.fit(trainFeatures, trainLabels)
    
    return rfe.support_, rfe.ranking_

# Predict test results using a RandomForestClassifier
def predictTest(trainFeatures, trainLabels, testFeatures):
    # Processed data
    trainFeatures_processed, testFeatures_processed = pre_pipe(trainFeatures, trainLabels, testFeatures)
    
    # Feature selection
    selected_features_mask, rankings = feature_selection(trainFeatures_processed, trainLabels)
    print("Selected features using RFE:")
    print(np.array(range(trainFeatures_processed.shape[1]))[selected_features_mask])
    
    # Use only the selected features for training and testing
    trainFeatures_selected = trainFeatures_processed[:, selected_features_mask]
    testFeatures_selected = testFeatures_processed[:, selected_features_mask]

    # Define the classifier
    classifier = RandomForestClassifier(criterion='entropy', max_depth=10, max_features='log2',
                                         min_samples_leaf=2, min_samples_split=5, n_estimators=100, random_state=1)

    # Fit the model
    classifier.fit(trainFeatures_selected, trainLabels)
    
    # Feature importances
    importances = classifier.feature_importances_
    
    # Sort features by importance
    sorted_indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for idx in sorted_indices:
        print(f"Feature {idx}: {importances[idx]}")
    
    # Predict probabilities for the test set
    testOutputs = classifier.predict_proba(testFeatures_selected)[:, 1]

    return testOutputs

    '''this also makes it worse lmaoooo'''