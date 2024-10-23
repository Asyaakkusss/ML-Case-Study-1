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

# Predict test results using a RandomForestClassifier
def predictTest(trainFeatures, trainLabels, testFeatures):
    #processed data
    trainFeatures_processed, testFeatures_processed = pre_pipe(trainFeatures, trainLabels, testFeatures)
    
    #GridSearchCV hyperparameters:
    '''
    hyperparameters = {
    'n_estimators': [20, 50, 100, 200, 300],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']    
    }
    '''

    classifier = RandomForestClassifier(criterion='entropy', max_depth=10, max_features='log2', min_samples_leaf=2, min_samples_split=5, n_estimators=100, random_state=1)
    #gridSearch = GridSearchCV(classifier, hyperparameters, cv=5, scoring='roc_auc', n_jobs=-1)
    #gridSearch.fit(trainFeatures_processed, trainLabels)
    #best_forest = gridSearch.best_estimator_

    # Use predict_proba for probability outputs
    classifier.fit(trainFeatures_processed, trainLabels)
    importances = classifier.feature_importances_

    # Sort features by importance
    sorted_indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for idx in sorted_indices:
        print(f"Feature {idx}: {importances[idx]}")
    
    testOutputs = classifier.predict_proba(testFeatures_processed)[:,1]
    #print(f"Best hyperparameters {gridSearch.best_params_}")

    return testOutputs
