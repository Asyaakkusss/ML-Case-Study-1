import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, classification_report
from sklearn.preprocessing import PowerTransformer


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
                          RandomForestClassifier(n_estimators=100, random_state=1))
    scores = cross_val_score(model, features, labels, cv=10, scoring='roc_auc')
    return scores

# Preprocessing pipeline: imputation and normalization
def pre_pipe(trainFeatures, trainLabels, testFeatures):
    # Imputation
    median_imputer = SimpleImputer(missing_values=-1, strategy='median')
    trainFeatures_imputed = median_imputer.fit_transform(trainFeatures)
    testFeatures_imputed = median_imputer.transform(testFeatures)

    # Normalization using PowerTransformer
    scaler = PowerTransformer(method='yeo-johnson')
    trainFeatures_transformed = scaler.fit_transform(trainFeatures_imputed)
    testFeatures_transformed = scaler.transform(testFeatures_imputed)

    return trainFeatures_transformed, testFeatures_transformed

# Predict test results using a RandomForestClassifier
def predictTest(trainFeatures, trainLabels, testFeatures):
    trainFeatures_processed, testFeatures_processed = pre_pipe(trainFeatures, trainLabels, testFeatures)
    classifier = RandomForestClassifier(n_estimators=300, random_state=1)
    classifier.fit(trainFeatures_processed, trainLabels)
    # Use predict_proba for probability outputs
    testOutputs = classifier.predict_proba(testFeatures_processed)[:,1]
    return testOutputs

if __name__ == "__main__":
    np.random.seed(1)
    desiredFPR = 0.01

    # Load and shuffle data
    data = np.loadtxt('spamTrain1.csv', delimiter=',')
    shuffleIndex = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffleIndex)
    data = data[shuffleIndex, :]
    
    # Split data into features and labels
    features = data[:, :-1]
    labels = data[:, -1]

    # 10-fold cross-validation to compute AUC
    print("10-fold cross-validation mean AUC: ", np.mean(aucCV(features, labels)))

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=1)

    # Make predictions
    predictions = predictTest(X_train, y_train, X_test)

    aucTestRun = roc_auc_score(y_test, predictions)
    tprAtDesiredFPR, fpr, tpr = tprAtFPR(y_test, predictions, desiredFPR)

    # Generate classification report
    report = classification_report(y_test, (predictions > 0.5).astype(int))

    # Plot ROC curve
    plt.plot(fpr, tpr)
    print(f'Test set AUC: {aucTestRun}')
    print(f'TPR at FPR = {desiredFPR}: {tprAtDesiredFPR}')
    print("Classification Report:\n", report)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for spam detector')
    plt.show()
    
    # Examine outputs compared to labels
    sortIndex = np.argsort(y_test)
    nTestExamples = y_test.size
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(nTestExamples), y_test[sortIndex], 'b.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Target')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(nTestExamples), predictions[sortIndex], 'r.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Output (predicted target)')
    plt.show()
