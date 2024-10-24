import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, classification_report, recall_score, precision_score
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV


#evaluate accuracy
def tprAtFPR(labels,outputs,desiredFPR):
    fpr,tpr,thres = roc_curve(labels,outputs)
    # True positive rate for highest false positive rate < 0.01
    maxFprIndex = np.where(fpr<=desiredFPR)[0][-1]
    fprBelow = fpr[maxFprIndex]
    fprAbove = fpr[maxFprIndex+1]
    # Find TPR at exactly desired FPR by linear interpolation
    tprBelow = tpr[maxFprIndex]
    tprAbove = tpr[maxFprIndex+1]
    tprAt = ((tprAbove-tprBelow)/(fprAbove-fprBelow)*(desiredFPR-fprBelow) 
             + tprBelow)
    return tprAt,fpr,tpr

#this encompasses all processes we do to the data before we run it through the model
def pre_pipe(trainFeatures, trainLabels, testFeatures):
    #-----------imputation------------
     # Handle missing values in the training and test sets
    median_imputer = SimpleImputer(missing_values=-1, strategy='median')
    # Impute missing values in the training features
    trainFeatures_imputed = median_imputer.fit_transform(trainFeatures)
    # Impute missing values in the test features
    testFeatures_imputed = median_imputer.transform(testFeatures)

    # Normalize the data using PowerTransformer for both training and test sets
    
    scaler = PowerTransformer(method='yeo-johnson')
    # Scale and transform the training features
    trainFeatures_transformed = scaler.fit_transform(trainFeatures_imputed)
    # Transform the test features based on the training set scaling
    testFeatures_transformed = scaler.transform(testFeatures_imputed)

    
    return trainFeatures_transformed, testFeatures_transformed


def predictTest(trainFeatures,trainLabels,testFeatures):
    #Get the processed data
    trainFeatures_processed, testFeatures_processed = pre_pipe(trainFeatures, trainLabels, testFeatures)
    
    # Normalize the data using PowerTransformer for both training and test sets
    #Create a dictionary of possible parameters
    
    params_grid = {'C': [0.001, 0.1, 1, 10, 100, 1000, 100000],
            'gamma': [0.0001, 0.001, 0.1, 1, 10, 100, 1000]}

    #Create the GridSearchCV object
    grid_clf = GridSearchCV(SVC(kernel='rbf', random_state=1, probability=True), params_grid, scoring='roc_auc', cv=5)
    #grid_clf = SVC(kernel='rbf', C=0.001, gamma=0.0001, random_state=1)
    grid_clf.fit(trainFeatures_processed, trainLabels)
    print(f"The best estimators are {grid_clf.best_estimator_}")

    # Make predictions on the test features
    predictions = grid_clf.predict_proba(testFeatures_processed)[:, 1]
    
    return predictions

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(1)
    desiredFPR = 0.01

    # Import data and shuffle it
    data = np.loadtxt('spamTrain1.csv', delimiter=',')
    shuffleIndex = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffleIndex)
    
    # Split the data into features and labels
    data = data[shuffleIndex, :]
    features = data[:, :-1]
    targets = data[:, -1]
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.5, random_state=1)
    
    # Call the predictTest function
    predictions = predictTest(X_train, y_train, X_test)
    
    # Get probabilities for class 1 and calculate AUC
    aucTestRun = roc_auc_score(y_test,predictions)
    tprAtDesiredFPR,fpr,tpr = tprAtFPR(y_test,predictions,desiredFPR)
    '''
    
    report = classification_report(y_test, predictions)
    # Make predictions and calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    # Cross-validation accuracy
    cv_scores = cross_val_score(classifier, features, targets, cv=5, scoring='accuracy')
    
    print(f'Accuracy: {accuracy}\n')
    print(f'Precision: {precision}\n')
    print(f'Recall: {recall}\n')
    print(f'Cross-validation accuracy: {accuracy}\n')
    print("Classification Report:\n", report)   

    
    '''
    plt.plot(fpr,tpr)
    print(f'Test set AUC: {aucTestRun}\n')
    print(f'TPR at FPR = {desiredFPR}: {tprAtDesiredFPR}\n')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for spam detector')    
    plt.show()
    