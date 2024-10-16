import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, classification_report
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



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

    #-----------standardization------------


    #-----------normalization---------
    # Normalize the data using PowerTransformer for both training and test sets
    '''
    scaler = PowerTransformer(method='yeo-johnson')
    # Scale and transform the training features
    trainFeatures_transformed = scaler.fit_transform(trainFeatures_imputed)
    # Transform the test features based on the training set scaling
    testFeatures_transformed = scaler.transform(testFeatures_imputed)'''
    
    #-----------feature reduction---------
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_train_lda = lda.fit_transform(trainFeatures_imputed, trainLabels)
    X_test_lda = lda.transform(testFeatures_imputed)
    
    return X_train_lda, X_test_lda


def predictTest(trainFeatures,trainLabels,testFeatures):
    #Get the processed data
    trainFeatures_processed, testFeatures_processed = pre_pipe(trainFeatures, trainLabels, testFeatures)

    # Initialize and train a Logistic Regression classifier
    classifier = SVC(kernel='rbf', C = 10, random_state=1)
    classifier.fit(trainFeatures_processed, trainLabels)

    # Make predictions on the test features
    predictions = classifier.predict(testFeatures_processed)
    
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
    
    # Evaluate the classifier accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Get probabilities for class 1 and calculate AUC
    aucTestRun = roc_auc_score(y_test,predictions)
    tprAtDesiredFPR,fpr,tpr = tprAtFPR(y_test,predictions,desiredFPR)

    report = classification_report(y_test, predictions)

    plt.plot(fpr,tpr)

    print(f'Test set AUC: {aucTestRun}')
    print(f'TPR at FPR = {desiredFPR}: {tprAtDesiredFPR}')
    print("Classification Report:\n", report)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for spam detector')    
    plt.show()
