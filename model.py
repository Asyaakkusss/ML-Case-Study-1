import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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

def predictTest(trainFeatures,trainLabels,testFeatures): 
    # Normalize the data using PowerTransformer for both training and test sets
    scaler = PowerTransformer(method='yeo-johnson')

    # Handle missing values in the training and test sets
    mean_imputer = SimpleImputer(missing_values=-1, strategy='median')

    # Impute missing values in the training features
    trainFeatures_imputed = mean_imputer.fit_transform(trainFeatures)
    
    # Impute missing values in the test features
    testFeatures_imputed = mean_imputer.transform(testFeatures)

    # Scale and transform the training features
    trainFeatures_transformed = scaler.fit_transform(trainFeatures_imputed)

    # Transform the test features based on the training set scaling
    testFeatures_transformed = scaler.transform(testFeatures_imputed)

    # Initialize and train a Logistic Regression classifier
    classifier = LogisticRegression(solver='liblinear', random_state=1, C=10)
    classifier.fit(trainFeatures_transformed, trainLabels)

    # Make predictions on the test features
    predictions = classifier.predict(testFeatures_transformed)
    
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

    plt.plot(fpr,tpr)

    print(f'Test set AUC: {aucTestRun}')
    print(f'TPR at FPR = {desiredFPR}: {tprAtDesiredFPR}')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for spam detector')    
    plt.show()
