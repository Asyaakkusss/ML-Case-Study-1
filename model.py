import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#set random seed 
np.random.seed(1)

#import data and shuffle it around 
data = np.loadtxt('spamTrain1.csv',delimiter=',')
shuffleIndex = np.arange(np.shape(data)[0])
np.random.shuffle(shuffleIndex)

#split the data into targets and features    
data = data[shuffleIndex,:]
features = data[:,:-1]
targets = data[:,-1]

#normalize data, normalize test and train sets separately 
scaler = PowerTransformer(method='yeo-johnson')

#train/test

#splitting data using train_test_split() w/ random_state = 1
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.5, random_state = 1)


mean_imputer = SimpleImputer(missing_values=-1, strategy='median')

X_train_mean = mean_imputer.fit_transform(X_train)
X_test_mean = mean_imputer.transform(X_test)

#fit and transform on training set 
f_t_train = scaler.fit_transform(X_train_mean)

#only fit on test set 
f_test = scaler.transform(X_test_mean)

#run the svm 
svm = LogisticRegression(solver='liblinear', random_state=1, C=10)
#svm = SVC(kernel='rbf', gamma='scale', C=10, random_state=1)
svm.fit(f_t_train, y_train)

#evaluate accuracy 
predictions = svm.predict(f_test)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
predictions_prob = svm.predict_proba(f_test)[:, 1]  # Get probabilities for class 1
print("Test set AUC: ", roc_auc_score(y_test, predictions_prob))


#evaluate accuracy 

