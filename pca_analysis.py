import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

trainData = np.loadtxt('spamTrain1.csv',delimiter=',')

# Randomly shuffle rows of training and test sets then separate labels
# (last column)
shuffleIndex = np.arange(np.shape(trainData)[0])
np.random.shuffle(shuffleIndex)
trainData = trainData[shuffleIndex,:]
trainFeatures = trainData[:,:-1]
trainLabels = trainData[:,-1]

# Assuming X is your feature matrix with 31 features
pca = PCA()
pca.fit(trainData)

# Plot explained variance ratio
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Number of Components')
plt.grid(True)
plt.show()

plt.plot(pca.explained_variance_ratio_, 'o-', lw=2)
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Individual Components')
plt.grid(True)
plt.show()
