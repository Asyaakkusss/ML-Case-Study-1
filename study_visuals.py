import classifySpam as classify
import evaluateClassifier as eval
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#visualize email data
email_data = pd.read_csv('spamTrain1.csv')
features = email_data.iloc[:, :-1]
labels = email_data.iloc[:, -1]

median_imputer = SimpleImputer(missing_values=-1, strategy='median')
features_imputed = median_imputer.fit_transform(features)

data_visualized = TSNE(random_state=1).fit_transform(features)

# Create a scatter plot of the t-SNE transformed data
plt.figure(figsize=(10, 7))
cmap = plt.get_cmap('viridis', 2)
scatter = plt.scatter(data_visualized[:, 0], data_visualized[:, 1], c=labels, cmap=cmap, s=5)
handles, legend_labels = scatter.legend_elements(prop="colors")
plt.legend(handles=handles, labels=['Not Spam', 'Spam'], title="Classes")
plt.title('t-SNE Visualization of Email Data')
plt.xlabel('t-SNE 1st Dimension')
plt.ylabel('t-SNE 2nd Dimension')
plt.show()