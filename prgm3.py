import numpy as np
import pandas as pd 
from  sklearn.datasets import load_iris
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt 

# Load the iris dataset 
iris = load_iris()
data = iris.data 
labels = iris.target
label_names = iris.target_names

# Convert to a data frame for better visual 
iris_df= pd.DataFrame(data,columns=iris.feature_names)

# Perfoms PCA to reduce dimensionality to 2
pca=PCA(n_components=2)
data_reduced=pca.fit_transform(data)

# Create a data frame for the reduced data
reduced_df=pd.DataFrame(data_reduced,columns=['Principle Component 1','Principle Component 2'])
reduced_df['label']=labels

# Plot the reduced data 
plt.figure(figsize=(8,6))
colors=['r','g','b']
for i,label in enumerate(np.unique(labels)):
    plt.scatter(
                reduced_df.loc[reduced_df['label']==label,'Principle Component 1'],
                reduced_df.loc[reduced_df['label']==label,'Principle Component 2'],
                label=label_names[label],
                color=colors[i]
                )
plt.title("PCA on Iris data")
plt.xlabel("principle component 1")
plt.ylabel("principle component 2")
plt.legend()
plt.grid()
plt.show()