import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# Read the CSV file
df = pd.read_csv('C:\\Abhishek\\IIITD\\Research\\ScalingML\\e-shop data and description\\e-shop clothing 2008_mod.csv')
df2 = pd.read_csv('C:\\Abhishek\\IIITD\\Research\\ScalingML\\e-shop data and description\\e-shop clothing 2008_mod.csv')

df2 = df2.drop('p2', axis=1)

column_types = df2.dtypes

# # Display the data types
# print(column_types)

# Convert DataFrame to NumPy array
array = df2.values

def Dsq(X, k):
    # Randomly select the first cluster center from X
    centers = [X[np.random.choice(len(X))]]

    for i in range(1, k):
        # Compute the squared distances from each point to the closest cluster center
        distances = cdist(X, centers, 'sqeuclidean')
        min_distances = np.min(distances, axis=1)

        # Compute the probabilities for selecting each point as the next cluster center
        probabilities = min_distances / np.sum(min_distances)

        # Select the next cluster center based on the probabilities
        next_center = X[np.random.choice(len(X), p=probabilities)]
        
        # Add the next center to the list of cluster centers
        centers.append(next_center)

    return np.array(centers)

# Fix centers
k = 3

cluster_centers = Dsq(array, k)
print(cluster_centers)
