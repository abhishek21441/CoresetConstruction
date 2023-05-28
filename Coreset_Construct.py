import pandas as pd
import numpy as np
import random
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

#Implementing D^2 Algorithm______________________________________________________

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

#Implementing Coreset Construction_________________________________________________

# Finding the Cluster Points to form Cluster = [B0,B1,B2...]

def finding_clusters(X,B):
    closest_points = []

    for center in B:
        distances = np.linalg.norm(X - center, axis=1)
        closest_indices = np.argwhere(distances == np.min(distances)).flatten()
        closest_points.append(X[closest_indices])

    return closest_points

# Finding closest points
def Find_closest_points(B, C):
    closest_points = []

    for i in range(len(B)):
        distances = np.linalg.norm(C - B[i], axis=1)
        closest_index = np.argmin(distances)
        closest_point = C[closest_index]
        closest_points.append(closest_point)

    return np.array(closest_points) 

# Finding Cphi X-array B-centers
def Finding_Cphi(X,B):

    min_distances = Find_closest_points(X, B)
    Cphi = np.mean(min_distances)

    return Cphi

#Sensitivity Calculations__________

#Minimum Distance (1st fraction-numerator)
def Min_Dist(x,B):
    distances = np.linalg.norm(B - x, axis=1)
    min_distance = np.min(distances)
    return min_distance


#Sum of distances (2nd fraction-numerator)
def sum_min_distances(CB, B): #CB is the cluster point x belongs to.... Assumme we find it before this step
    sum_min_distances = 0

    for cb in CB:
        distances = np.linalg.norm(B - cb, axis=1)
        min_distance = np.min(distances)
        sum_min_distances += min_distance

    return sum_min_distances



# Upper Sensitivity bound
def Sensitivity(X,B,al,Cphi,Clusters):
    Sensivities = []
    for i in range(len(X)):
        sens = 0
        sens += al*Min_Dist(X[i],B)/Cphi #fraction1
        clid = 0
        for j in range(len(Clusters)):
            if X[i] in Clusters[j]:
                sens += 2*al*sum_min_distances(Clusters[j], B)/(np.size(Clusters[j]) * Cphi) #fraction2
                clid = j
                break
        sens += 4 * len(X)/len(Clusters[clid]) #3rd fraction
        Sensivities.append(sens)
    
    return Sensivities #returns an array of sensitivities of all points

def Probability(Sensitivies):
    Probabilities = []
    SenSum = sum(Sensitivies)
    for i in range(len(Sensitivies)):
        Probabilities.append(Sensitivies[i]/SenSum)
    
    return Probabilities # returns array of probabilities of all points

def Weight(m,Probabilities):
    Weights = []
    for i in range(len(Probabilities)):
        Weights.append(1/(m*Probabilities[i]))
    
    return Weights # returns an array of weights of each point

def CorestSample(m,X,Probabilities,Weights): # constructs the corest using importance sampling as described in the last line of algo 2
    normalized_weights = [w / sum(Weights) for w in Weights]
    cumulative_weights = [sum(normalized_weights[:i+1]) for i in range(len(X))]
    sorted_indices = sorted(range(len(X)), key=lambda i: Probabilities[i], reverse=True)

    selected_elements = []
    r = random.uniform(0, 1)
    for index in sorted_indices:
        if cumulative_weights[index] > r:
            selected_elements.append(X[index])
            r -= cumulative_weights[index]
            if len(selected_elements) == m:
                break
    
    return selected_elements # returns an array of our coreset




# Fix centers
k = 3
# Displaying centers
# cluster_centers = Dsq(array, k)
# print(cluster_centers)

# Fixing alpha
al = 16*(np.log(k) + 2)

# Finding closest points
#Clusters =  