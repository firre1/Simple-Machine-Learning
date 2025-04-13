#!/usr/bin/env python
# coding: utf-8

# Some required imports.
# Make sure you have these packages installed on your system.
import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt
import random as rd

# Distance function used by the kmeans algorithm (euklidean distance)
def distance(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))

# This method contains the implementation of the kmeans algorithm, which 
# takes following input parameters:
#   data_points - A Pandas DataFrame, where each row contains an x and y coordinate.
#   termination_tol - Terminate when the total distance (sum of distance between each datapoint
#                     and its centroid) is smaller than termination_tol. 
#   max_iter - Stop after maximum max_iter iterations.
#
# The method should return the following three items
#   centroids - Pandas Dataframe containing centroids
#   data_points - The data_points with added cluster column
#   total_dist - The total distance for the final clustering
#
# return centroids, data_points, total_dist
#
#  
def kmeans(data_points, num_clusters, termination_tol, max_iter):
    """
    Fills in the kmeans() function from the skeleton code.
    """
    # Convert data_points to numpy arrays
    points = data_points[['x', 'y']].values

    # Randomly choose initial centroids from the data points
    np.random.seed(42)  # for reproducibility
    random_indices = np.random.choice(len(points), size=num_clusters, replace=False)
    centroids = points[random_indices, :]

    # Keep track of old total distance for convergence checks
    prev_total_dist = float('inf')
    total_dist = 0.0

    # For storing cluster assignments
    clusters = np.zeros(len(points), dtype=int)

    for iteration in range(max_iter):
        # Step 1: Assign each point to nearest centroid
        for i, p in enumerate(points):
            # Find closest centroid
            min_dist = float('inf')
            closest_cluster = 0
            for c_idx, c in enumerate(centroids):
                d = distance(p, c)
                if d < min_dist:
                    min_dist = d
                    closest_cluster = c_idx
            clusters[i] = closest_cluster

        # Step 2: Recompute centroids
        new_centroids = np.copy(centroids)
        for c_idx in range(num_clusters):
            cluster_points = points[clusters == c_idx]
            if len(cluster_points) > 0:
                new_centroids[c_idx] = cluster_points.mean(axis=0)
        centroids = new_centroids

        # Step 3: Calculate total distance of points to assigned centroids
        total_dist = 0.0
        for i, p in enumerate(points):
            total_dist += distance(p, centroids[clusters[i]])

        # Step 4: Check convergence
        if abs(prev_total_dist - total_dist) < termination_tol:
            break
        prev_total_dist = total_dist

    # Convert centroids to DataFrame
    centroids_df = pd.DataFrame(centroids, columns=['x', 'y'])

    # Add cluster labels to the original DataFrame (copy to avoid altering the original)
    data_points_with_clusters = data_points.copy()
    data_points_with_clusters['cluster'] = clusters

    # Return the required values
    return centroids_df, data_points_with_clusters, total_dist


# Test elbow method using this code

# Read data points from csv file
data_points = pd.read_csv("cluster_data_points.csv")

# Set termination criteria
termination_tol = 0.001
max_iter = 100

# Plot random data using matplotlib
fig, ax = plt.subplots()
ax.scatter(data_points['x'], data_points['y'], c='black')
plt.title("Data points")
plt.show()

num_clusters_to_test = 15
total_dist_elbow = []

for k in range(1,num_clusters_to_test+1):
    kmeans_output = kmeans(data_points, k, termination_tol, max_iter)
    total_dist_elbow.append(kmeans_output[2])
    
#Plot elbow curve
plt.plot(list(range(1,num_clusters_to_test+1)), total_dist_elbow, marker='o')
plt.title("Elbow method")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Total distance")
plt.show()

# Plot clusters for different values of k using this code
data_points = pd.read_csv("cluster_data_points.csv")

termination_tol = 0.001
max_iter = 100

for k in range(1,11):
    centroids_df, data_points_labeled, total_dist = kmeans(data_points, k, termination_tol, max_iter)
    
    fig, ax = plt.subplots()
    # Plot centroids
    ax.scatter(centroids_df['x'], centroids_df['y'], c='black', marker='*', s=200)

    for centroid_id in range(k):
        points_in_cluster = data_points_labeled.loc[data_points_labeled['cluster'] == centroid_id]
        ax.scatter(points_in_cluster['x'], points_in_cluster['y'])

    plt.title("Clusters for k=" + str(k))
    plt.show()
