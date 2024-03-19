import sys
import numpy as np
import matplotlib.pyplot as plt

# Function to determine euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Calculate SSE using the formula
def calculate_sse(data, centroids, cluster_ID):
    sse = 0
    for i in range(len(data)):
        sse += euclidean_distance(data[i], centroids[cluster_ID[i]])
    return sse

# K means clustering
def kmeans(data, k, iterations=20):

    # SEt the seed for reproducability - so that values wont change each time code is run!
    # if we want to change value of random seed it can be done by replacing the value in paranthesis below
    np.random.seed(42)

    # We extract the number of rows and columns 
    # For example, for pendigits dataset, there are 7494 samples
    # and 16 features
    num_samples, num_features = data.shape

    # Randomly assign K centroid points
    cluster_ID = np.random.randint(0, k, num_samples)
    
    # For 20 iterations, do this
    for _ in range(iterations):

        # Re-calculate the centroids using the newly created clusters.
        centroids = np.array([data[cluster_ID == i].mean(axis=0) for i in range(k)])

        # For each data point, iterate and do
        for i in range(num_samples):

            # Calculate the Euclidean distance to all current centroids.
            distances = [euclidean_distance(data[i], centroid) for centroid in centroids]

            # Assign each data point to its nearest centroid to create K clusters
            # np.argmin determines minimum euclidean distance
            cluster_ID[i] = np.argmin(distances)
    
    sse = calculate_sse(data, centroids, cluster_ID)
    return sse

# Main function
def main(data_file):

    # Load the dataset 
    data = np.loadtxt(data_file)
    
    # Initialize K, number of clusters
    # This needs to be 2 to 10 values
    k_values = list(range(2, 11)) 

    # Define an empty list to store SSE values for each K
    sse_values = []

    # For each value of k, run the K-means clustering
    for k in k_values:

        # We ignore the last column of the data [class labels] for clustering using 
        # the format: data[:, :-1]
        sse = kmeans(data[:, :-1], k)

        # Append the values to the empty list for plotting
        sse_values.append(sse)
        print(f"For k = {k} After 20 iterations: SSE error = {sse:.4f}")
    
    # Use matplotlib to plot the K vs SSE graph
    plt.plot(k_values, sse_values, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('SSE Error')
    plt.title('SSE vs K')
    plt.show()

# This is the starting point of the program.
if __name__ == "__main__":

    # Display an error if the user does not provide the file path in the argument
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <data_file>")
        sys.exit(1)
    
    # Take up the file path name and call tha main function
    data_file = sys.argv[1]
    main(data_file)