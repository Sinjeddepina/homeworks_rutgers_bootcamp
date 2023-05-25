# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from path import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
In [ ]:
# Load the data into a Pandas DataFrame
df_market_data = pd.read_csv(
    Path("Resources/crypto_market_data.csv"),
    index_col="coin_id")

# Display sample data
df_market_data.head(10)
In [ ]:
# Generate summary statistics
df_market_data.describe()
In [ ]:
# Plot your data to see what's in your DataFrame
df_market_data.hvplot.line(
    width=800,
    height=400,
    rot=90
)

# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
scaled_data = StandardScaler().fit_transform(df_market_data)

# Create a DataFrame with the scaled data
df_market_data_scaled = pd.DataFrame(
    scaled_data,
    columns=df_market_data.columns
)

# Copy the crypto names from the original data
df_market_data_scaled["coin_id"] = df_market_data.index

# Set the coinid column as index
df_market_data_scaled = df_market_data_scaled.set_index("coin_id")

# Display sample data
df_market_data_scaled.head()



# Create a list with the number of k-values to try
# Use a range from 1 to 11
# YOUR CODE HERE!
k_values = list(range(1, 12))
print(k_values)

# Create an empy list to store the inertia values
# YOUR CODE HERE!
inertia_values = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_scaled`
# 3. Append the model.inertia_ to the inertia list
# YOUR CODE HERE!
inertia_values = []

for k in k_values:
   
    kmeans_model = KMeans(n_clusters=k)
    
    
    kmeans_model.fit(df_market_data_scaled)
    
    inertia_values.append(kmeans_model.inertia_)





# Create a DataFrame with the data to plot the Elbow curve
# YOUR CODE HERE!
elbow_data = {"k_values": k_values, "inertia_values": inertia_values}

# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
# YOUR CODE HERE!

plt.plot(k_values, inertia_values, marker='o')


plt.title("Elbow Curve")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")


plt.show()

# Initialize the K-Means model using the best value for k
# YOUR CODE HERE!
kmeans_model = KMeans(n_clusters=best_k)


kmeans_model.fit(df_market_data_scaled)

# Fit the K-Means model using the scaled data
# YOUR CODE HERE!

kmeans_model.fit(df_market_data_scaled)

# Predict the clusters to group the cryptocurrencies using the scaled data
# YOUR CODE HERE!

clusters = kmeans_model.predict(df_market_data_scaled)

# View the resulting array of cluster values.
# YOUR CODE HERE!

print(clusters)



# YOUR CODE HERE!
df_market_data_copy = df_market_data.copy()

# Add a new column to the DataFrame with the predicted clusters
# YOUR CODE HERE!

df_market_data_copy['cluster'] = clusters


# Display sample data
# YOUR CODE HERE!
sample_data = df_market_data_copy.head()
print(sample_data)

# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
# YOUR CODE HERE!
df_market_data_copy.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    c="cluster",
    colormap="viridis",
    hover_cols=["coin_id"],
    title="Cryptocurrency Clusters"
)
# Create a PCA model instance and set `n_components=3`.
# YOUR CODE HERE!
pca_model = PCA(n_components=3)

# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
# YOUR CODE HERE!

principal_components = pca_model.fit_transform(df_market_data_scaled)

# View the first five rows of the DataFrame. 
# YOUR CODE HERE!

first_five_rows = df_market_data_copy.head()
print(first_five_rows)

# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
# YOUR CODE HERE!


explained_variance = pca_model.explained_variance_ratio_
print(explained_variance)

# Create a new DataFrame with the PCA data.

pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
# Note: The code for this step is provided for you

# Creating a DataFrame with the PCA data
# YOUR CODE HERE!
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])

# Copy the crypto names from the original data
# YOUR CODE HERE!

crypto_names = df_market_data.index.tolist()

# Set the coinid column as index
# YOUR CODE HERE!
pca_df.set_index('coinid', inplace=True)

# Display sample data
# YOUR CODE HERE!

pca_df.head()


# Create a list with the number of k-values to try
# Use a range from 1 to 11
# YOUR CODE HERE!
k_values = list(range(1, 11))

# Create an empy list to store the inertia values
# YOUR CODE HERE!

inertia_values = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list
# YOUR CODE HERE!

or k in k_values:
   
    kmeans_model = KMeans(n_clusters=k, random_state=0)
    
    
    kmeans_model.fit(df_market_data_pca)
    
   
    inertia_values.append(kmeans_model.inertia_)

# Create a dictionary with the data to plot the Elbow curve
# YOUR CODE HERE!

elbow_data = {"k": k_values, "inertia": inertia_values}

# Create a DataFrame with the data to plot the Elbow curve
# YOUR CODE HERE!

df_elbow = pd.DataFrame({'k': k_values, 'inertia': inertia_values})



# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
# YOUR CODE HERE!

df_elbow.hvplot.line(x='k', y='inertia', title='Elbow Curve', xlabel='k', ylabel='Inertia')

# Initialize the K-Means model using the best value for k
# YOUR CODE HERE!

kmeans = KMeans(n_clusters=k, random_state=42)

# Fit the K-Means model using the PCA data
# YOUR CODE HERE!

kmeans.fit(df_market_data_pca)


# Predict the clusters to group the cryptocurrencies using the PCA data
# YOUR CODE HERE!

clusters = kmeans.predict(df_market_data_pca)

# View the resulting array of cluster values.
# YOUR CODE HERE!

print(clusters)

# Create a copy of the DataFrame with the PCA data
# YOUR CODE HERE!

df_pca_copy = df_market_data_pca.copy()

# Add a new column to the DataFrame with the predicted clusters
# YOUR CODE HERE!

df_pca_copy['cluster'] = clusters

# Display sample data
# YOUR CODE HERE!

print(df_pca_copy.head())

# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
# YOUR CODE HERE!

df_pca_copy.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    by="cluster",
    hover_cols=["coin_id"],
    cmap="Set1"
)


# Composite plot to contrast the Elbow curves
# YOUR CODE HERE!

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(range(1, 11), inertia_values_original, marker='o', label='Original Data')

ax.plot(range(1, 11), inertia_values_pca, marker='o', label='PCA Data')

ax.set_title('Elbow Curve - Original Data vs. PCA Data')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia')

ax.legend()

plt.show()
# Compoosite plot to contrast the clusters
# YOUR CODE HERE!

import matplotlib.pyplot as plt

# Create a figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the scatter plot for the original data
ax.scatter(
    df_market_data['price_change_percentage_24h'],
    df_market_data['price_change_percentage_7d'],
    c=df_market_data['cluster_original'],
    cmap='viridis',
    label='Original Data'
)

# Plot the scatter plot for the PCA data
ax.scatter(
    df_market_data_pca['component1'],
    df_market_data_pca['component2'],
    c=df_market_data_pca['cluster_pca'],
    cmap='viridis',
    label='PCA Data'
)

# Set the plot title and labels
ax.set_title('Cluster Visualization - Original Data vs. PCA Data')
ax.set_xlabel('Price Change Percentage 24h')
ax.set_ylabel('Price Change Percentage 7d')

# Add a legend
ax.legend()

# Show the plot
plt.show()
