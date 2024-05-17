# Suggesting Improvements in Supply chain using Machine learning and Linear Programming
**1.** We used **linear programming** to determine the optimal number of units to order for each Product
type and also for each SKU (Stock Keeping Unit) in a supply chain to minimize costs.

**Objective** is to minimize the total purchasing cost, which is calculated as the sum of products
purchased multiplied by their respective prices.

**Constraints** were to meet demand, where each SKU must have enough inventory to meet the
sales demand.

**Decision variables** : Non-Negative Orders  (number of units ordered for each
SKU) are constrained to be non-negative (you can't order a negative number of items).

**2.** **Neural network** implementation to Predict Delays
Built a model for predicting delays in shipments, using neural networks.

WE used 80% data for training the model and 20% to test the model.

**Components of the Neural Network in the Code:**

**Input Layer:** The features (Shipping times, Lead times, Shipping costs) are treated as input neurons. 

**Weights:** The code initializes a set of weights (weights = 2 * np.random.random((X_train.shape[1], 1)) - 1) that connect each input neuron to the output neuron. 

**Activation Function:** A sigmoid function is used as the activation function (def sigmoid(x): return 1 / (1 + np.exp(-x))), since it is a  binary classification of delay.

**Output Layer:** The output layer consists of a single neuron that uses the sigmoid activation function to produce the final prediction. 
This output can be interpreted as the probability of a shipment being delayed.

**Training Process:** The network is trained using a simple form of the gradient descent algorithm. 
In each iteration of the training loop, the following steps occur:

● Forward Propagation:  Calculate the predicted output based on the current weights.
● Error Calculation: Determine the difference between the actual labels and the predicted labels. 
● Backpropagation: Compute the gradient of the loss function with respect to the weights. This involves taking the derivative of the sigmoid function and adjusting the weights based on the gradient.

After training, the network uses the learned weights to make predictions on the test data, based on the probability that a given shipment will be delayed.
We were able to predict delays **accurately 70%** of the time.

**3.** We applied the **K-means clustering** technique to cluster the suppliers into three groups according to their lead time and defect rates. This model leverages the balance of product quality and efficiency and could potentially help the company to choose more proper suppliers in the future.

We first performed **Data Encoding and Normalization:**
Categorical features ('Supplier name,' 'Location,' and 'Transportation modes') were converted to numerical values using Label Encoding.
Numerical features ('Lead times' and 'Defect rates') were standardized using StandardScaler to ensure they have a mean of 0 and a standard deviation of 1.

**K-means Clustering** was applied to the processed features, grouping the data into 3 clusters. Each data point (observation) was assigned a cluster label, which was added to the original dataset.

Dimensionality Reduction with **Principal Component Analysis (PCA)**:
PCA was performed on the clustering features to reduce the data from its original dimensionality to 2 principal components(most significant).


A **scatter plot** was created using the two principal components (PC1 = 'Supplier and Transportation.' and PC2 ='Geographic and Defect rates.') as the axes, with colors representing the different clusters.

![image](https://github.com/Pratyusha3Purdue/USing-NEural-networks-and-CLustering-to-Improve-Supply-chain/assets/141969918/f0c861f1-289c-4591-8de2-83de06fa18c7)


**Cluster Insights:**
Yellow Cluster: Observations in this cluster showed a wide range of supplier and transportation variations but were mostly around lower geographic and defect rate variations.
Green Cluster: Observations in this cluster displayed moderate variations in both supplier and transportation factors as well as geographic and defect rate factors.
Purple Cluster: Observations in this cluster had high supplier and transportation variations and a wide range of geographic and defect rate variations.

