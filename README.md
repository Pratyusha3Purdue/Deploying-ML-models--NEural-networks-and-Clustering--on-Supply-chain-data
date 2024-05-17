# Suggesting Improvements in Supply chain using Machine learning and Linear Programming
**1.** We used linear programming to determine the optimal number of units to order for each Product
type and also for each SKU (Stock Keeping Unit) in a supply chain to minimize costs.

**Objective** is to minimize the total purchasing cost, which is calculated as the sum of products
purchased multiplied by their respective prices.

**Constraints** were to meet demand, where each SKU must have enough inventory to meet the
sales demand.

**Decision variables** : Non-Negative Orders  (number of units ordered for each
SKU) are constrained to be non-negative (you can't order a negative number of items).

**2.** Neural network implementation to Predict Delays
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

● Forward Propagation:  Calculate the predicted output (outputs = sigmoid(np.dot(input_layer, weights))) based on the current weights.
● Error Calculation: Determine the difference between the actual labels and the predicted labels (error = y_train - outputs.squeeze()). 
● Backpropagation: Compute the gradient of the loss function with respect to the weights. This involves taking the derivative of the sigmoid function (sigmoid_derivative(outputs)) and adjusting the weights based on the gradient (weights += np.dot(input_layer.T, adjustments)).

After training, the network uses the learned weights to make predictions on the test data, based on the probability that a given shipment will be delayed.
We were able to predict delays accurately 70% of the time.

