import numpy as np

class NN_trial:
    """
    NN. Tanh activation functions, also on the output layer. This means the output will be between -1 and 1
    """

    def __init__(self, N_layers: int, N_nodes: int, input_dim = 1, output_dim = 1):
        
        self.input_dim = input_dim
        self.N_nodes   = [input_dim] + [N_nodes] * N_layers + [output_dim] # number of nodes in each layer. The first layer corresponds to the entry data, the last one to the output

        self.weights  = []
        self.bs       = []
        self.N_layers = N_layers

        # initialise the weights (Glorot initialisation) and the intercept (good for tanh/sigmoid)
        for k in range(N_layers+1):
            a = np.sqrt(6./(self.N_nodes[k] + self.N_nodes[k+1]))# half-length of the interval from which we draw the uniform distribution
            self.weights.append(np.random.uniform(-a, a, size = ((self.N_nodes[k+1], self.N_nodes[k])) ))
            self.bs.append(np.zeros(self.N_nodes[k+1])) # the intercept can be initialized to 0

        # initialize velocities for training loop
        self.weights_v = [np.zeros_like(w) for w in self.weights]
        self.bs_v      = [np.zeros_like(b) for b in self.bs]

        pass


    def predict(self, input_coordinates):
        """
        Forward pass, find predicted value of the solution
        """

        prediction = input_coordinates
        for k in range(self.N_layers + 1):
            prediction = np.tanh(np.dot(self.weights[k], prediction) + self.bs[k])

        return prediction
    

    def forward_pass(self, input_coordinates):
        """
        Forward pass, find predicted value of the solution. Store outputs of the nodes
        """

        nodes_output = [input_coordinates]

        for k in range(self.N_layers + 1):
            nodes_output.append(np.tanh(np.dot(self.weights[k], nodes_output[-1]) + self.bs[k]))

        return nodes_output
    
    
    def compute_gradients_one_data_point(self, input, target_output):
        """
        Compute gradients via backpropagation. Quadratic loss. Only works with scalar output. One single datapoint
        """
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        dE_over_dnet = [np.zeros_like(b) for b in self.bs] 
        # derivatives of the loss wrt the arguments of the activation functions at each node
        # dE_over_dnet is by construction equal to dE_over_db (the derivative of the error wrt the intercepts)

        nodes_outputs = self.forward_pass(input)
        output = nodes_outputs[-1][0]
        loss = 0.5 * (output - target_output)**2

        # first compute the gradients for the output layer
        dE_over_dnet[-1]     = (output - target_output)* (1 - output ** 2)
        weight_gradients[-1] = np.outer(dE_over_dnet[-1], nodes_outputs[-2])

        # now go over the inner layers, backwards
        for k in range(1, self.N_layers + 1):
            dE_over_dnet[-1-k]     = (1 - nodes_outputs[-1-k]**2) * np.dot(np.transpose(self.weights[-k]), dE_over_dnet[-k])
            weight_gradients[-1-k] = np.outer(dE_over_dnet[-1-k], nodes_outputs[-2-k])

        return loss, dE_over_dnet, weight_gradients
    

    def train_one_data_point(self, input, target_output, lr = 0.02):
        """update the weights, only one data point. lr: learning rate"""

        loss, bs_gradients, weight_gradients = self.compute_gradients_one_data_point(input, target_output)
        
        # use the given learning rate
        for k in range(len(bs_gradients)):
            self.bs[k] -= lr * bs_gradients[k]
            self.weights[k] -= lr * weight_gradients[k]

        return loss
    
    def train(self, input_batch, target_output_batch, lr = 0.02, beta = 0.9):
        """train with multiple data. For the moment no mini-batches, as the data size will not be too large"""

        if len(input_batch) != len(target_output_batch):
            print("Error: input and output data should have the same size")
            pass

        data_size = len(input_batch)
        weight_gradients = [np.zeros_like(ww) for ww in self.weights]
        bs_gradients = [np.zeros_like(bb) for bb in self.bs] 
        loss = 0

        # average the gradients over all data
        for j in range(data_size):
            l, db, dw = self.compute_gradients_one_data_point(input_batch[j], target_output_batch[j])
            loss += l
            for k in range(len(bs_gradients)):
                bs_gradients[k] += db[k] / data_size
                weight_gradients[k] += dw[k] / data_size

        loss = loss / data_size # normalize to get average loss 

        # use the given learning rate
        for k in range(len(bs_gradients)):
            self.bs_v[k] = beta * self.bs_v[k] - lr * bs_gradients[k] 
            self.bs[k] += self.bs_v[k]
            self.weights_v[k] = beta * self.weights_v[k] - lr * weight_gradients[k] 
            self.weights[k] += self.weights_v[k]

        return loss