# INPUTS:
# epochs - number of epochs to train for 
# mini_batch_size - size of mini batches used when sampling 
# etta - learnin rate 

def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data) #randomly shuffles the training data in each opoch
            
            #divides the data into mini batches and applies a single step of gradient descent
                #does this using the function update_mini_batch 
                #update_mini_batch computes the gradients for each training example and updates 
                    #the weight and biases
            
            #backprop function is called in update_mini_batch function which computes gradient 
                #of the cost function
            mini_batches = [ 
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)



def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        #creating two arrays in the shape of the biases and weights 
        nabla_b = [np.zeros(b.shape) for b in self.biases] 
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b #calculating z vector 
            zs.append(z) #adding newest calculated z vector to list of zs
            activation = sigmoid(z)
            activations.append(activation) #adding newest calulcated activation value to the list 

        # backward pass
        # delta - calculation for calculating the error
        delta = self.cost_derivative(activations[-1], y) * \ 
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta #error for biases
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #error or weights 

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w) 

        #gives output of updated weights and biases