### Back Propagation ##
- provides an expression for how fast the cost changes when the weights/biases change
### Notation
- Weights: w<sup>l</sup><sub>jk</sub>
    -  Denotes the connection from k<sup>th</sup> neuron in (l-1)<sup>th</sup> layer to the j<sup>th</sup> neuron in l<sup>th</sup> layer
- Baises/Activations: 
    - b<sup>l</sup><sub>j</sub> : bias of the j<sub>th</sub> neuron in the l<sup>th</sup> layer
    - a<sup>l</sup><sub>j</sub> 
        - activation of j<sup>th</sup> neuron in l<sup>th</sup> layer)
    - activations in the neuron in the l<sup>th</sup> layer layer are related to activations in the (l-1)<sup>th</sup> layer 
- writing the equation for a in matrix form 
    - define a weight matrix and bias matrix for each layer (l)
    - vectorization: need to apply the function sigma to each element in the vector 
    - final result: a<sup>1</sup> = σ(w<sup>1</sup>a<sup>(l-1)</sup> + b<sup>1</sup>)

- Weighted input (z<sup>l)
    - related to the neurons in layer l
    - z<sup>1</sup> = w<sup>1</sup>v a<sup>l-1</sup> + b<sup>l</sup>
    - can write a1 = σ(z<sup>1)

### Assumptions About the Cost Function 
- It can be written as an average of the sum of all the cost functions of the individual training examples 
    - How does this relate to back propagation? 
        - Backpropagation computes the partial derivatives of the cost function individual training examples and then averages over all the training examples 
- It can be written as a function of the outputs 

### Hadamard Product 
- s⊙t : elementwise product of the two vectors

### Four Fundamental Equations Behind Backpropagation

- Error (intermediate value): δ<sup>l</sup><sub>j</sub>
    - Occurs at the j neuron in l layer and slightly changes the weighted input which affects later layers
    - helps in minimizing the cost
    - partical derivative of C with respect to z
    - via backpropagation, we can evaluate all values for the error at every layer and then use those errors 

- the 4 equations allow us to compute the error and the gradient of C

- Error equation in output layer 
    - includes the partial derivative of C with respect to a (measures how fast the cost function is changing with respect to the j<sup>th</sup> output activation)
        - if C doesn't depend on that output neuron, then the error is very small 
    - also includes how fast the activation function is changing with respect to z

 ### Backpropagation Algorithm 
 



