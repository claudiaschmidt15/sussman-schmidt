### Back Propagation ##
- provides an expression for how fast the cost changes when the weights/biases change
### Notation
- Weights: w<sup>l<sub>jk
    -  Denotes the connection from k</sup>th neuron in (l-1) layer to the j</sup>th neuron in l</sup>th layer
- Baises/Activations: 
    - b<sup>l</sub>j (bias of j</sup>neuron in l</sup>th layer)
    - a<sup>l</sub>j 
        - activation of j</sup>th neuron in l<sup>th layer)
    - activations in the l</sup>th layer are related to activations in the (l-1)</sup>th layer
- writing the equation for a in matrix form 
    - define a weight matrix and bias matrix for each layer (l)
    - vectorization: need to apply the function sigma to each element in the vector 
    - final result: a1 = σ(w1a(l-1) + b1)

- Weighted input (z<sup>l)
    - related to the neurons in layer l
    - z</sup>1 = w1al-1 + bl
    - can write a1 = σ(z<sup>1)

Assumptions About the Cost Function 
- It can be written as an average of the sum of all the cost functions of the individual training examples 
    - How does this relate to back propagation? 
        - Backpropagation computes the partial derivatives of the cost function individual training examples and then averages over all the training examples 
- It can be written as a function of the outputs 

Hadamard Product 
- s⊙t : elementwise product of the two vectors

Four Fundamental Equations Behind Backpropagation
