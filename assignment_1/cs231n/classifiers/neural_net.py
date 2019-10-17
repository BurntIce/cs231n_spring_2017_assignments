from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    # let 'q' stands for intermediate
    # let 'h' stands for hidden layer
    
    # X's shape: (N, D)
    # W1: First layer weights; has shape (D, H)
    # b1: First layer biases; has shape (H,)

    q1 = X.dot(W1) # shape: (N, H)
    q2 = q1 + b1 # shape: (N, H)

    # ReLU
    h = np.maximum(q2, 0) # shape: (N, H)

    # W2: Second layer weights; has shape (H, C)
    # b2: Second layer biases; has shape (C,)

    q3 = h.dot(W2) # shape: (N, C)
    scores = q3 + b2 # shape: (N, C)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    stabilised_scores = scores - np.max(scores, axis=1, keepdims=True) # shape: (N, C)
    exp_scores = np.exp(stabilised_scores) # shape: (N, C)

    softmax_prob = exp_scores/np.sum(exp_scores, axis=1, keepdims=True) # shape: (N, C)

    # TODO: figure out softmax_prob[np.arange(N), y] (or softmax_prob[range(N), y])
    loss_i = -np.log(softmax_prob[np.arange(N), y]) # shape: (N, C)
    # print(loss_i)

    loss = np.sum(loss_i)/N # scalar

    loss += reg * (np.sum(W1**2) + np.sum(W2**2)) # scalar
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    dL_dscores = softmax_prob
    dL_dscores[np.arange(N), y] -= 1 # why???
    dL_dscores /= N # why???; shape: (N, C)

    # using computational graph,

    # local gradients at q3 + b2
    dscores_dq3 = 1
    dscores_db2 = 1

    dL_dq3 = dL_dscores.dot(dscores_dq3) # shape: (N, C)
    dL_db2 = np.sum(dL_dscores.dot(dscores_db2), axis=0) # shape: (C, )

    grads["b2"] = dL_db2

    # local gradients at h.dot(W2)
    dq3_dh = W2 # shape: (H, C)
    dq3_dW2 = h # shape: (N, H)

    dL_dh = dL_dq3.dot(dq3_dh.T) # shape: (N, H)
    dL_dW2 = dq3_dW2.T.dot(dL_dq3) # shape: (H, C)

    grads["W2"] = dL_dW2

    # gradient after backprop through np.maximum(q2, 0); ReLU
    dL_dq2 = dL_dh # shape: (N, H)
    dL_dq2[h <= 0] = 0 # TODO: brush up on numpy indexing

    # local gradients at q1 + b1
    dq2_dq1 = 1
    dq2_db1 = 1

    dL_dq1 = dL_dq2.dot(dq2_dq1) # shape: (N, H)
    dL_db1 = np.sum(dL_dq2.dot(dq2_db1), axis=0) # shape: (H, )

    grads["b1"] = dL_db1

    # relevant local gradient at X.dot(W1)
    dq1_dW1 = X # shape: (N, D)

    dL_dW1 = dq1_dW1.T.dot(dL_dq1) # shape (D, H)

    grads["W1"] = dL_dW1

    grads["W1"] += reg * 2 * W1
    grads["W2"] += reg * 2 * W2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      batch_indices = np.random.choice(num_train, batch_size)
      X_batch = X[batch_indices]
      y_batch = y[batch_indices]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      for key in self.params:
        self.params[key] -= learning_rate * grads[key]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = np.mean((self.predict(X_batch) == y_batch))
        val_acc = np.mean((self.predict(X_val) == y_val))
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    # W1: First layer weights; has shape (D, H)
    # b1: First layer biases; has shape (H,)
    # W2: Second layer weights; has shape (H, C)
    # b2: Second layer biases; has shape (C,)
    scores = np.maximum(X.dot(self.params["W1"]) + self.params["b1"], 0)
    scores = scores.dot(self.params["W2"]) + self.params["b2"]
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


