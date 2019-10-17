import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1] # C
  num_train = X.shape[0] # N
  loss = 0.0

  for i in xrange(num_train): # N
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    for j in xrange(num_classes): # C

      # correct class
      if j == y[i]:
        continue

      # incorrect class
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

        # gradient for incorrect class which did not meet margin
        dW[:,j] += X[i]
        # refer to derivation of optimisation notes
        dW[:,y[i]] -= X[i]

      # gradient for incorrect class which met margin = 0, so do nothing

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # ^same thing for gradient
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # ^same thing for gradient
  # why is gradient of np.sum(W**2) wrt W = W??
  # what about np.sum(W)?
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  N = X.shape[0]
  
  vec_scores = X.dot(W) # shape = (N, C)
  vec_correct_classes_scores = vec_scores[np.arange(N), y] # shape = (N, )

  # element-wise maximum
  margins = np.maximum(vec_scores - vec_correct_classes_scores.reshape(N,1) + 1, 0) # shape = (N, C)
  margins[np.arange(N), y] = 0

  loss = np.sum(margins)/N
  loss += 0.5 * reg * np.sum(W * W) # where did 0.5 come from??
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # source: https://github.com/yunjey/cs231n/blob/master/assignment1/cs231n/classifiers/linear_svm.py

  d_scores = np.zeros_like(vec_scores) # shape = (N, C)
  d_scores[margins > 0] = 1 # indicator function
  d_scores[np.arange(N),y] -= np.sum(d_scores, axis=1)

  dW = X.T.dot(d_scores)
  dW /=N
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
