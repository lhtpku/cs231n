import numpy as np
from random import shuffle
def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    score = X[i].dot(W)
    tmp_sum = np.exp(score).sum()
    loss += -score[y[i]] + np.log(tmp_sum)

    for j in range(num_classes):
      dW[:,j] += np.exp(score[j]) * X[i] / tmp_sum
      if j == y[i]:
        dW[:,j] -= X[i]

  loss /= num_train 
  loss += 0.5 *reg * np.sum(W*W)
  dW /= num_train
  dW += 1 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores_exp = np.exp(scores)
  loss = -scores[np.arange(num_train),y].sum() + np.log(scores_exp.sum(axis=1)).sum()
  loss /= num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  margins = scores_exp / np.sum(scores_exp,axis=1)[:,np.newaxis]
  margins[np.arange(num_train),y] -= 1
  dW = np.dot(X.T,margins) / num_train
  dW += 2 * reg * W
  return loss, dW

