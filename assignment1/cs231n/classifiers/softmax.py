import numpy as np

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(len(X)):
      f = X[i].dot(W)
      f -= np.max(f)
      denom = np.sum(np.exp(f))
      loss += -f[y[i]] + np.log(denom)
      
      for j in range(W.shape[1]):
          p = np.exp(f[j]) / denom
          dW[:, j] += X[i, :] * (p - (j == y[i]))
  
  loss /= len(X)
  dW /= len(X)
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = X.dot(W)
  f -= np.array([np.max(f, axis=1)]).T
  f_exp = np.exp(f)
  f_sum = np.sum(f_exp, axis=1)
  loss = -np.sum(f[range(X.shape[0]), y]) + np.sum(np.log(f_sum))
  
  p = f_exp / np.array([f_sum]).T
  p[range(len(X)), y] -= 1
  dW = X.T.dot(p)
  
  loss /= len(X)
  dW /= len(X)
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

