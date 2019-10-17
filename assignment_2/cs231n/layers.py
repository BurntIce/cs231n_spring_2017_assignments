from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # out = w*x + b
    reshaped_x = np.reshape(x, (x.shape[0], -1)) # shape (N, D)
    out = reshaped_x.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # since out = w*x + b, therefore:

    # dout_dw = x
    reshaped_x = np.reshape(x, (x.shape[0], -1)) # shape (N, D)
    dw = reshaped_x.T.dot(dout)

    # dout_dx = w
    dx = dout.dot(w.T).reshape(x.shape)

    # dout_db = 1
    db = np.sum(dout, axis=0) * 1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(x, 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # how does numpy works?!
    # https://github.com/haofeixu/stanford-cs231n-2018/blob/master/assignment2/cs231n/layers.py
    mask = x > 0
    dx = dout * mask
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        # m = 200 # batch size
        # batch_index = np.random.choice(N, size=m)
        
        # start of computational graph
        # mean and variance are batch mean and variance respectively

        # batch = x[batch_index]
        #mean = np.sum(x[batch_index], axis=0) / m # shape: (D, )

        # batch norm does not seem to work
        # have to do full norm
        # perhaps I misunderstood what is meant by 'batch'

        mean = np.sum(x, axis=0) / N

        x_minus_mean = x - mean # shape: (N, D)
        square = x_minus_mean**2 # shape: (N, D)
        # variance = np.sum(square[batch_index], axis=0) / m # shape: (D, )
        variance = np.sum(square, axis=0) / N

        variance_plus_eps = variance + eps # shape: (D, )
        sqrt = np.sqrt(variance_plus_eps) # shape: (D, )
        inverse = 1.0/sqrt # shape: (D, )

        x_hat = x_minus_mean * inverse # shape: (N, D)

        gamma_times_xhat = gamma * x_hat # shape: (N, D)
        out = gamma_times_xhat + beta # shape: (N, D)

        # end of computational graph

        running_mean = momentum * running_mean + (1.0 - momentum) * mean
        running_var = momentum * running_var + (1.0 - momentum) * variance

        cache = (x, # shape: (N, D)
                # m, # shape: scalar
                x_minus_mean, # shape: (N, D)
                square, # shape: (N, D)
                variance_plus_eps, # shape: (D, )
                sqrt, # shape: (D, )
                inverse, # shape: (D, )
                x_hat, # shape: (N, D)
                gamma) # shape: (D, )
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    # (x, m, x_minus_mean, square, variance_plus_eps, \ # m shape: scalar
    (x, x_minus_mean, square, variance_plus_eps, \
      sqrt, inverse, x_hat, gamma)  = cache
    # x shape: (N, D)
    # x_minus_mean, # shape: (N, D)
    # square, # shape: (N, D)
    # variance_plus_eps, # shape: (D, )
    # sqrt, # shape: (D, )
    # inverse, # shape: (D, )
    # x_hat, # shape: (N, D)
    # gamma_times_xhat) # shape: (N, D)

    # local gradients at gamma_times_xhat plus beta
    dout_dbeta = 1.0 # shape: scalar
    dout_d_gamma_times_xhat = 1.0 # shape: scalar

    dbeta = np.sum(dout, axis=0) * dout_dbeta # shape: (D, )
    d_gamma_times_xhat = dout * dout_d_gamma_times_xhat # shape: (N, D)

    # local gradients at gamma times xhat
    d_gamma_times_xhat_dgamma = x_hat # shape: (N, D)
    d_gamma_times_xhat_dxhat = gamma # shape: (D, )

    dgamma = np.sum(d_gamma_times_xhat * d_gamma_times_xhat_dgamma, axis=0) # shape: (D, )
    dxhat = d_gamma_times_xhat * d_gamma_times_xhat_dxhat # shape: (N, D)

    # local gradients at x_minus_mean times inverse
    dxhat_d_x_minus_mean = inverse # shape: (D, )
    dxhat_dinverse = x_minus_mean # shape: (N, D)

    dinverse = np.sum(dxhat * dxhat_dinverse, axis=0) # shape: (D, )

    # local gradient at inverse of sqrt
    dinverse_dsqrt = -1.0/sqrt**2 # shape: (D, )

    dsqrt = dinverse * dinverse_dsqrt # shape: (D, )

    # local gradient at sqrt of variance_plus_eps
    dsqrt_d_var_plus_eps = 0.5 / np.sqrt(variance_plus_eps) # shape: (D, )

    d_var_plus_eps = dsqrt * dsqrt_d_var_plus_eps # shape: (D, )

    # local gradient at variance plus eps
    d_var_plus_eps_dvariance = 1.0 # shape: scalar

    dvariance  = d_var_plus_eps * d_var_plus_eps_dvariance # shape: (D, )

    # local gradient at variance
    dvariance_dsquare = np.ones(square.shape) / square.shape[0] # m # shape: (N, D)

    dsquare = dvariance * dvariance_dsquare # shape: (N, D)

    # local gradient at squaring of x_minus_mean
    dsquare_d_x_minus_mean = 2.0 * x_minus_mean # shape: (N, D)

    # upstream gradient at x minus mean
    d_x_minus_mean = dxhat * dxhat_d_x_minus_mean + dsquare * dsquare_d_x_minus_mean # shape: (N, D)

    # local gradients at x minus mean
    d_x_minus_mean_dx = 1.0 # shape: scalar
    d_x_minus_mean_dmean = -1.0 # shape: scalar

    dmean = np.sum(d_x_minus_mean * d_x_minus_mean_dmean, axis=0) # shape: (D, )

    # local gradient at mean
    dmean_dx = np.ones(x.shape) / x.shape[0] # m # shape: (N, D)

    dx = d_x_minus_mean * d_x_minus_mean_dx + dmean * dmean_dx
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    (_, _, _, _, _, inverse, x_hat, gamma)  = cache
    # x shape: (N, D)
    # x_minus_mean, # shape: (N, D)
    # square, # shape: (N, D)
    # variance_plus_eps, # shape: (D, )
    # sqrt, # shape: (D, )
    # inverse, # shape: (D, )
    # x_hat, # shape: (N, D)
    # gamma_times_xhat) # shape: (N, D)

    N = dout.shape[0]

    # https://kevinzakka.github.io/2016/09/14/batch_normalization/

    # dx = dout * gamma * (-0.5) * x_minus_mean * inverse / sqrt * (np.sum(x**2, axis=0)/N - 2*mean + mean**2)

    dxhat = dout * gamma

    dx = (1. / N) * inverse * (N*dxhat - np.sum(dxhat, axis=0) - x_hat*np.sum(dxhat*x_hat, axis=0))

    dgamma = np.sum(dout * x_hat, axis=0)

    dbeta = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # is *x.shape a pointer to x.shape???
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dout_dx = mask
        dx = dout * dout_dx
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    padded_x = np.pad(x, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0) # shape (N, C, H + 2*pad, W + 2*pad)

    padded_H = padded_x.shape[2]
    padded_W = padded_x.shape[3]

    N, _, H, W = x.shape
    F, _, HH, WW = w.shape

    H_prime = int(1 + (H + 2 * pad - HH) / stride)
    W_prime = int(1 + (W + 2 * pad - WW) / stride)

    out = np.zeros((N, F, H_prime, W_prime))

    # O(N * F * H * W / stride) time complexity; nice
    for data_point in range(N):
      for filter in range(F):
        for height in range(0, padded_H-HH+1, stride):
          for width in range(0, padded_W-WW+1, stride):
            out_height = int(1 + (height + 2 * pad - HH) / stride)
            out_width = int(1 + (width + 2 * pad - WW) / stride)

            # single line
            # out[data_point, filter, out_height, out_width] = np.sum(w[filter] * padded_x[data_point, : , height:height+HH, width:width+WW]) + b[filter]

            # computational graph
            sliced_x = padded_x[data_point, : , height:height+HH, width:width+WW] # shape (C, HH, WW)
            w_x = w[filter] * sliced_x # shape (C, HH, WW)
            sum = np.sum(w_x) # shape scalar
            out[data_point, filter, out_height, out_width] = sum + b[filter] # shape scalar
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives. shape (N, F, H_prime, W_prime)
        H_prime = int(1 + (H + 2 * pad - HH) / stride)
        W_prime = int(1 + (W + 2 * pad - WW) / stride)
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x - shape (N, C, H, W)
    - dw: Gradient with respect to w - shape (F, C, HH, WW)
    - db: Gradient with respect to b - shape (F, )
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']

    padded_x = np.pad(x, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0) # shape (N, C, H + 2*pad, W + 2*pad)

    padded_H = padded_x.shape[2]
    padded_W = padded_x.shape[3]

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    dx = np.zeros(x.shape) # shape (N, C, H, W)
    dw = np.zeros(w.shape) # shape (F, C, HH, WW)
    db = np.zeros(b.shape) # shape (F, )

    dpx = np.zeros(padded_x.shape)  # shape (N, C, H + 2*pad, W + 2*pad)

    # time complexity: O(N * F * H * W / stride)
    for data_point in range(N):
      for filter in range(F):
        for height in range(0, padded_H-HH+1, stride):
          for width in range(0, padded_W-WW+1, stride):
            out_height = int(1 + (height + 2 * pad - HH) / stride)
            out_width = int(1 + (width + 2 * pad - WW) / stride)

            dout_i = dout[data_point, filter, out_height, out_width]

            # dout_db = 1
            # db = dout * dout_db
            db[filter] += dout_i # shape scalar

            # dout_dsum = 1
            # dsum = dout * dout_dsum
            dsum = dout_i # shape scalar

            # distribute gradient of sum to each contributing element
            dsum_dwx = np.ones((C, HH, WW))
            dwx = dsum * dsum_dwx # shape (C, HH, WW)

            sliced_x = padded_x[data_point, : , height:height+HH, width:width+WW] # shape (C, HH, WW)

            # one of the local gradients at w * x (sliced)
            dwx_dw = sliced_x # shape (C, HH, WW)

            dw[filter] += (dwx * dwx_dw) # shape (C, HH, WW)

            # one of the local gradients at w * x (sliced)
            dwx_dsx = w[filter] # # shape (C, HH, WW)

            # gradient of sliced_x
            dsx = dwx * dwx_dsx # shape (C, HH, WW)

            # local gradient at slice
            # dsx_dpx = 1

            # gradient of padded_x
            # dpx = dsx * dsx_dpx
            dpx[data_point, : , height:height+HH, width:width+WW] += dsx # shape (C, HH, WW)

    # local gradient at pad
    # dpx_dx = 1

    # remove pads
    # dx = dpx * dpx_dx
    dx = dpx[:, :, pad:H+pad, pad:W+pad] # shape (N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    # poor readability, good luck
    N, C, H, W = x.shape

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    out_height = int((H - pool_height)/stride + 1)
    out_width = int((W - pool_width)/stride + 1)

    out = np.zeros((N, C, out_height, out_width))

    # time complexity: O(N * C * H * W / stride)
    for n in range(N):
      for c in range(C):
        for h in range(0, H-pool_height+1, stride):
          for w in range(0, W-pool_width+1, stride):
            out_height = int((h - pool_height)/stride + 1)
            out_width = int((w - pool_width)/stride + 1)

            out[n, c, out_height, out_width] = np.max(x[n, c, h:h+pool_height, w:w+pool_width])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives - shape (N, C, out_height, out_width)
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x - shape (N, C, H, W)
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache

    # poor readability, good luck
    N, C, H, W = x.shape

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    dx = np.zeros(x.shape)

    # time complexity: O(N * C * H * W / stride)
    for n in range(N):
      for c in range(C):
        for h in range(0, H-pool_height+1, stride):
          for w in range(0, W-pool_width+1, stride):
            out_height = int((h - pool_height)/stride + 1)
            out_width = int((w - pool_width)/stride + 1)

            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
            # https://github.com/haofeixu/stanford-cs231n-2018/blob/master/assignment2/cs231n/layers.py
            max_indices = np.unravel_index(np.argmax(x[n, c, h:h+pool_height, w:w+pool_width]), (pool_height, pool_width))
            
            dx[n, c, h:h+pool_height, w:w+pool_width][max_indices] = dout[n, c, out_height, out_width]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    # https://www.reddit.com/r/cs231n/comments/443y2g/hints_for_a2/
    # https://www.reddit.com/r/cs231n/comments/6agt4s/assignment_2_spatial_batchnorm/

    # from sources above:

    # This is a hint for spatial batch normalization: you will need to reshape numpy arrays. When you do so you need to be careful and think about the order that numpy iterates over elements when reshaping. Suppose that x has shape (A, B, C) and you want to "collapse" the first and third dimensions into a single dimension, resulting in an array of shape (A*C, B).

    # Calling y = x.reshape(A * B, C) will give an array of the right shape, but it will be wrong. This will put x[0, 0, 0] into y[0, 0], then x[0, 0, 1] into y[0, 1], etc until eventually x[0, 0, C - 1] will go to y[0, C - 1] (assuming C < B); then x[0, 1, 0] will go to y[0, C]. This probably isn't the behavior you wanted.

    # Due this order for moving elements in a reshape, the rule of thumb is that it is only safe to collapse adjacent dimension; reshaping (A, B, C) to (A*C, B) is unsafe since the collapsed dimensions are not adjacent. To get the correct result, you should first use the transpose method to permute the dimensions so that the dimensions you want to collapse are adjacent, and then use reshape to actually collapse them.

    # Therefore for the above example you should call y = x.transpose(0, 2, 1).reshape(A * C, B)

    # First of all, you are transposing to (C, N, H, W). Then you are reshaping to (N * H * W,C). This means you are calculating the batch norm with respect to every pixel in an image in the whole minibatch. You only distinguish between the channels. So, in the end, you "get three batch norms" for every channel.

    N, C, H, W = x.shape
    vanilla_bn_input = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    intermediate, cache = batchnorm_forward(vanilla_bn_input, gamma, beta, bn_param)
    out = intermediate.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # TODO: look at this magic again (and understand it)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    # refer to spatial_batchnorm_forward for explanation
    N, C, H, W = dout.shape

    alt_bn_dout = dout.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    intermediate_dx, dgamma, dbeta = batchnorm_backward_alt(alt_bn_dout, cache)
    dx = intermediate_dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
