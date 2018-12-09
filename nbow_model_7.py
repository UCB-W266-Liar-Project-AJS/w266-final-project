 
### Model_7: Same as Model_1 except don't sum the xs_'s. 
from __future__ import print_function
from __future__ import division

import tensorflow as tf


### Add a helper function for matrix multiplication with 3-dimensional input matrices (from A3)
def matmul3d(X, W):
    """Wrapper for tf.matmul to handle a 3D input tensor X.
    Will perform multiplication along the last dimension.
    Args:
      X: [m,n,k]
      W: [k,l]
    Returns:
      XW: [m,n,l]
    """
    Xr = tf.reshape(X, [-1, tf.shape(X)[2]])
    XWr = tf.matmul(Xr, W)
    newshape = [tf.shape(X)[0], tf.shape(X)[1], tf.shape(W)[1]]
    return tf.reshape(XWr, newshape)


"""
# From A1----------------------------------------------------------------------------------------------------
def affine_layer(hidden_dim, x):
    '''Create an affine transformation.
    An affine transformation from linear algebra is "xW + b".
    Note that we want to compute this affine function on each
    feature vector "x" in a batch of examples and return the corresponding
    transformed vectors, each of dimension "hidden_dim".
    We'll see another way of implementing this using more sophisticated APIs
    in Assignment 2.
    Args:
      x: an op representing the features/incoming layer.
         The tensor that this op provides is of shape [batch_size x #features].
         (recall batch_size is the # of examples we want to predict in parallel)
      hidden_dim: a scalar defining the dimension of each output vector.
    Returns: a tensorflow op, when evaluated returns a tensor of dimension
             [batch_size x hidden_dim].
    Hint: On scrap paper, drop a picture of the matrix math xW + b.
    Hint: When doing the previous, make sure you draw "x" as [batch size x features]
          and the shape of the desired output as [batch_size x hidden_dim].
    Hint: use tf.get_variable to create trainable variables.
    Hint: use xavier initialization to initialize "W"
    Hint: always initialize "b" as 0s.  It isn't a constant though!
          It needs to be a trainable variable!
    '''
    #pass

    # START YOUR CODE

    # Draw the sketch suggested in the hint above.
    # Include a photo of the sketch in your submission.
    # In your sketch, label all matrix/vector dimensions.

    # Create trainable variables "W" and "b"
    # Hint: use tf.get_variable, tf.zeros_initializer, and tf.contrib.layers.xavier_initializer

    # Return xW + b.

    W = tf.get_variable(name="W", shape=(x.shape[-1],hidden_dim), initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name="b", shape=(hidden_dim,), initializer=tf.zeros_initializer())

    return tf.matmul(x,W) + b

    # END YOUR CODE

    
def fully_connected_layers(hidden_dims, x):
    '''Construct fully connected layer(s).
    You want to construct:
    x ---> [ xW + b -> relu(.) ]* ---> output
    where the middle block is repeated 0 or more times, determined
    by the len(hidden_dims).
    Args:
      hidden_dims: A list of the width(s) of the hidden layer.
      x: a TensorFlow "op" that will evaluate to a tensor of dimension [batch_size x input_dim].
    To get the tests to pass, you must use tf.nn.relu(.) as your element-wise nonlinearity.
    Hint: see tf.variable_scope - you'll want to use this to make each layer
    unique.
    Hint: a fully connected layer is a nonlinearity of an affine of its input.
          your answer here only be a couple of lines long (mine is 4).
    Hint: use your affine_layer(.) function above to construct the affine part
          of this graph.
    Hint: if hidden_dims is empty, just return x.
    '''

    # START YOUR CODE
    #pass

    #tf.reset_default_graph()  #(DON'T WANT TO CLEAR THE GRAPH!)

    #print('hidden_dims', hidden_dims)
    #print('x.shape', x.shape)

    #if len(hidden_dims)==0:
    #    return x

    for i in range(len(hidden_dims)):
        print(i, hidden_dims[i])
        with tf.variable_scope("FCL"+str(i)):
            #z = affine_layer(hidden_dims[i], x)
            #h = tf.nn.relu(z)
            x = tf.nn.relu(affine_layer(hidden_dims[i], x))
            #print("h.shape", h.shape)

    return x


    # END YOUR CODE

# From A1----------------------------------------------------------------------------------------------------
"""






def embedding_layer(ids_, V, embed_dim, init_scale=0.001):
    """Construct an embedding layer.

    Define a variable for the embedding matrix, and initialize it
    using tf.random_uniform_initializer to values in [-init_scale, init_scale].

    Args:
        ids_: [batch_size, max_len] Tensor of int32, integer ids
        V: (int) vocabulary size
        embed_dim: (int) embedding dimension
        init_scale: (float) scale to initialize embeddings

    Returns:
        xs_: [batch_size, max_len, embed_dim] Tensor of float32, embeddings for
            each element in ids_
    """
    
    W_embed_ = tf.get_variable("W_embed", shape=[V, embed_dim], initializer=tf.random_uniform_initializer(-init_scale, init_scale, dtype=tf.float32))
    xs_ = tf.nn.embedding_lookup(W_embed_, ids_)

    return xs_

def fully_connected_layers(h0_, hidden_dims, activation=tf.tanh,
                           dropout_rate=0, is_training=None):
    """Construct a stack of fully-connected layers.

    Args:
        h0_: [batch_size, d] Tensor of float32, the input activations
        hidden_dims: list(int) dimensions of the output of each layer
        activation: TensorFlow function, such as tf.tanh. Passed to
            tf.layers.dense.
        dropout_rate: if > 0, will apply dropout to activations.
        is_training: (bool) if true, is in training mode

    Returns:
        h_: [batch_size, hidden_dims[-1]] Tensor of float32, the activations of
            the last layer constructed by this function.
    """
    h_ = h0_
    for i, hdim in enumerate(hidden_dims):
        h_ = tf.layers.dense(h_, hdim, activation=activation, name=("Hidden_%d"%i))  #### NEED TO USE matmul3d NOT layers.dense?

        # Add dropout after each hidden layer (1-2 lines of code).
        if dropout_rate > 0:
            h_ = tf.layers.dropout(h_, rate=dropout_rate, training=is_training)  
    return h_

def softmax_output_layer(h_, labels_, num_classes):
    """Construct a softmax output layer.

    Implements:
        logits = h W + b
        loss = cross_entropy(softmax(logits), labels)

    Define variables for the weight matrix W_out and bias term
    b_out. Initialize the weight matrix with random normal noise (use
    tf.random_normal_initializer()), and the bias term with zeros (use
    tf.zeros_initializer()).

    For the cross-entropy loss, use tf.nn.sparse_softmax_cross_entropy_with_logits. 
    This produces output of shape [batch_size], the loss for each example. Use
    tf.reduce_mean to reduce this to a scalar.

    Args:
        h_: [batch_size, d] Tensor of float32, the input activations from a
            previous layer
        labels_: [batch_size] Tensor of int32, the target label ids
        num_classes: (int) the number of output classes

    Returns: (loss_, logits_)
        loss_: scalar Tensor of float32, the cross-entropy loss
        logits_: [batch_size, num_classes] Tensor of float32, the logits (hW + b)
    """
    with tf.variable_scope("Logits"):
        logits_ = None  # replace with (h W + b)
        # Please name your variables 'W_out' and 'b_out', as in:
        #   W_out_ = tf.get_variable("W_out", ...)
        
        W_out_ = tf.get_variable(name="W_out", shape=(h_.shape[-1],num_classes), initializer=tf.random_normal_initializer())
        b_out_ = tf.get_variable(name="b_out", shape=(num_classes,), initializer=tf.zeros_initializer())
        logits_ = tf.matmul(h_, W_out_) + b_out_

    # If no labels provided, don't try to compute loss.
    if labels_ is None:
        return None, logits_

    with tf.name_scope("Softmax"):       
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_, logits=logits_)
        loss_ = tf.reduce_mean(loss_)
        
    return loss_, logits_

def BOW_encoder(ids_, ns_, V, embed_dim, hidden_dims, dropout_rate=0,
                is_training=None,
                **unused_kw):
    """Construct a bag-of-words encoder.

    Steps:
        - Build the embeddings (using embedding_layer(...))
        - Apply the mask to zero-out padding indices, and sum the embeddings
            for each example
        - Build a stack of hidden layers (using fully_connected_layers(...))

    Note that this function returns the final encoding h_ as well as the masked
    embeddings xs_. The latter is used for L2 regularization, so that we can
    penalize the norm of only those vectors that were actually used for each
    example.

    Args:
        ids_: [batch_size, max_len] Tensor of int32, integer ids
        ns_:  [batch_size] Tensor of int32, (clipped) length of each sequence
        V: (int) vocabulary size
        embed_dim: (int) embedding dimension
        hidden_dims: list(int) dimensions of the output of each layer
        dropout_rate: (float) rate to use for dropout
        is_training: (bool) if true, is in training mode

    Returns: (h_, xs_)
        h_: [batch_size, hidden_dims[-1]] Tensor of float32, the activations of
            the last layer constructed by this function.
        xs_: [batch_size, max_len, embed_dim] Tensor of float32, the per-word
            embeddings as returned by embedding_layer and with the mask applied
            to zero-out the pad indices.
    """
    assert is_training is not None, "is_training must be explicitly set to True or False"
    # Embedding layer should produce:
    #   xs_: [batch_size, max_len, embed_dim]
    
    with tf.variable_scope("Embedding_Layer"):     
        xs_ = embedding_layer(ids_, V, embed_dim)

    # Mask off the padding indices with zeros
    #   mask_: [batch_size, max_len, 1] with values of 0.0 or 1.0
    mask_ = tf.expand_dims(tf.sequence_mask(ns_, xs_.shape[1],
                                            dtype=tf.float32), -1)
    # Multiply xs_ by the mask to zero-out pad indices. 
    xs_ = tf.multiply(xs_, mask_)
    print('xs_.shape:', xs_.shape)
    
    
    """
    # From A3----------------------------------------------------------------------------------------------------
    ## consult rnnlm and nplm.ipynb::
    self.W_out_ = tf.get_variable("W_out", shape=[self.H, self.V], 
                               initializer=tf.random_uniform_initializer(-1.0, 1.0, dtype=tf.float32))
    self.b_out_ = tf.get_variable(name="b", shape=[self.V,], initializer=tf.zeros_initializer())  # FROM A1

    #print('shape o_:', self.o_.shape)
    ### 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size] AKA [batch_size, max_len, embed_dim]

    #print('shape W_out_:', self.W_out_.shape)
    #print('shape b_out_:', self.b_out_.shape)
    self.logits_ = tf.add(matmul3d(self.o_, self.W_out_), self.b_out_, name="logits")
    # Output logits, which can be used by loss functions or for prediction.
    # Overwrite this with an actual Tensor of shape  [batch_size, max_time, V].


    print('shape o_:', self.o_.shape)  # shape o_: (?, ?, 200)
    #print('shape W_out_:', self.W_out_.get_shape)
    print('shape W_out_:', self.W_out_.shape)  # shape W_out_: (200, 10000)
    print('shape b_out_:', self.b_out_.shape)  # shape b_out_: (10000,)
    print('shape target_y_:', self.target_y_.shape)  # shape target_y_: (?, ?)
    # [replace labels=tf.expand_dims(self.target_y_, 1) ]

    o_r_ = tf.reshape(self.o_, [self.batch_size_*self.max_time_, -1])
    target_y_r_ = tf.reshape(self.target_y_, [self.batch_size_*self.max_time_, -1])
    # From A3----------------------------------------------------------------------------------------------------
    """
    


    """
    # Sum embeddings: [batch_size, max_len, embed_dim] -> [batch_size, embed_dim]
    x_ = tf.reduce_sum(xs_, axis=1)
    print('x_.shape:', x_.shape)
    """
    
    ### SIMPLEST THING TO TRY:
    #x_ = tf.reshape(xs_, [tf.shape(xs_)[0]*tf.shape(xs_)[1], -1])   # [batch_size, max_len, -1]
    #x_ = tf.reshape(xs_, [-1, tf.shape(xs_)[1]*tf.shape(xs_)[2]])   # [batch_size, max_len*embed_dim]
    x_ = tf.reshape(xs_, [-1, 40*50])   # [batch_size, max_len*embed_dim]
    print('x_.shape:', x_.shape)
    ###
    
    
    
    
    

    # Build a stack of fully-connected layers
    h_ = fully_connected_layers(x_, hidden_dims, dropout_rate=0.5, is_training=True)
    print('h_.shape:', h_.shape)

    return h_, xs_

def classifier_model_fn(features, labels, mode, params):
    # Seed the RNG for repeatability
    tf.set_random_seed(params.get('rseed', 10))

    # Check if this graph is going to be used for training.
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    if params['encoder_type'] == 'bow':
        with tf.variable_scope("Encoder"):
            h_, xs_ = BOW_encoder(features['ids'], features['ns'],
                                  is_training=is_training,
                                  **params)
    else:
        raise ValueError("Error: unsupported encoder type "
                         "'{:s}'".format(params['encoder_type']))

    # Construct softmax layer and loss function
    
    with tf.variable_scope("Output_Layer"):
        ce_loss_, logits_ = softmax_output_layer(h_, labels, params['num_classes'])   # use "labels_2" . ???

    with tf.name_scope("Prediction"):
        pred_proba_ = tf.nn.softmax(logits_, name="pred_proba")
        pred_max_ = tf.argmax(logits_, 1, name="pred_max")
        predictions_dict = {"proba": pred_proba_, "max": pred_max_}

    if mode == tf.estimator.ModeKeys.PREDICT:
        # If predict mode, don't bother computing loss.
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions_dict)

    # L2 regularization (weight decay) on parameters, from all layers
    with tf.variable_scope("Regularization"):
        l2_penalty_ = tf.nn.l2_loss(xs_)  # l2 loss on embeddings
        for var_ in tf.trainable_variables():
            if "Embedding_Layer" in var_.name:
                continue
            l2_penalty_ += tf.nn.l2_loss(var_)
        l2_penalty_ *= params['beta']  # scale by regularization strength
        tf.summary.scalar("l2_penalty", l2_penalty_)
        regularized_loss_ = ce_loss_ + l2_penalty_

    with tf.variable_scope("Training"):
        if params['optimizer'] == 'adagrad':
            optimizer_ = tf.train.AdagradOptimizer(params['lr'])
        elif params['optimizer'] == 'adam':
            optimizer_ = tf.train.AdamOptimizer(params['lr'])
        else:
            optimizer_ = tf.train.GradientDescentOptimizer(params['lr'])
        train_op_ = optimizer_.minimize(regularized_loss_,
                                        global_step=tf.train.get_global_step())

    tf.summary.scalar("cross_entropy_loss", ce_loss_)
    eval_metrics = {"cross_entropy_loss": tf.metrics.mean(ce_loss_),
                    "accuracy": tf.metrics.accuracy(labels, pred_max_)}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions_dict,
                                      loss=regularized_loss_,
                                      train_op=train_op_,
                                      eval_metric_ops=eval_metrics)
