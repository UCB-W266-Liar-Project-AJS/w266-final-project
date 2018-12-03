## nbow_model_2 is same as nbow_model_1 except it uses GloVe word embeddings rather than initializing with random numbers.  

from __future__ import print_function
from __future__ import division

# NumPy and TensorFlow
import numpy as np     ### ADD THIS
import pandas as pd    ### ADD THIS
import tensorflow as tf

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
    #####
    
    #Edited these lines relative to nbow_model_1:    
    #W_embed_ = tf.get_variable("W_embed", shape=[V, embed_dim], initializer=tf.random_uniform_initializer(-init_scale, init_scale, dtype=tf.float32))
    
    # For this model, make the GloVe embeddings constant.
    
    ## References:
    # 1. https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
    # 2. https://github.com/monk1337/word_embedding-in-tensorflow/blob/master/Use%20Pre-trained%20word_embedding%20in%20Tensorflow.ipynb
    # 3. http://ai.intelligentonlinetools.com/ml/convert-word-to-vector-glove-python/
    
    ## Example from 1. (assumes existing embedding file called "word_embedding" and not trainable):
    # Word_embedding = tf.get_variable(name="W", shape=[400000,100], initializer=tf.constant_initializer(np.array(word_embedding)), trainable=False)
    # embedding_lookup= tf.nn.embedding_lookup(Word_embedding,input_x)
    
    # Based on above examples, set this up to use the 50-dim GloVe embeddings based on a vocab size of V words
    #  that have been prioritized based on usage in the ISOT data.
    
    
    #W_embed_ = tf.get_variable(name="W_embed", shape=[V, embed_dim], initializer = tf.constant_initializer(np.array(word_embedding)), trainable=False)   # shape=[V, embed_dim]
       
    isot_glove50 = pd.read_pickle('parsed_data/isot_vocab_glove50embed.pkl')
    
    ### NOTE: Set trainable=False to keep GloVe embeddings constant; set trainable=true to allow embed weights to train
    isot_glove50_embed_ = tf.get_variable(name="isot_glove50_embed", shape=[V, embed_dim], initializer = tf.constant_initializer(np.array(isot_glove50.values)), trainable=True)   # shape=[V, embed_dim]
 
    xs_ = tf.nn.embedding_lookup(isot_glove50_embed_, ids_)

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
        h_ = tf.layers.dense(h_, hdim, activation=activation, name=("Hidden_%d"%i))

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

    # Sum embeddings: [batch_size, max_len, embed_dim] -> [batch_size, embed_dim]
    x_ = tf.reduce_sum(xs_, axis=1)

    # Build a stack of fully-connected layers
    h_ = fully_connected_layers(x_, hidden_dims, dropout_rate=0.5, is_training=True)

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

    # Construct softmax layer and loss functions
    with tf.variable_scope("Output_Layer"):
        ce_loss_, logits_ = softmax_output_layer(h_, labels, params['num_classes'])

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
