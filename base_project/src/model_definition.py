from hyper_param import *
import keras
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from keras.layers import Dense, Input, LSTM, Bidirectional, Dropout, Concatenate
from keras.models import Model


"""
Hyper parameter for the model
"""
DROP_RATE = 0.28


def word_embbedding_layer(max_seq_length, tokenizer):
    embedding_layer = keras.layers.Embedding(input_dim=tokenizer.get_info()['vocab_size'],  # num of words in vocabulary
                                             output_dim=EMBEDDING_DIM,
                                             input_length=max_seq_length,
                                             weights=[tokenizer.embedding_matrix],
                                             trainable=False,
                                             mask_zero=True)
    return embedding_layer

def build_context_feature_vector(context_layer, context_pos, context_exact_match):
    context_feature_vector = Concatenate()([context_layer, context_pos, tf.expand_dims(context_exact_match, axis=-1)])
    return context_feature_vector

def build_context_to_query(context_layer, query_layer, n_heads=4, head_dim=8):
    attention_layer_c2q = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=head_dim)
    return attention_layer_c2q(context_layer, query_layer)  # output shape (context_lenght, Emb_dim)


def build_query_to_context(context_layer, query_layer, context_lenght, n_heads=2, head_dim=4):
    attention_layer_q2c = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=head_dim)
    attention_tensor = attention_layer_q2c(query_layer, context_layer)  # output shape (query_lenght, Emb_dim)
    max_column = tf.nn.softmax(tf.math.reduce_max(attention_tensor, axis=-2))  # compute the most important words in the context wrt the query -> shape (1, Emb_dim)
    q2c_tensor = tf.repeat(tf.expand_dims(max_column, -2), context_lenght, axis=-2)    # duplicate the column for every context word -> shape (context_lenght, Emb_dim)
    return q2c_tensor

def build_overall_attention_tensor(context_tensor, context_to_query_tensor, query_to_context_tensor):
    context_c2q = tf.math.multiply(context_tensor, context_to_query_tensor)
    context_q2c = tf.math.multiply(context_tensor, query_to_context_tensor)
    return tf.concat(values=[context_tensor, context_to_query_tensor, context_c2q, context_q2c], axis=-1)

def attention_layer_simil_Bidaf(context_layer, query_layer, n_heads=1, head_dim=4):
    attention_layer_c2q = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=head_dim)
    q2c_tensor, attention_scores = attention_layer_c2q(context_layer, query_layer, return_attention_scores = True)  # output shape (context_lenght, Emb_dim)
    attention_tensor = tf.squeeze(tf.nn.softmax(tf.math.reduce_max(attention_scores, axis=-1)), 1)
    c2q_tensor = tf.linalg.matmul(tf.linalg.diag(attention_tensor), context_layer)
    context_c2q = tf.math.multiply(context_layer, c2q_tensor)
    context_q2c = tf.math.multiply(context_layer, q2c_tensor)
    return tf.concat(values=[context_layer, c2q_tensor, context_c2q, context_q2c], axis=-1)

def attention_layer(context_layer, query_layer, n_heads=4, head_dim=4):
    # Context to query
    attention_layer_c2q = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=head_dim)
    c2q_tensor, attention_scores = attention_layer_c2q(context_layer, query_layer, return_attention_scores=True)  # output shape (context_lenght, Emb_dim), attention shape (context_lenght, query_lenght)

    return tf.concat(values=[context_layer, c2q_tensor], axis=-1)


def weighted_cross_entropy_with_logits_modified(labels, logits, pos_weight, neg_weights, name=None):
    log_weight = neg_weights + (pos_weight - neg_weights) * labels
    return math_ops.add(
        (1 - labels) * logits * neg_weights,
        log_weight * (math_ops.log1p(math_ops.exp(-math_ops.abs(logits))) +
                      nn_ops.relu(-logits)),  # pylint: disable=invalid-unary-operand-type
        name=name)


def custom_loss_fn(y_true, y_pred):
    pos_weight = tf.constant(1.0)
    neg_weight = tf.constant(0.0)
    bn_crossentropy = weighted_cross_entropy_with_logits_modified(y_true, y_pred, pos_weight, neg_weight)
    return tf.reduce_mean(bn_crossentropy, axis=-1)


def model_definition(context_max_lenght, query_max_lenght, tokenizer_x, pos_max_lenght):

    # initialize two distinct models
    context_input = Input(shape=(context_max_lenght,))
    context_pos = Input(shape=(context_max_lenght, pos_max_lenght))
    context_exact_match = Input(shape=(context_max_lenght,))
    query_input = Input(shape=(query_max_lenght,))

    # adding the Embedding (words) layer to both the models
    context_embedding = word_embbedding_layer(context_max_lenght, tokenizer_x)(context_input)
    query_embedding = word_embbedding_layer(query_max_lenght, tokenizer_x)(query_input)

    # generating context features vectors
    context_feature_vector = build_context_feature_vector(context_embedding, context_pos, context_exact_match)

    # Contextual Embedding Layer
    context_contestual_embedding_compressed = Dropout(DROP_RATE)(Dense(EMBEDDING_DIM, use_bias=True, activation='relu')(context_feature_vector))
    context_contestual_embedding = Dropout(DROP_RATE)(Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True))(context_contestual_embedding_compressed))
    query_contestual_embedding = Dropout(DROP_RATE)(Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True))(query_embedding))


    # Attention Flow Layer
    #context_to_query_tensor = build_context_to_query(context_contestual_embedding_compressed, query_contestual_embedding)
    #query_to_context_tensor = build_query_to_context(query_contestual_embedding, context_contestual_embedding_compressed, context_max_lenght)
    #overall_attention_tensor = build_overall_attention_tensor(context_contestual_embedding_compressed, context_to_query_tensor, query_to_context_tensor)
    overall_attention_tensor = attention_layer(context_contestual_embedding, query_contestual_embedding, n_heads=8, head_dim=8)

    # Modeling Layer

    query_context_contextual_tensor_1 = Dropout(DROP_RATE)(Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True))(overall_attention_tensor))
    query_context_contextual_tensor_2 = Dropout(DROP_RATE)(Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True))(query_context_contextual_tensor_1))

    # Output Layer

    # The Dense layer behaviour, in the following configuration, is the same of a vector by matrix product. (trainable)
    dense_start_layer = Dense(1, use_bias=False, activation='linear')(tf.concat([overall_attention_tensor, query_context_contextual_tensor_1], axis=-1))
    dense_end_layer = Dense(1, use_bias=False, activation='linear')(tf.concat([overall_attention_tensor, query_context_contextual_tensor_2], axis=-1))

    dense_start_layer = Dropout(DROP_RATE)(dense_start_layer)
    dense_end_layer = Dropout(DROP_RATE)(dense_end_layer)
    prob_start_index = tf.nn.softmax(tf.squeeze(dense_start_layer, -1))
    prob_end_index = tf.nn.softmax(tf.squeeze(dense_end_layer, -1))

    #  Create the model
    model = Model(inputs=[context_input, context_pos, context_exact_match, query_input], outputs=[prob_start_index, prob_end_index])

    #  Compile the model with custom compiling settings
    # TODO remove binary cross
    model.compile(optimizer=tf.keras.optimizers.Nadam(), loss=custom_loss_fn,
                  metrics=[tf.keras.metrics.BinaryCrossentropy(name="bn_cross"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.Precision(name="precision")])
    model.summary()

    return model
