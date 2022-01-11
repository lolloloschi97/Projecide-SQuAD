from hyper_param import *
import keras
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from keras.layers import Dense, Input, LSTM, Bidirectional, Dropout, Concatenate, Conv1D, Conv2D, DepthwiseConv2D
from keras.models import Model

"""
!!!

THIS IS THE BASELINE MODEL, NOT LINKED TO THE PROJECT ANYMORE. We keep that to comparison purposes.

!!!
"""




"""
Hyper parameter for the model
"""
DROP_RATE = 0.3


def word_embbedding_layer(max_seq_length, tokenizer):
    embedding_layer = keras.layers.Embedding(input_dim=tokenizer.get_info()['vocab_size'],  # num of words in vocabulary
                                             output_dim=EMBEDDING_DIM,
                                             input_length=max_seq_length,
                                             weights=[tokenizer.embedding_matrix],
                                             trainable=False,
                                             mask_zero=True)
    return embedding_layer


def build_context_feature_vector(context_layer, query_layer, context_pos, context_exact_match, n_heads=8, head_dim=16):
    attention_layer_c2q = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=head_dim)
    attention_tensor = attention_layer_c2q(context_layer, query_layer)
    context_feature_vector = Concatenate()([context_layer, attention_tensor, context_pos, tf.expand_dims(context_exact_match, axis=-1)])
    return context_feature_vector


def question_encoding_layer(query_layer,n_heads=8, head_dim=16):
    b = Conv1D(EMBEDDING_DIM,1, activation ='relu')(query_layer)
    b = Conv1D(EMBEDDING_DIM/2, 1, activation='relu')(b)
    b = Conv1D(1,1,activation= 'softmax')(b)
    print(b.shape)
    attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=head_dim)
    attention_tensor = attention_layer(b,query_layer)
    question_encoding_tensor = tf.linalg.matmul(attention_tensor, query_layer, transpose_a = True)
    print(question_encoding_tensor.shape)
    return question_encoding_tensor


def final_prediction(query_encoding, context_layer):
    c_out = query_encoding.shape[-1]
    print(c_out)
    w = Conv1D(c_out,1, activation = 'relu')(query_encoding)
    print(w)
    pred = tf.linalg.matmul(context_layer,w, transpose_b=True)
    print(pred.shape)
    #pred_index = tf.nn.softmax(tf.squeeze(tf.nn.relu(pred), -1))
    pred_index = tf.nn.softmax(tf.nn.elu(tf.squeeze(pred, -1)))
    print(pred_index.shape)
    return pred_index


def weighted_cross_entropy_with_logits_modified(labels, logits, pos_weight, neg_weights, name = None):
    log_weight = neg_weights + (pos_weight - neg_weights) * labels
    return math_ops.add(
        (1 - labels) * logits * neg_weights,
        log_weight * (math_ops.log1p(math_ops.exp(-math_ops.abs(logits))) +
                      nn_ops.relu(-logits)),  # pylint: disable=invalid-unary-operand-type
        name=name)


def custom_loss_fn(y_true, y_pred):
    pos_weight = tf.constant(100.0)
    neg_weight =  tf.constant(0.0)
    bn_crossentropy = weighted_cross_entropy_with_logits_modified(y_true, y_pred, pos_weight,neg_weight)
    return tf.reduce_mean(bn_crossentropy, axis=-1)


def model_definition_Dr_qa(context_max_lenght, query_max_lenght, tokenizer_x, pos_max_lenght):

    # initialize two distinct models
    context_input = Input(shape=(context_max_lenght,))
    context_pos = Input(shape=(context_max_lenght, pos_max_lenght))
    context_exact_match = (Input(shape=(context_max_lenght,)))
    query_input = Input(shape=(query_max_lenght,))


    # adding the Embedding (words) layer to both the models
    context_embedding = Dropout(DROP_RATE)(word_embbedding_layer(context_max_lenght, tokenizer_x)(context_input))
    query_embedding = Dropout(DROP_RATE)(word_embbedding_layer(query_max_lenght, tokenizer_x)(query_input))

    # generating context features vectors
    context_feature_vector = build_context_feature_vector(context_embedding,query_embedding, context_pos, context_exact_match)

    # Contextual Embedding Layer
    context_contestual_embedding_1 = Dropout(DROP_RATE)(Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True))(context_feature_vector))
    #context_contestual_embedding_2 = Dropout(DROP_RATE)(Bidirectional(LSTM(128, return_sequences=True))(context_contestual_embedding_1))
    #context_contestual_embedding_3 = Dropout(DROP_RATE)(Bidirectional(LSTM(128, return_sequences=True))(context_contestual_embedding_2))

    query_contestual_embedding_1 = Dropout(DROP_RATE)(Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True))(query_embedding))
    #query_contestual_embedding_2 = Dropout(DROP_RATE)(Bidirectional(LSTM(128, return_sequences=True))(query_contestual_embedding_1))
    #query_contestual_embedding_3 = Dropout(DROP_RATE)(Bidirectional(LSTM(128, return_sequences=True))(query_contestual_embedding_2))


    # Query embedding
    query_encoding = question_encoding_layer(query_contestual_embedding_1)

    # Output Layer
    prob_start_index = final_prediction(query_encoding,context_contestual_embedding_1)
    prob_end_index = final_prediction(query_encoding, context_contestual_embedding_1)


    #  Create the model
    model = Model(inputs=[context_input,context_pos, context_exact_match, query_input], outputs=[prob_start_index, prob_end_index])

    #  Compile the model with custom compiling settings
    model.compile( optimizer=tf.keras.optimizers.Nadam(), loss=custom_loss_fn,
                  metrics=[tf.keras.metrics.BinaryCrossentropy(name="bn_cross"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.Precision(name="precision")])
    model.summary()

    return model
