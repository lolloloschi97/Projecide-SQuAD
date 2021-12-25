from hyper_param import *
import keras
from keras import layers
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense, Multiply
from keras.layers import LSTM, Bidirectional
from keras.models import Model


def word_embbedding_layer(max_seq_length, tokenizer):
    embedding_layer = keras.layers.Embedding(input_dim=tokenizer.get_info()['vocab_size'],  # num of words in vocabulary
                                             output_dim=EMBEDDING_DIM,
                                             input_length=max_seq_length,
                                             weights=[tokenizer.embedding_matrix],
                                             trainable=False,
                                             mask_zero=True)
    return embedding_layer


########  ATTENTION ########


def pairwise_cosine_sim(params):
    """
    TO BE FIXED

    A [batch x n x d] tensor of n rows with d dimensions
    B [batch x m x d] tensor of m rows with d dimensions
    returns:
    D [n x m] tensor of similarity matrix obtained using the trainable weights
    """

    C, Q, weights = params
    matrix = []
    for t in range(C.shape[1]):
        row = []
        for j in range(Q.shape[1]):
            h_u_hdu = tf.concat([C[:, t, :], Q[:, j, :], tf.math.multiply(C[:, t, :], Q[:, j, :])], axis=1)
            s_tj = tf.reduce_sum(tf.math.multiply(weights, h_u_hdu))
            row.append(s_tj)
        matrix.append(tf.stack(row))
        print(t)

    similarity_matrix = tf.stack(matrix)
    print("Similarity matrix shape: " + str(similarity_matrix.shape))
    return similarity_matrix


def model_definition(context_max_lenght, query_max_lenght, tokenizer_x):
    # initialize two distinct models
    model_context = Sequential()
    model_query = Sequential()

    # adding the Embedding (words) layer to both the models
    model_context.add(word_embbedding_layer(context_max_lenght, tokenizer_x))
    model_query.add(word_embbedding_layer(query_max_lenght, tokenizer_x))

    # Contextual Embedding Layer
    model_context.add(Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True)))
    model_query.add(Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True)))

    # Attention Flow Layer
    weights = tf.Variable(initial_value=tf.random.normal((300,)), trainable=True)
    # weights = Dense(300, activation='linear', use_bias=False)
    similarity_matrix_layer = keras.layers.Lambda(pairwise_cosine_sim)((model_context.output, model_query.output, weights))

    place_holder = Dense(1)(similarity_matrix_layer)  # FIXME
    #  print
    model = keras.Model(inputs=[model_context.input, model_query.input], outputs=place_holder)
    model.compile()
    print(model.summary())
