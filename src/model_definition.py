from hyper_param import *
import keras
from keras import layers
from keras.models import Sequential
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Input
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


def build_context_to_query(context_layer, query_layer, n_heads=2, head_dim=4):
    attention_layer_c2q = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=head_dim)
    return attention_layer_c2q(context_layer, query_layer)  # output shape (context_lenght, Emb_dim)


def build_query_to_context(context_layer, query_layer, context_lenght, n_heads=2, head_dim=4):
    attention_layer_q2c = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=head_dim)
    attention_tensor = attention_layer_q2c(query_layer, context_layer)  # output shape (query_lenght, Emb_dim)
    max_column = tf.nn.softmax(tf.math.reduce_max(attention_tensor, axis=-2))  # compute the most important words in the context wrt the query
    q2c_tensor = tf.repeat(tf.expand_dims(max_column, -2), context_lenght, axis=-2)    # duplicate the column for every context word
    return q2c_tensor


def build_overall_attention_tensor(context_tensor, context_to_query_tensor, query_to_context_tensor):
    context_c2q = tf.math.multiply(context_tensor, context_to_query_tensor)
    context_q2c = tf.math.multiply(context_tensor, query_to_context_tensor)
    return tf.concat(values=[context_tensor, context_to_query_tensor, context_c2q, context_q2c], axis=-1)


def model_definition(context_max_lenght, query_max_lenght, tokenizer_x):

    # initialize two distinct models
    context_input = Input(shape=(context_max_lenght,))
    query_input = Input(shape=(query_max_lenght,))

    # adding the Embedding (words) layer to both the models
    context_embedding = word_embbedding_layer(context_max_lenght, tokenizer_x)(context_input)
    query_embedding = word_embbedding_layer(query_max_lenght, tokenizer_x)(query_input)

    # Contextual Embedding Layer
    context_contestual_embedding = Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True))(context_embedding)
    query_contestual_embedding = Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True))(query_embedding)

    # Attention Flow Layer

    context_to_query_tensor = build_context_to_query(context_contestual_embedding, query_contestual_embedding)
    query_to_context_tensor = build_query_to_context(query_contestual_embedding, context_contestual_embedding, context_max_lenght)
    overall_attention_tensor = build_overall_attention_tensor(context_contestual_embedding, context_to_query_tensor, query_to_context_tensor)

    # Modeling Layer

    query_context_contextual_tensor = Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True))(overall_attention_tensor)
    query_context_contextual_tensor = Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True))(query_context_contextual_tensor)

    # END
    place_holder = Dense(1)(query_context_contextual_tensor)  # FIXME
    #  print
    model = keras.Model(inputs=[context_input, query_input], outputs=place_holder)
    model.compile()
    print(model.summary())
