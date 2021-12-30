from hyper_param import *
import keras
import tensorflow as tf
from keras.layers import Dense, Input, LSTM, Bidirectional, Dropout, GRU, Dot, Concatenate
from keras.models import Model


"""
Hyper parameter for the model
"""
DROP_RATE = 0.2
L1 = 0.00005
L2 = 0.0001


def word_embbedding_layer(max_seq_length, tokenizer):
    embedding_layer = keras.layers.Embedding(input_dim=tokenizer.get_info()['vocab_size'],  # num of words in vocabulary
                                             output_dim=EMBEDDING_DIM,
                                             input_length=max_seq_length,
                                             weights=[tokenizer.embedding_matrix],
                                             trainable=False,
                                             mask_zero=True)
    return embedding_layer


def build_context_to_query(context_layer, query_layer, n_heads=16, head_dim=8):
    attention_layer_c2q = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=head_dim)
    return attention_layer_c2q(context_layer, query_layer)  # output shape (context_lenght, Emb_dim)


def build_query_to_context(context_layer, query_layer, context_lenght, n_heads=16, head_dim=8):
    attention_layer_q2c = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=head_dim)
    attention_tensor = attention_layer_q2c(query_layer, context_layer)  # output shape (query_lenght, Emb_dim)
    max_column = tf.nn.softmax(tf.math.reduce_max(attention_tensor, axis=-2))  # compute the most important words in the context wrt the query -> shape (1, Emb_dim)
    q2c_tensor = tf.repeat(tf.expand_dims(max_column, -2), context_lenght, axis=-2)    # duplicate the column for every context word -> shape (context_lenght, Emb_dim)
    return q2c_tensor


def build_overall_attention_tensor(context_to_query_tensor, query_to_context_tensor):
    return tf.concat(values=[context_to_query_tensor, query_to_context_tensor], axis=-1)


def model_definition(context_max_lenght, query_max_lenght, tokenizer_x):

    # initialize two distinct models
    context_input = Input(shape=(context_max_lenght,))
    query_input = Input(shape=(query_max_lenght,))

    # adding the Embedding (words) layer to both the models
    context_embedding = word_embbedding_layer(context_max_lenght, tokenizer_x)(context_input)
    query_embedding = word_embbedding_layer(query_max_lenght, tokenizer_x)(query_input)

    # Contextual Embedding Layer
    context_contestual_embedding = Dropout(DROP_RATE)(Bidirectional(GRU(EMBEDDING_DIM, return_sequences=False))(context_embedding))
    query_contestual_embedding = Dropout(DROP_RATE)(Bidirectional(GRU(EMBEDDING_DIM, return_sequences=False))(query_embedding))

    # Attention Flow Layer

    cosine_similarity_layer = Dot(axes=1)([context_contestual_embedding, query_contestual_embedding])
    merging_layer = Dropout(DROP_RATE)(Concatenate()([context_contestual_embedding, query_contestual_embedding, cosine_similarity_layer]))


    # Modeling Layer


    classifier_layer = Dropout(DROP_RATE)(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2))(merging_layer))
    classifier_layer = Dropout(DROP_RATE)(Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1, l2=L2))(classifier_layer))

    # Output Layer

    # The Dense layer behaviour, in the following configuration, is the same of a vector by matrix product. (trainable)

    output_classifier = Dense(1, activation='sigmoid')(classifier_layer)

    #  Create the model
    model = Model(inputs=[context_input, query_input], outputs=[output_classifier])

    #  Compile the model with custom compiling settings
    model.compile(optimizer=tf.keras.optimizers.Nadam(), loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.FalseNegatives(name="fls_neg")])
    model.summary()

    return model
