import gensim
import gensim.downloader as gloader
import numpy as np
import tensorflow as tf
from typing import List, Callable, Dict
import tqdm
from hyper_param import *
import collections


def extract_numpy_structures(train_set, val_set, test_set):
    # train
    x_train_question = train_set['question'].values
    x_train_context = train_set['context'].values
    x_train_match = train_set['exact_match'].values
    x_train_pos = train_set['pos'].values
    y_train_answer_start = train_set['start_index'].values
    y_train_answer_end = train_set['end_index'].values
    # val
    x_val_question = val_set['question'].values
    x_val_context = val_set['context'].values
    x_val_match = val_set['exact_match'].values
    x_val_pos = val_set['pos'].values
    y_val_answer_start = val_set['start_index'].values
    y_val_answer_end = val_set['end_index'].values
    # test
    x_test_question = test_set['question'].values
    x_test_context = test_set['context'].values
    x_test_match = test_set['exact_match'].values
    x_test_pos = test_set['pos'].values
    y_test_answer_start = test_set['start_index'].values
    y_test_answer_end = test_set['end_index'].values

    return x_train_question, x_train_context, x_train_match, x_train_pos, y_train_answer_start, y_train_answer_end, \
           x_val_question, x_val_context, x_val_match, x_val_pos, y_val_answer_start, y_val_answer_end, \
           x_test_question, x_test_context, x_test_match, x_test_pos, y_test_answer_start, y_test_answer_end


def load_embedding_model(model_type: str,
                         embedding_dimension: int = 200) -> gensim.models.keyedvectors.KeyedVectors:
    """
    Loads a pre-trained word embedding model via gensim library.

    :param model_type: name of the word embedding model to load.
    :param embedding_dimension: size of the embedding space to consider

    :return
        - pre-trained word embedding model (gensim KeyedVectors object)
    """

    # Find the correct embedding model name
    if model_type.strip().lower() == 'word2vec':
        download_path = "word2vec-google-news-300"
    elif model_type.strip().lower() == 'glove':
        download_path = "glove-wiki-gigaword-{}".format(embedding_dimension)
    elif model_type.strip().lower() == 'fasttext':
        download_path = "fasttext-wiki-news-subwords-300"
    else:
        raise AttributeError("Unsupported embedding model type! Available ones: word2vec, glove, fasttext")

    # Check download
    try:
        emb_model = gloader.load(download_path)
    except ValueError as e:
        print("Invalid embedding model name! Check the embedding dimension:")
        print("Word2Vec: 300")
        print("Glove: 50, 100, 200, 300")
        raise e

    return emb_model


def check_OOV_terms(embedding_model, word_listing):
    """
    Checks differences between pre-trained embedding model vocabulary
    and dataset specific vocabulary in order to highlight out-of-vocabulary terms.

    :param embedding_model: pre-trained word embedding model (gensim wrapper)
    :param word_listing: dataset specific vocabulary (list)

    :return
        - list of OOV terms
    """
    embedding_vocabulary = set(embedding_model.key_to_index.keys())
    oov = set(word_listing).difference(embedding_vocabulary)
    return list(oov)


def build_embedding_matrix(embedding_model: gensim.models.keyedvectors.KeyedVectors,
                           embedding_dimension: int,
                           word_to_idx: Dict[str, int],
                           vocab_size: int,
                           oov_terms: List[str]) -> np.ndarray:
    """
    Builds the embedding matrix of a specific dataset given a pre-trained word embedding model

    :param embedding_model: pre-trained word embedding model (gensim wrapper)
    :param word_to_idx: vocabulary map (word -> index) (dict)
    :param vocab_size: size of the vocabulary
    :param oov_terms: list of OOV terms (list)

    :return
        - embedding matrix that assigns a high dimensional vector to each word in the dataset specific vocabulary (shape |V| x d)
    """

    embedding_matrix = np.zeros((vocab_size, embedding_dimension), dtype=np.float32)
    for word, idx in tqdm.tqdm(word_to_idx.items()):
        try:
            embedding_vector = embedding_model[word]
        except (KeyError, TypeError):
            embedding_vector = np.random.uniform(low=-0.05, high=0.05, size=embedding_dimension)

        embedding_matrix[idx] = embedding_vector

    return embedding_matrix


class KerasTokenizer(object):
    """
    A simple high-level wrapper for the Keras tokenizer.
    """

    def __init__(self, build_embedding_matrix=False, embedding_dimension=None,
                 embedding_model_type=None, tokenizer_args=None):
        if build_embedding_matrix:
            assert embedding_model_type is not None
            assert embedding_dimension is not None and type(embedding_dimension) == int

        self.build_embedding_matrix = build_embedding_matrix
        self.embedding_dimension = embedding_dimension
        self.embedding_model_type = embedding_model_type
        self.embedding_model = None
        self.embedding_matrix = None
        self.vocab = None

        tokenizer_args = {} if tokenizer_args is None else tokenizer_args
        assert isinstance(tokenizer_args, dict) or isinstance(tokenizer_args, collections.OrderedDict)

        self.tokenizer_args = tokenizer_args

    def build_vocab(self, data, **kwargs):
        print('Fitting tokenizer...')
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(**self.tokenizer_args)
        self.tokenizer.fit_on_texts(data)
        print('Fit completed!')

        self.vocab = self.tokenizer.word_index

        if self.build_embedding_matrix:
            print('Loading embedding model! It may take a while...')
            self.embedding_model = load_embedding_model(model_type=self.embedding_model_type,
                                                        embedding_dimension=self.embedding_dimension)

            print('Checking OOV terms...')
            self.oov_terms = check_OOV_terms(embedding_model=self.embedding_model,
                                             word_listing=list(self.vocab.keys()))

            print('Building the embedding matrix...')
            self.embedding_matrix = build_embedding_matrix(embedding_model=self.embedding_model,
                                                           word_to_idx=self.vocab,
                                                           vocab_size=len(self.vocab) + 1,
                                                           embedding_dimension=self.embedding_dimension,
                                                           oov_terms=self.oov_terms)
            print('Done!')

    def get_info(self):
        return {
            'build_embedding_matrix': self.build_embedding_matrix,
            'embedding_dimension': self.embedding_dimension,
            'embedding_model_type': self.embedding_model_type,
            'embedding_matrix': self.embedding_matrix.shape if self.embedding_matrix is not None else self.embedding_matrix,
            'embedding_model': self.embedding_model,
            'vocab_size': len(self.vocab) + 1,
        }

    def tokenize(self, text):
        return text

    def convert_tokens_to_ids(self, tokens):
        if type(tokens) == str:
            return self.tokenizer.texts_to_sequences([tokens])[0]
        else:
            return self.tokenizer.texts_to_sequences(tokens)

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.sequences_to_texts(ids)


def build_tokenizer(x_train_context):
    tokenizer_args = {
        'oov_token': 1,  # The vocabulary id for unknown terms during text conversion
    }
    tokenizer_X = KerasTokenizer(tokenizer_args=tokenizer_args,
                                 build_embedding_matrix=True,
                                 embedding_dimension=EMBEDDING_DIM,
                                 embedding_model_type="glove")
    tokenizer_X.build_vocab(x_train_context)
    tokenizer_X_info = tokenizer_X.get_info()
    print('Tokenizer X info: ', tokenizer_X_info)
    return tokenizer_X


def build_pos_tokenizer(x_train_pos):
    tokenizer_args = None
    tokenizer = KerasTokenizer(tokenizer_args=tokenizer_args,
                               build_embedding_matrix=False)
    tokenizer.build_vocab(x_train_pos)
    tokenizer_info = tokenizer.get_info()
    print('Tokenizer POS info: ', tokenizer_info)
    return tokenizer


def convert_text(texts, tokenizer, is_training=False, max_seq_length=None):
    text_ids = tokenizer.convert_tokens_to_ids(texts)
    # Padding
    if is_training:
        max_seq_length = int(np.quantile([len(seq) for seq in text_ids], 1))
    else:
        assert max_seq_length is not None

    text_ids = [seq + [0] * (max_seq_length - len(seq)) for seq in text_ids]
    text_ids = np.array([seq[:max_seq_length] for seq in text_ids])

    if is_training:
        return text_ids, max_seq_length
    else:
        return text_ids


def exact_match_to_numpy(exact_match, context_len):
    exact_match_np = np.zeros((exact_match.shape[0], context_len))
    for i, row in enumerate(exact_match):
        exact_match_np[i, :] = np.pad(np.array(row[:context_len]), (0, max(0, context_len - len(row))))
    return exact_match_np


def data_conversion(train_set, val_set, test_set, load):
    """
    Prepare the data in numpy arrays for the neural networks.
    """
    x_train_question, x_train_context, x_train_match, x_train_pos, y_train_answer_start, y_train_answer_end, \
    x_val_question, x_val_context, x_val_match, x_val_pos, y_val_answer_start, y_val_answer_end, \
    x_test_question, x_test_context, x_test_match, x_test_pos, y_test_answer_start, y_test_answer_end = extract_numpy_structures(
        train_set, val_set, test_set)

    if not load:
        tokenizer_x = build_tokenizer(x_train_context)
        tokenizer_pos = build_pos_tokenizer(x_train_pos)
        with open(UTILS_ROOT + 'tokenizer_x.pickle', 'wb') as handle:
            pickle.dump(tokenizer_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(UTILS_ROOT + 'tokenizer_pos.pickle', 'wb') as handle:
            pickle.dump(tokenizer_pos, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Tokenizers saved')
        print()
    else:
        with open(UTILS_ROOT + 'tokenizer_x.pickle', 'rb') as handle:
            tokenizer_x = pickle.load(handle)
        with open(UTILS_ROOT + 'tokenizer_pos.pickle', 'rb') as handle:
            tokenizer_pos = pickle.load(handle)
        print('Tokenizers loaded')

    print("Data conversion...")
    # Train
    x_train_context, max_seq_length_x_context = convert_text(x_train_context, tokenizer_x, True)
    x_train_question, max_seq_length_x_question = convert_text(x_train_question, tokenizer_x, True)
    x_train_pos, max_seq_length_x_pos = convert_text(x_train_pos, tokenizer_pos, True)
    x_train_match = exact_match_to_numpy(x_train_match, max_seq_length_x_context)
    print("Max token sequence context: {}".format(max_seq_length_x_context))
    print("Max token sequence pos: {}".format(max_seq_length_x_pos))
    print("Max token sequence question: {}".format(max_seq_length_x_question))
    print('X POS train shape: ', x_train_pos.shape)
    print('X context train shape: ', x_train_context.shape)
    print('X question train shape: ', x_train_question.shape)
    print('X match train shape: ', x_train_match.shape)

    # Val
    x_val_context = convert_text(x_val_context, tokenizer_x, False, max_seq_length_x_context)
    x_val_question = convert_text(x_val_question, tokenizer_x, False, max_seq_length_x_question)
    x_val_pos = convert_text(x_val_pos, tokenizer_pos, False, max_seq_length_x_pos)
    x_val_match = exact_match_to_numpy(x_val_match, max_seq_length_x_context)
    print('X context val shape: ', x_val_context.shape)
    print('X POS val shape: ', x_val_pos.shape)
    print('X question val shape: ', x_val_question.shape)
    print('X match val shape: ', x_val_match.shape)

    # Test
    x_test_context = convert_text(x_test_context, tokenizer_x, False, max_seq_length_x_context)
    x_test_question = convert_text(x_test_question, tokenizer_x, False, max_seq_length_x_question)
    x_test_pos = convert_text(x_test_pos, tokenizer_pos, False, max_seq_length_x_pos)
    x_test_match = exact_match_to_numpy(x_test_match, max_seq_length_x_context)
    print('X context test shape: ', x_test_context.shape)
    print('X POS test shape: ', x_test_pos.shape)
    print('X question test shape: ', x_test_question.shape)
    print('X match test shape: ', x_test_match.shape)

    return tokenizer_x, x_train_question, x_train_context, x_train_match, x_train_pos, y_train_answer_start, y_train_answer_end,\
           x_val_question, x_val_context, x_val_match, x_val_pos, y_val_answer_start, y_val_answer_end, x_test_question,\
           x_test_context, x_test_match, x_test_pos, y_test_answer_start, y_test_answer_end
