import numpy as np
import gensim
import gensim.downloader as gloader
import tensorflow as tf
from typing import List, Callable, Dict
from collections import OrderedDict
import tqdm

EMBEDDING_DIM = 100


def extract_numpy_structures(train_set, val_set):
    x_train_question = train_set['question'].values
    x_train_context = train_set['context'].values
    y_train_answer_start = train_set['answer_start'].values
    y_train_text = train_set['text'].values
    x_val_question = val_set['question'].values
    x_val_context = val_set['context'].values
    y_val_answer_start = val_set['answer_start'].values
    y_val_text = val_set['text'].values

    return x_train_question, x_train_context, y_train_answer_start, y_train_text, x_val_question, x_val_context, y_val_answer_start, y_val_text


def load_embedding_model(model_type: str,
                         embedding_dimension: int = 100) -> gensim.models.keyedvectors.KeyedVectors:
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
        self.key_to_index = None

        tokenizer_args = {} if tokenizer_args is None else tokenizer_args
        assert isinstance(tokenizer_args, dict) or isinstance(tokenizer_args, collections.OrderedDict)

        self.tokenizer_args = tokenizer_args

    def build_vocab(self, data, **kwargs):
        print('Fitting tokenizer...')
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(**self.tokenizer_args)
        self.tokenizer.fit_on_texts(data)
        print('Fit completed!')

        self.key_to_index = self.tokenizer.word_index

        if self.build_embedding_matrix:
            print('Loading embedding model! It may take a while...')
            self.embedding_model = load_embedding_model(model_type=self.embedding_model_type,
                                                        embedding_dimension=self.embedding_dimension)

            print('Checking OOV terms...')
            self.oov_terms = check_OOV_terms(embedding_model=self.embedding_model,
                                             word_listing=list(self.key_to_index.keys()))

            print('Building the embedding matrix...')
            self.embedding_matrix = build_embedding_matrix(embedding_model=self.embedding_model,
                                                           word_to_idx=self.key_to_index,
                                                           vocab_size=len(self.key_to_index) + 1,
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
            'vocab_size': len(self.key_to_index) + 1,
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
    tokenizer_X_claim = KerasTokenizer(tokenizer_args=tokenizer_args,
                                       build_embedding_matrix=True,
                                       embedding_dimension=EMBEDDING_DIM,
                                       embedding_model_type="glove")

    tokenizer_X.build_vocab(x_train_context)
    tokenizer_X = tokenizer_X.get_info()
    print('Tokenizer X info: ', tokenizer_X)
    return tokenizer_X


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


def data_conversion(train_set, val_set, load):
    x_train_question, x_train_context, y_train_answer_start, y_train_text, x_val_question, x_val_context, y_val_answer_start, y_val_text = extract_numpy_structures(
        train_set, val_set)
    if not load:
        with open(UTILS_ROOT + 'tokenizer_x.pickle', 'wb') as handle:
            tokenizer_x = build_tokenizer(x_train_context)
            pickle.dump(tokenizer_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Tokenizer X saved')
            print()
        with open(UTILS_ROOT + 'tokenizer_y.pickle', 'wb') as handle:
            tokenizer_y = build_tokenizer(y_train_text)
            pickle.dump(tokenizer_y, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Tokenizer y saved')
            print()
    else:
        with open(UTILS_ROOT + 'tokenizer_x.pickle', 'rb') as handle:
            tokenizer_x = pickle.load(handle)
            print('Tokenizer X loaded')
        with open(UTILS_ROOT + 'tokenizer_y.pickle', 'rb') as handle:
            tokenizer_y = pickle.load(handle)
            print('Tokenizer y loaded')

    # tokenizer_x.


def data_conversion(train_set, val_set):
    x_train_question, x_train_context, y_train_answer_start, y_train_text, x_val_question, x_val_context, y_val_answer_start, y_val_text = extract_numpy_structures(train_set, val_set)
    build_tokenizer(x_train_context)

    return x_train_question, x_train_context, y_train_answer_start, y_train_text, x_val_question, x_val_context, y_val_answer_start, y_val_text