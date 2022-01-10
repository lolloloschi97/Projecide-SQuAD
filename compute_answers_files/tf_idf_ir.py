from settings_file import *
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer


def jensenshannon_metric(questions_tf_idf, passages_test_tf_idf, TOP_K):
    """
    This function implements the Jensen-Shannon distance above
    and returns the top k indices of the smallest jensen shannon distances.
    Slow but more accurate.
    """
    queries_matrix = questions_tf_idf.toarray()
    contexts_matrix = passages_test_tf_idf.toarray()
    n_queries = queries_matrix.shape[0]
    n_contexts = contexts_matrix.shape[0]
    matrix_best_indexes = []
    print("Computing Jensen Shannon metric")
    for qi in tqdm.tqdm(range(n_queries)):
        distances_query_contexts = np.empty((n_contexts,))
        for ci in range(n_contexts):
            distances_query_contexts[ci] = jensenshannon(queries_matrix[qi, :], contexts_matrix[ci, :])
        matrix_best_indexes.append(np.argsort(distances_query_contexts)[:TOP_K])
    return matrix_best_indexes


def cosine_similarity_metric(question_tf_idf, passage_test_tf_idf, TOP_K):
    """
    Compute a simple cosine similarity to find the closest document to each quesiton. Select TOP_K results.
    Fast but not very accurate.
    """

    cosine_matrix = cosine_similarity(question_tf_idf, passage_test_tf_idf)
    matrix_best_indexes = []
    print("Computing Cosine similarity")
    for i in tqdm.tqdm(range(cosine_matrix.shape[0])):
        # flip because the results because the last value has the highest probability
        matrix_best_indexes.append(np.flip(np.argsort(cosine_matrix[i, :])[-TOP_K:]))

    return matrix_best_indexes


def load_vectorizer():
    # Load the vectorizer fir on training data
    cwd = os.getcwd()
    tf_idf_vectorizer = pickle.load(open(cwd + UTILS_ROOT + 'tfidf.pickle', 'rb'))
    print("TF-IDF Vectorizer loaded")
    return tf_idf_vectorizer


def tf_idf_ir(input_df, mode, TOP_K):
    """
    Called to pick the tok_k more appropriate document for the given queries.

    'input_df' : cleaned input dataframe
    'mode' : 1 for cosine, 2 for JS distance.
    'TOP_K' : k best result to return for each question

    return : list of ndarray containing the the TOP_K document indexes for each question
    """

    print("Computing TF-IDF values")
    contexts_input_np = input_df.drop_duplicates('context_id').context.to_numpy()
    questions_input_np = input_df.question.to_numpy()

    # Load the vectorizer fir on training data
    tf_idf_vectorizer = load_vectorizer()

    print("Computing documents and questions trasformation")
    # Transform the testing context and question based on pre-fit vectorizer
    passages_input_tf_idf = tf_idf_vectorizer.transform(list(contexts_input_np))
    questions_input_tf_idf = tf_idf_vectorizer.transform(list(questions_input_np))

    best_indexes = None
    if mode == 1:
        best_indexes = cosine_similarity_metric(questions_input_tf_idf, passages_input_tf_idf, TOP_K)
    elif mode == 2:
        best_indexes = jensenshannon_metric(questions_input_tf_idf, passages_input_tf_idf, TOP_K)

    return best_indexes


def compare_question_answer(question, answers_list, tf_idf_vectorizer):
    """
    Called to choose among the options return from several documents.

    'question': question text cleaned
    'answers_lsit': list of str. List of possible answers coming from different documents
    'tf_idf_vectorizer': TfIDFVectorizer() already loaded (it avoid to re-load each time)

    return: int. The best answer index according to jensenshannon distance (minimum distance).
    """
    answers_input_tf_idf = tf_idf_vectorizer.transform(answers_list)
    question_input_tf_idf = tf_idf_vectorizer.transform([question])
    question_mat = question_input_tf_idf.toarray()
    answers_mat = answers_input_tf_idf.toarray()
    n_answers = answers_mat.shape[0]

    # Compute the distance of the question from all the predictions using the jensen shannon distance (more accurate than cosine)
    distances = np.empty((n_answers, ))
    for ai in range(n_answers):
        distances[ai] = jensenshannon(question_mat[0, :], answers_mat[ai, :])
    return np.argmin(distances)
