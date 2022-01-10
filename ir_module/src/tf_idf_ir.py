from hyper_param import *

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt

TOP_K = 5
N_UNIQUE_QUESTIONS = 5           # Nedeed for time/memory reasons
N_UNIQUE_CONTEXTS = 100            # Nedeed for time/memory reasons


class Plotter:
    def __init__(self, test_df, contexts_test_np):
        self.vdf = test_df
        self.cv = contexts_test_np

    def plot_performances(self, best_indexes_matrix_list, metric_name_list):
        """
        The ndarray 'positives' counts the top-1 to top-TOP_K matches. For each query check if its context is contained
        in the TOP_K list of output. If True, save the solution rank [0, TOP_K - 1].
        Then accumulate the positive rates for each k to plot a probability.
        Repeat the procedure for all the metrics
        """
        print("Plotting!")

        positives_arrays = []
        for ind in range(len(best_indexes_matrix_list)):
            best_indexes_matrix = best_indexes_matrix_list[ind]
            positives = np.zeros(TOP_K)
            for qi, indexes in enumerate(best_indexes_matrix):
                if self.vdf.loc[qi, 'context'] in [context for context in self.cv[list(indexes)]]:
                    positives[[context for context in self.cv[list(indexes)]].index(self.vdf.loc[qi, 'context'])] += 1
            for k in range(1, TOP_K):
                positives[k] += positives[k - 1]
            positives_arrays.append(positives)

        fig = plt.figure(figsize=[10, 7])
        ax = plt.subplot(1, 1, 1)
        l1 = ax.fill_between(list(range(1, TOP_K + 1)), (positives_arrays[0]) / len(best_indexes_matrix_list[0]), step='pre')
        l2 = ax.fill_between(list(range(1, TOP_K + 1)), (positives_arrays[1]) / len(best_indexes_matrix_list[1]), step='pre')
        ax.set_xlabel("Top-K")
        ax.set_ylabel("# of correct detections / # of queries")
        l1.set_facecolors([[.5, .5, .8, .5]])
        l1.set_edgecolors([[0, 0, .5, .6]])
        l1.set_linewidths([3])
        l2.set_facecolors([[.85, .6, .1, .5]])
        l2.set_edgecolors([[.95, .3, .1, .6]])
        l2.set_linewidths([3])
        ax.set_yticks(np.linspace(0., 1., 21))
        ax.set_xticks(np.arange(TOP_K + 1))
        ax.spines['right'].set_color((.8, .8, .8))
        ax.spines['top'].set_color((.8, .8, .8))
        ax.grid('on', alpha=0.5)
        ax.set_title("Metrics Comparison")
        ttl = ax.title
        ttl.set_weight('bold')
        ax.legend(metric_name_list)
        fig.show()


def jensenshannon_metric(questions_tf_idf, passages_test_tf_idf):
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


def cosine_similarity_metric(question_tf_idf, passage_test_tf_idf):
    """
    Compute a simple cosine similarity to find the closest document to each quesiton. Select TOP_K results.
    Fast but not very accurate.
    """

    cosine_matrix = cosine_similarity(question_tf_idf, passage_test_tf_idf)
    matrix_best_indexes = []
    print("Computing Cosine similarity")
    for i in tqdm.tqdm(range(cosine_matrix.shape[0])):
        matrix_best_indexes.append(np.flip(np.argsort(cosine_matrix[i, :])[
                                           -TOP_K:]))  # flip because the results because the last value has the highest probability

    return matrix_best_indexes


def tf_idf_ir(training_df, testing_df):
    """
    Compute TF-IDF using the 'TfidfVectorizer' of sklearn. The vectorizer is fitted on training contexts and used.
    Finally the test dataframe is transformed to elaborate the results on the test set.
    -- For efficiency reasons it was not possible using the whole dataset to compute the metrics. --
    """
    print("Computing TF-IDF values")
    contexts_train_np = training_df.drop_duplicates('context_id').context.to_numpy()
    contexts_test_np = testing_df.drop_duplicates('context_id').context.to_numpy()
    questions_test_np = testing_df.question.to_numpy()

    # Fit the tf-idf vectorizer on the training contexts
    tf_idf_vectorizer = TfidfVectorizer()
    tf_idf_vectorizer.fit(list(contexts_train_np))      # Fit on the whole training context

    pickle.dump(tf_idf_vectorizer, open(UTILS_ROOT + 'tfidf.pickle', 'wb'))

    # tf_idf_vectorizer = pickle.load(open(UTILS_ROOT + 'tfidf.pickle', 'rb'))

    ### TEST ###
    print("Computing tests...")
    # Transform the testing context and question based on already fitted vectorizer
    passages_test_tf_idf = tf_idf_vectorizer.transform(list(contexts_test_np))
    questions_test_tf_idf = tf_idf_vectorizer.transform(list(questions_test_np))

    cosine_best_indexes_test = cosine_similarity_metric(questions_test_tf_idf, passages_test_tf_idf)
    jensenshannon_best_indexes_test = jensenshannon_metric(questions_test_tf_idf[:N_UNIQUE_QUESTIONS], passages_test_tf_idf[:N_UNIQUE_CONTEXTS])

    plotter_test = Plotter(testing_df, contexts_test_np)
    plotter_test.plot_performances([cosine_best_indexes_test, jensenshannon_best_indexes_test],
                              ["Cosine similarity", "Jensen Shannon Distance"])
