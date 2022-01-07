from hyper_param import *
from model_definition import custom_loss_fn
import nltk
from nltk.corpus import stopwords
import spacy
# spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")


PRED_WINDOW = 20
TOP_K = 5
MAX_ANSWER_LEN = 20

try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))


def remove_stopwords(text: str) -> str:
    return ' '.join([x for x in text.split() if x and x not in STOPWORDS])


def make_predictions(test_df, x_test_context, x_test_context_pos, x_test_context_exact_match, x_test_question):
    custom_objects = {'custom_loss_fn': custom_loss_fn}
    model = tf.keras.models.load_model(UTILS_ROOT + "saved_model", custom_objects=custom_objects)

    """
    Make PRED_WINDOW predictions and save the TOP_K probabilities indexes
    """
    y_start, y_end = model.predict((x_test_context[:PRED_WINDOW], x_test_context_pos[:PRED_WINDOW], x_test_context_exact_match[:PRED_WINDOW], x_test_question[:PRED_WINDOW]))
    print(np.argmax(y_start, axis=-1))
    print(np.argmax(y_end, axis=-1))
    start_results = []
    end_results = []
    for i in range(PRED_WINDOW):
        indexes_start = np.argsort(y_start[i, :])[-TOP_K:]
        indexes_end = np.argsort(y_end[i, :])[-TOP_K:]
        start_results.append(indexes_start)
        end_results.append(indexes_end)
    print(start_results)
    print(end_results)

    """
    Create all the possible [start, end] indexes for each context picked from the TOP_K starts and ends.
    Check start <= end. Check that the answer is no more than 25 words
    """
    span_candidates = []
    for sample_id in range(PRED_WINDOW):
        sample_candidates = []
        for i in range(TOP_K):
            for j in range(TOP_K):
                if start_results[sample_id][i] <= end_results[sample_id][j] and end_results[sample_id][j] - start_results[sample_id][i] < MAX_ANSWER_LEN:
                    sample_candidates.append([start_results[sample_id][i], end_results[sample_id][j]])
        span_candidates.append(sample_candidates)

    """
    Evaluate the [start, end] tuple based on the similarity between the question and the extracted sentence.
    """
    best_couples_list = []
    for sample_id in range(PRED_WINDOW):
        question_encoding = nlp(remove_stopwords(test_df.loc[sample_id, 'question']))
        best_couple = None
        best_sim = 0
        for couple in span_candidates[sample_id]:
            sentence_extraction = ' '.join((test_df.loc[sample_id, 'context'].split(' '))[int(couple[0]):int(couple[1]) + 1])
            sentence_encoding = nlp(remove_stopwords(sentence_extraction))
            sim = question_encoding.similarity(sentence_encoding)
            if sim > best_sim:
                best_sim = sim
                best_couple = couple
        best_couples_list.append(best_couple)
    print(best_couples_list)
    return best_couples_list


def evaluate(best_couples_list, y_test_start_index, y_test_end_index):
    true_positive_list = []
    false_positive_list = []
    false_negative_list = []
    for i, couple in enumerate(best_couples_list):
        false_positive = 0
        true_positive = 0
        false_negative = 0
        for value in range(couple[0], couple[1] + 1):
            if value < y_test_start_index[i] or value > y_test_end_index[i]:
                false_positive += 1
            elif y_test_start_index[i] <= value <= y_test_end_index[i]:
                true_positive += 1
        for value in range(y_test_start_index[i], y_test_end_index[i] + 1):
            if value < couple[0] or value > couple[1]:
                false_negative += 1
        true_positive_list.append(true_positive)
        false_negative_list.append(false_negative)
        false_positive_list.append(false_positive)

    for i in range(len(true_positive_list)):
        precision = true_positive_list[i] / (true_positive_list[i] + false_positive_list[i])
        recall = true_positive_list[i] / (true_positive_list[i] + false_negative_list[i])
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
        print("Precision", precision)
        print("Recall", recall)
        print("F1 score", f1_score)


def predict_and_evaluate(test_df, x_test_context, x_test_context_pos, x_test_context_exact_match, x_test_question,
                         y_test_start_index, y_test_end_index):
    best_couples_list = make_predictions(test_df, x_test_context, x_test_context_pos, x_test_context_exact_match,
                                         x_test_question)
    evaluate(best_couples_list, y_test_start_index, y_test_end_index)

