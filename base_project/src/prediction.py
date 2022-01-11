from hyper_param import *
from model_definition import custom_loss_fn
import nltk
from nltk.corpus import stopwords
import spacy
# spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")


TOP_K = 4
MAX_ANSWER_LEN = 18

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
    y_start, y_end = model.predict((x_test_context, x_test_context_pos, x_test_context_exact_match, x_test_question))
    print(np.argmax(y_start, axis=-1))
    print(np.argmax(y_end, axis=-1))
    predicted_start_indexes = []
    predicted_end_indexes = []
    for i in range(len(x_test_question)):
        indexes_start = np.argsort(y_start[i, :])[-TOP_K:]
        indexes_end = np.argsort(y_end[i, :])[-TOP_K:]
        predicted_start_indexes.append(indexes_start)
        predicted_end_indexes.append(indexes_end)

    """
    Create all the possible [start, end] indexes for each context picked from the TOP_K starts and ends.
    Check start <= end. Check that the answer is no more than 25 words
    """
    span_candidates = []
    for sample_id in range(len(x_test_question)):
        sample_candidates = []
        for i in range(TOP_K):
            for j in range(TOP_K):
                if predicted_start_indexes[sample_id][i] <= predicted_end_indexes[sample_id][j] and predicted_end_indexes[sample_id][j] - predicted_start_indexes[sample_id][i] < MAX_ANSWER_LEN:
                    sample_candidates.append([predicted_start_indexes[sample_id][i], predicted_end_indexes[sample_id][j]])
        span_candidates.append(sample_candidates)

    """
    Evaluate the [start, end] tuple based on the similarity between the question and the extracted sentence.
    """
    best_couples_list = []
    for sample_id in range(len(x_test_question)):
        question_encoding = nlp(remove_stopwords(test_df.loc[sample_id, 'question']))
        best_couple = [0, 0]
        best_sim = 0
        for couple in span_candidates[sample_id]:
            # extract a sentence from indexes
            sentence_extraction = ' '.join((test_df.loc[sample_id, 'context'].split(' '))[int(couple[0]):int(couple[1]) + 1])
            sentence_encoding = nlp(remove_stopwords(sentence_extraction))
            try:
                sim = question_encoding.similarity(sentence_encoding)
                if sim > best_sim:
                    best_sim = sim
                    best_couple = couple
            except UserWarning:
                pass
        best_couples_list.append(best_couple)
    return best_couples_list, predicted_start_indexes, predicted_end_indexes


def evaluate(predicted_start_indexes, predicted_end_indexes, y_test_start_index, y_test_end_index):
    """
    Evaluate the probability that the correct index appears in the TOP-4 outcomes
    """
    start_positive = 0
    end_positive = 0
    for i in range(len(y_test_end_index)):
        if y_test_start_index[i] in predicted_start_indexes[i]:
            start_positive += 1
        if y_test_end_index[i] in predicted_end_indexes[i]:
            end_positive += 1
    print("Precision in Top-4, start index:", start_positive/len(y_test_end_index))
    print("Precision in Top-4, end index:", end_positive/len(y_test_end_index))


def predict_and_evaluate(test_df, x_test_context, x_test_context_pos, x_test_context_exact_match, x_test_question,
                         y_test_start_index, y_test_end_index):
    best_couples_list, predicted_start_indexes, predicted_end_indexes = make_predictions(test_df, x_test_context, x_test_context_pos, x_test_context_exact_match,
                                         x_test_question)
    evaluate(predicted_start_indexes, predicted_end_indexes, y_test_start_index, y_test_end_index)

