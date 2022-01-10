from settings_file import *
import nltk
from tensorflow.python.ops import math_ops, nn_ops
from nltk.corpus import stopwords
import spacy
try:
    nlp = spacy.load("en_core_web_lg")
except:
    spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

TOP_K_ANSWERS = 4
MAX_ANSWER_LEN = 18     # empirically observed from training set

"""
Whenever a custom object is defined (custom loss function in our case), Tensorflow requires to know the definition of 
that object before loading the model. So if model.compile() contains custom_loss_fn then it must appear in a dictionary
with the same name and definition to be passed as argument to model.load(). 
The two functions below are copied from base_project/src/model_definition.py only for this purpose. More info there. 
"""

def weighted_cross_entropy_with_logits_modified(labels, logits, pos_weight, neg_weights, name=None):
    log_weight = neg_weights + (pos_weight - neg_weights) * labels
    return math_ops.add(
        (1 - labels) * logits * neg_weights,
        log_weight * (math_ops.log1p(math_ops.exp(-math_ops.abs(logits))) +
                      nn_ops.relu(-logits)),  # pylint: disable=invalid-unary-operand-type
        name=name)

def custom_loss_fn(y_true, y_pred):
    pos_weight = tf.constant(1.0)
    neg_weight = tf.constant(0.0)
    bn_crossentropy = weighted_cross_entropy_with_logits_modified(y_true, y_pred, pos_weight, neg_weight)
    return tf.reduce_mean(bn_crossentropy, axis=-1)

custom_objects = {'custom_loss_fn': custom_loss_fn}


def load_model():
    print()
    print("Loading model...")
    cwd = os.getcwd()
    model = tf.keras.models.load_model(cwd + UTILS_ROOT + "saved_model", custom_objects=custom_objects)
    print("Model loaded!")
    return model

"""
STOPWORDS are defined to better evaluate the similarity between different answers proposals and question.
"""

try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))


def remove_stopwords(text: str) -> str:
    return ' '.join([x for x in text.split() if x and x not in STOPWORDS])


# Load the pre-trained model and make predictions
def make_predictions(model, input_df, x_input_context, x_input_pos_enc, x_input_match, x_input_question):

    """
    Make PRED_WINDOW predictions and save the TOP_K probabilities indexes
    """
    print("Make predictions...")
    y_start, y_end = model.predict((x_input_context, x_input_pos_enc, x_input_match, x_input_question))
    start_results = []
    end_results = []
    for i in range(len(input_df)):
        indexes_start = np.argsort(y_start[i, :])[-TOP_K_ANSWERS:]
        indexes_end = np.argsort(y_end[i, :])[-TOP_K_ANSWERS:]
        start_results.append(indexes_start)
        end_results.append(indexes_end)

    """
    Create all the possible [start, end] indexes for each context picked from the TOP_K starts and ends.
    Check start <= end. Check that the answer is no more than 25 words
    """
    span_candidates = []
    for sample_id in range(len(input_df)):
        sample_candidates = []
        for i in range(TOP_K_ANSWERS):
            for j in range(TOP_K_ANSWERS):
                if start_results[sample_id][i] <= end_results[sample_id][j] and end_results[sample_id][j] - start_results[sample_id][i] < MAX_ANSWER_LEN:
                    sample_candidates.append([start_results[sample_id][i], end_results[sample_id][j]])
        span_candidates.append(sample_candidates)

    """
    Evaluate the [start, end] tuple based on the similarity between the question and the extracted sentence.
    """
    print("Choose the best answer")
    best_couples_list = []
    for sample_id in tqdm.tqdm(range(len(input_df))):
        question_encoding = nlp(remove_stopwords(input_df.loc[sample_id, 'question']))
        best_couple = [0, 0]
        best_sim = 0
        for couple in span_candidates[sample_id]:
            sentence_extraction = ' '.join((input_df.loc[sample_id, 'context'].split(' '))[int(couple[0]):int(couple[1]) + 1])
            sentence_encoding = nlp(remove_stopwords(sentence_extraction))
            try:
                sim = question_encoding.similarity(sentence_encoding)
                if sim > best_sim:
                    best_sim = sim
                    best_couple = couple
            except UserWarning:
                pass
        best_couples_list.append(best_couple)
    return best_couples_list


def format_predictions(input_df, best_couples_list):
    """
    'input_df' : the original dataframe to extract the answer text from start and end indexes.
    'best_couples_list' : list of pairs containing, for each query, the start and the end index of the answer in the relative context

    return: ordered list of start-end index answer. Return a dict {id: answer}-like
    """
    predictions_pairs = []
    for i, row in input_df.iterrows():
        list_of_words = row['context'].split()
        selected_words = list_of_words[best_couples_list[i][0]:best_couples_list[i][1] + 1]
        build_sentence = ' '.join(selected_words)
        predictions_pairs.append((row['id'], build_sentence))
    return predictions_pairs


def predict(model, input_df, x_input_context, x_input_pos_enc, x_input_match, x_input_question):
    best_couples_list = make_predictions(model, input_df, x_input_context, x_input_pos_enc, x_input_match, x_input_question)
    predictions_pairs = format_predictions(input_df, best_couples_list)
    return dict(predictions_pairs)

