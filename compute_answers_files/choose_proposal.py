from settings_file import *
from tf_idf_ir import load_vectorizer, compare_question_answer
import nltk
from nltk.corpus import stopwords

try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

# TF-IDF performs better without stopwords according to our experiments
def remove_stopwords(text: str) -> str:
    return ' '.join([x for x in text.split() if x and x not in STOPWORDS])


def choose_best_prediction_proposal(top_k_predictions_dict, complete_input_df):
    """
    'top_k_predictions_dict': dictionary {query_1: [pred_1, ..., pred_k], ..., query_n: [pred_1, ..., pred_k]}
    'complete_input_df': the original dataframe used to retrieve the question from the id

    return: dictionary with only one pred for query {query_1: pred_1, ..., query_n: pred_n} like for a simple QA task
    """
    queries = complete_input_df['question'].to_numpy()
    tfidf_vectorizer = load_vectorizer()
    predictions_dict = {}
    print("Select the best answer for every question among the k candidates...")
    for i, (query_id, pred_list) in enumerate(tqdm.tqdm(top_k_predictions_dict.items())):
        query = queries[i]
        best_answer_index = compare_question_answer(query, list(map(remove_stopwords, pred_list)), tfidf_vectorizer)
        predictions_dict[query_id] = pred_list[best_answer_index]
    return predictions_dict
