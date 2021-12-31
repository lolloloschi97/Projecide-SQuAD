import copy

from hyper_param import *


def query_document_counter(dataframe):
    print("Query Document counter started...")
    PRIORITY_LIST = {v: k for k, v in POS_LIST.items()}
    queries = dataframe[['question', 'context_id']].to_numpy()
    contexts = dataframe[['document_tag', 'context_id']].drop_duplicates('context_id').to_numpy()   # row index == id
    predicted_context_id = []
    for qi in range(queries.shape[0]):
        current_query_words = queries[qi, 0].split(' ')
        best_match_id = []
        best_match_count = None
        for cid in range(contexts.shape[0]):
            current_match = dict.fromkeys(POS_LIST.keys())
            for tag_category in current_match.keys():
                acc = 0
                for word in contexts[cid, 0][tag_category]:
                    acc += current_query_words.count(word)
                current_match[tag_category] = acc
            if cid == 0:
                best_match_id.append(cid)
                best_match_count = copy.deepcopy(current_match)
            else:
                for priority in range(1, max(PRIORITY_LIST.keys()) + 1):
                    if current_match[PRIORITY_LIST[priority]] > best_match_count[PRIORITY_LIST[priority]]:
                        best_match_id.append(cid)
                        best_match_count = current_match
        predicted_context_id.append((queries[qi, 1], best_match_id[-10:]))
        print("qi: ", qi)

    pos = 0
    neg = 0
    for pred in predicted_context_id:
        real, predictions = pred
        if real in predictions:
            pos += 1
        else:
            neg += 1
    print("pos: ", pos)
    print("neg: ", neg)
