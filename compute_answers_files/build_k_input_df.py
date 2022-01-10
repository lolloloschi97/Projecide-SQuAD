from settings_file import *


def build_k_input_df(complete_input_df, best_indexes, top_k):
    """
    'complete_input_df': preprocessed dataframe containing cleaned data
    'best_indexes': list of ndarray. Example: the m-th ndarray element of the n-th list element contains the m-th most
                    likely document for the n-th query. The context_id represents the document.
    'top_k': number of elements in the ndarray

    return: a list of K dataframe with all the selected query-context pairs
    """
    print()
    print("Build the top-k possible dataframes based on the document retrieval outcomes")
    query_df = complete_input_df[['id', 'question']]
    context_df = (complete_input_df[['context_id', 'context', 'pos']]).drop_duplicates('context_id')

    # create the k possible 'context_id' columns
    context_ids_list = []
    for k in range(top_k):
        new_context_id = np.zeros((len(best_indexes),))
        for i, ids in enumerate(best_indexes):
            new_context_id[i] = ids[k]
        context_ids_list.append(new_context_id)

    # use the 'context_id' just computed to merge the contexts of context_df (keeping the information about POS and exact match)
    input_df_list = []
    for k in range(top_k):
        new_input_df = copy.deepcopy(query_df)
        new_input_df['context_id'] = (pd.DataFrame({'context_id': context_ids_list[k]})).astype(int)
        new_input_df = new_input_df.merge(context_df, how='left', on='context_id', sort=False)
        input_df_list.append(new_input_df)

    print("Dataframes built")
    return input_df_list

