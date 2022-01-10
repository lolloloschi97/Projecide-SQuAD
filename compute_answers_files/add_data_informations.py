from settings_file import *
import nltk
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def add_exact_match(dataframe):
    """
    Take a dataframe as input and return the same dataframe with a column indicating whenever a word in the context
    appears in the current query. I.e.
           query                            context                     exact_match
    "Where is the cat?"              "The cat is on the table"     [1, 1, 1, 0, 1, 0]

    In case of document retrieval, it must be performed after building the new query-document pairs.
    """
    print()
    print("Add exact match. It takes a while...")
    match_column = []
    for i, row in dataframe.iterrows():
        match_list = np.zeros(len(row.context.split(' ')))
        for k, c_word in enumerate(row.context.split(' ')):
            if c_word in row.question.split(' '):
                match_list[k] = 1
        match_column.append(list(match_list))
    dataframe['exact_match'] = pd.DataFrame({'exact_match': match_column})
    print("Exact match computed.")
    return dataframe


def add_POS_tagging(dataframe):
    """
    Return a dataframe with one more column containing the POS tagging of the context
    """
    def apply_pos_tagging(x):
        doc = nltk.word_tokenize(x)
        tag_list = nltk.pos_tag(doc)
        for i in range(len(tag_list)):
            tag_list[i] = tag_list[i][1]
        return tag_list

    print()
    print("Apply POS tagging")
    orig = copy.deepcopy(dataframe)
    # Duplicate the context column ad apply POS tagging only once for context_id (drop duplicates)
    dataframe['pos'] = dataframe['context']
    dataframe = dataframe[['context', 'context_id', 'pos']]
    dataframe = dataframe.drop_duplicates('context_id')
    dataframe.pos = dataframe.pos.apply(lambda x: apply_pos_tagging(x))

    # build a dictionary to map the original context_id to the correct POS tagging.
    dict_pos = dict(zip(list(dataframe.context_id.to_numpy()), list(dataframe.pos.to_numpy())))
    orig['pos'] = orig['context_id']
    orig.pos = orig.pos.apply(lambda x: dict_pos[x])
    print("Tags applied!")
    return orig


