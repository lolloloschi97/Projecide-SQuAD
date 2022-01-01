import random

from hyper_param import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.test.utils import get_tmpfile
import collections


def doc2vec(training_df, validation_df):
    context_values = list(training_df['context'].drop_duplicates().apply(lambda x: x.split(' ')).to_numpy())
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(context_values)]
    print(documents[:2])
    print("Doc2Vec...")
    model = Doc2Vec(documents, vector_size=200, window=2, min_count=1, workers=4, epochs=12, seed=16)

    #fname = get_tmpfile("my_doc2vec_model")
    #model.save(fname)
    #model = Doc2Vec.load(fname)     # you can continue training with the loaded model!
    model.train(documents,  total_examples=model.corpus_count, epochs=model.epochs)

    val_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(list(validation_df['context'].drop_duplicates().apply(lambda x: x.split(' ')).to_numpy()))]
    # vector = model.infer_vector(val_documents[0])
    # print(vector)

    ranks = []
    second_ranks = []
    for doc_id in range(len(documents)):
        inferred_vector = model.infer_vector(documents[doc_id].words)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)

        second_ranks.append(sims[1])
    counter = collections.Counter(ranks)
    print("Training (0 means exact match): ", counter)

    # VALIDATION
    # Pick a random document from the test corpus and infer a vector from the model
    doc_id = random.randint(0, len(val_documents) - 1)
    inferred_vector = model.infer_vector(val_documents[doc_id].words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

    # Compare and print the most/median/least similar documents from the train corpus
    print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(val_documents[doc_id].words)))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(documents[sims[index][0]].words)))