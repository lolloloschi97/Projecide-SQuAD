from hyper_param import *
from model_definition import custom_loss_fn
import spacy
# spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")


def make_predictions(validation_df, x_val_context, x_val_context_pos, x_val_context_exact_match, x_val_question, y_val_start_enc, y_val_end_enc):
    custom_objects = {'custom_loss_fn': custom_loss_fn}
    model = tf.keras.models.load_model(UTILS_ROOT + "saved_model", custom_objects=custom_objects)
    pred_window = 10
    top_k = 5

    y_start, y_end = model.predict((x_val_context[:pred_window], x_val_context_pos[:pred_window], x_val_context_exact_match[:pred_window], x_val_question[:pred_window]))
    print(np.argmax(y_start, axis=-1))
    print(np.argmax(y_end, axis=-1))
    start_results = []
    end_results = []
    for i in range(pred_window):
        indexes_start = np.argsort(y_start[i, :])[-top_k:]
        indexes_end = np.argsort(y_end[i, :])[-top_k:]
        start_results.append(indexes_start)
        end_results.append(indexes_end)
    print(start_results)
    print(end_results)
    np.savetxt(UTILS_ROOT + "start_indexes.csv", np.array(start_results))
    np.savetxt(UTILS_ROOT + "end_indexes.csv", np.array(end_results))

    span_candidates = []
    for sample_id in range(pred_window):
        sample_candidates = []
        for i in range(top_k):
            for j in range(top_k):
                if start_results[sample_id][i] <= end_results[sample_id][j] and end_results[sample_id][j] - start_results[sample_id][i] < 25:
                    sample_candidates.append([start_results[sample_id][i], end_results[sample_id][j]])
        span_candidates.append(sample_candidates)

    print(span_candidates)

    best_couples_list = []
    for sample_id in range(pred_window):
        question_encoding = nlp(validation_df.loc[sample_id, 'question'])
        best_couple = None
        best_sim = 0
        for couple in span_candidates[sample_id]:
            sentence_extraction = ' '.join((validation_df.loc[sample_id, 'context'].split(' '))[int(couple[0]):int(couple[1]) + 1])
            sentence_encoding = nlp(sentence_extraction)
            sim = question_encoding.similarity(sentence_encoding)
            if sim > best_sim:
                best_sim = sim
                best_couple = couple
        best_couples_list.append(best_couple)
        print(best_sim)

    print(best_couples_list)
