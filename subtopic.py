import model

def main():
    filter_keywords = ['smoking', 'tobacco', 'cigarette', 'cigar', 'hookah', 'hooka']
    [df_document_topic, num_of_topics] = model.main(filter_keywords)

    # various search params to get best combination
    # n_components is Number of topics.
    search_params = {'n_components': [
        4, 10, 20], 'learning_decay': [.8, .12]}

    for i in range(num_of_topics):
        print ('Sub-Topic for Topic ' + str(i))
        filter_doc = df_document_topic.loc[df_document_topic['dominant_topic'] == i]
        processed_docs = list(filter_doc['doc'])
        [best_lda_model_sub_topic, num_of_sub_topics, vectorizer] = model.train_lda(processed_docs, search_params)
        model.show_result(best_lda_model_sub_topic, processed_docs, num_of_sub_topics, vectorizer)
main()