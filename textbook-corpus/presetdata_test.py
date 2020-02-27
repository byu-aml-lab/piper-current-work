import ankura
import random
import time

PRELABELED_SIZE = 2

corpus = ankura.corpus.sciencep()
split = ankura.pipeline.train_test_split(corpus, return_ids=True)
(train_ids, train_corpus), (test_ids, test_corpus) = split
it = 0
while PRELABELED_SIZE < 80000:
  it += 1

  if PRELABELED_SIZE%10==0:
    print(PRELABELED_SIZE)

  LABELS = ['negative', 'positive']
  GOLD_ATTR_NAME = 'binary_rating'
  LABEL_WEIGHT = 1
  smoothing=1e-4
  NUM_TOPICS = 180
  USER_LABEL_ATTR = 'user_label'
  ta_epsilon=1e-15
  rt_epsilon=1e-5
  THETA_ATTR = 'theta'
  starting_labeled_labels = set()

  random.seed(4880)

  all_label_set = set(LABELS)

  while starting_labeled_labels != all_label_set:
    starting_labeled_ids = set(random.sample(range(len(train_corpus.documents)), PRELABELED_SIZE))
    starting_labeled_labels = set(train_corpus.documents[i].metadata[GOLD_ATTR_NAME] for i in starting_labeled_ids)

  Q, labels, D = ankura.anchor.build_labeled_cooccurrence(train_corpus,
                                                        GOLD_ATTR_NAME,
                                                        starting_labeled_ids,
                                                        label_weight=LABEL_WEIGHT,
                                                        smoothing=smoothing,
                                                        get_d=True,
                                                        labels=LABELS)

  gs_anchor_indices = ankura.anchor.gram_schmidt_anchors(train_corpus, Q,
                                                           k=NUM_TOPICS, return_indices=True)

  gs_anchor_vectors = Q[gs_anchor_indices]

  gs_anchor_tokens = [[train_corpus.vocabulary[index]] for index in gs_anchor_indices]

  for doc_id in starting_labeled_ids:
    doc = train_corpus.documents[doc_id]
    # Assign "user_label" to be the correct label
    doc.metadata[USER_LABEL_ATTR] = doc.metadata[GOLD_ATTR_NAME]
    doc.metadata['Prelabeled'] = True

  anchor_tokens = gs_anchor_tokens

  anchor_vectors = ankura.anchor.tandem_anchors(anchor_tokens, Q,
                                                  train_corpus, epsilon=ta_epsilon)

  C, topics = ankura.anchor.recover_topics(Q, anchor_vectors, epsilon=rt_epsilon, get_c=True)

  clf = ankura.topic.free_classifier_dream(train_corpus,
                                             attr_name=USER_LABEL_ATTR,
                                             labeled_docs=starting_labeled_ids, topics=topics,
                                             C=C, labels=labels)
  contingency = ankura.validate.Contingency()

  for doc in test_corpus.documents:
    gold = doc.metadata[GOLD_ATTR_NAME]
    pred = clf(doc)
    contingency[gold, pred] += 1

  f = open("numlabeled_accuracies.txt","a")
  f.write(str(PRELABELED_SIZE) + " - " + str(contingency.accuracy())+"\n")
  f.close()
  if it%5 == 0:
    it = 0
    PRELABELED_SIZE = PRELABELED_SIZE*2
