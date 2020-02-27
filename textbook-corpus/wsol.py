import ankura
import random
import time

it = 0
while it < 5:
  it += 1

  PRELABELED_SIZE = 1
  NUM_TOPICS = 180
  corpus = ankura.corpus.sciencep()
  
  exercise_ids = []

  split = ankura.pipeline.train_test_split(corpus, return_ids=True)
  (train_ids, train_corpus), (test_ids, test_corpus) = split
  
  for d in range(len(corpus[0])):
    try:
      if train_corpus[0][d][-1]['id'].find('excercisewsoln') > -1:
        exercise_ids.append(d)
    except:
      continue
  
  random.shuffle(exercise_ids)
  
  LABELS = ['negative', 'positive']
  GOLD_ATTR_NAME = 'binary_rating'
  LABEL_WEIGHT = 1
  smoothing=1e-4
  USER_LABEL_ATTR = 'user_label'
  ta_epsilon=1e-15
  rt_epsilon=1e-5
  THETA_ATTR = 'theta'
  starting_labeled_labels = set()
  
  random.seed(4880)
  
  all_label_set = set(LABELS)

  while PRELABELED_SIZE < len(exercise_ids):
    starting_labeled_ids = set(exercise_ids[:PRELABELED_SIZE])
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
  
    f = open("wsoln2.txt","a")
    f.write(str(PRELABELED_SIZE) + " - " + str(contingency.accuracy()) + "\n")
    f.close()
    print(contingency.accuracy(),len(anchor_vectors))
    PRELABELED_SIZE = PRELABELED_SIZE * 2
