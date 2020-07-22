"""
Disclaimer: The evaluation method is implemented based on ideas (i.e. convert entities to set and take intersection) from
            https://github.com/chakki-works/seqeval/blob/master/seqeval/metrics/sequence_labeling.py
"""


def eval_entity(y_true, y_pred, entity_tags):
    """
    Evaluate at entity level.
    Input data type: list of labels ==> [[ 'O', 'O', 'B-PER', 'I-PER', 'O', 'O', 'B-ORG', 'O', 'B-ORG', .....]]

    :param y_true: list of list of golden labels
    :param y_pred: list of list of predicted labels
    :param entity_tags: list of entity tags to calculate scores
    """

    # from the input, get list of entities
    y_true_entities = get_entities(y_true)
    y_pred_entities = get_entities(y_pred)

    # calculate scores (precision, recall, f1) for all entities
    scores_all = calculate_scores(y_true_entities, y_pred_entities)
    print_scores(scores_all, 'all entities')

    # calculate scores (precision, recall, f1) for each entity
    # first, get unique and without BIO info in the tags
    tags = []
    for tag in entity_tags:
        if tag != 'O' and tag[2:] not in tags:
            tags.append(tag[2:])

    for tag in tags:

        # get list of that particular entity
        y_true_entity_tag = []
        y_pred_entity_tag = []

        for sent_true, sent_pred in zip(y_true_entities, y_pred_entities):
            sent_true_tag = [entity for entity in sent_true if entity[0] == tag]
            sent_pred_tag = [entity for entity in sent_pred if entity[0] == tag]

            y_true_entity_tag.append(sent_true_tag)
            y_pred_entity_tag.append(sent_pred_tag)

        scores_entity = calculate_scores(y_true_entity_tag, y_pred_entity_tag)
        print('Number of true', tag, str(sum([len(sent) for sent in y_true_entity_tag])))
        print('Number of pred', tag, str(sum([len(sent) for sent in y_pred_entity_tag])))
        print_scores(scores_entity, tag)


def get_entities(labels):
    """
    Get list of list of entities from a list of list of labels. Each entity should be a set ('entity_tag', start_span, end_span)
    An entity --> ('PER', 2, 3)
    List of entities --> [ ('PER', 2, 3), ('ORG', 1, 1), ('ORG', 3, 3) ... ]

    :param labels: list of list of labels

    :return: list of list of entities
    """

    entities = list()

    for sent in labels:
        sent_entities = list()
        idx_token = 0
        while idx_token < len(sent):
            if sent[idx_token].startswith('B'):
                tag = sent[idx_token][2:]  # get the entity tag, not the BI tag
                start_span = idx_token

                # get end_span from the remaining of the seq
                end_span = get_end_span(sent[idx_token:], start_span)

                entity = (tag, start_span, end_span)
                sent_entities.append(entity)
                idx_token = end_span + 1
            else:
                idx_token += 1
        entities.append(sent_entities)

    return entities


def get_end_span(seq, start):
    """
    Get the end span of the entity after detectung the start

    :param seq: the list of labels to detect the end span of the entity
    :param start: start of the entity span

    :return: index of the end span
    """

    # if B or O --> end index is at that index
    # if I --> continue checking
    # goal: find the first token with B or O of seq
    # note: if reaching the end of seq --> end = end idx of seq
    for idx in range(len(seq)):
        if idx == len(seq) - 1 or seq[idx+1].startswith('B') or seq[idx+1].startswith('O'):
            return start + idx


def calculate_scores(true_entities, pred_entities):

    tp, fp, fn = 0, 0, 0

    for sent_true, sent_pred in zip(true_entities, pred_entities):
        # get the intersection of two sets, which is also TP value
        intersection = set(sent_true) & set(sent_pred)

        tp_sent = len(intersection)
        fp_sent = len(sent_pred) - len(intersection)
        fn_sent = len(sent_true) - len(intersection)

        tp += tp_sent
        fp += fp_sent
        fn += fn_sent

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1


def print_scores(scores, value_type):

    print('Scores for', value_type, ': {:.2f}, {:.2f}, {:.2f}\n'.format(scores[0]*100, scores[1]*100, scores[2]*100))

