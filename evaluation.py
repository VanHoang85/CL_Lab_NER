"""
Disclaimer: The evaluation method is implemented based on ideas (i.e. convert entities to set and take intersection) from
            https://github.com/chakki-works/seqeval/blob/master/seqeval/metrics/sequence_labeling.py
"""


def eval_entity(y_true, y_pred, entity_tags):
    """
    Evaluate at entity level.
    Input data type: list of labels ==> [ 'O', 'O', 'B-PER', 'I-PER', 'O', 'O', 'B-ORG', 'O', 'B-ORG', .....]

    :param y_true: list of golden labels
    :param y_pred: list of predicted labels
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
        y_true_entity = [entity for entity in y_true_entities if entity[0] == tag]
        y_pred_entity = [entity for entity in y_pred_entities if entity[0] == tag]

        scores_entity = calculate_scores(y_true_entity, y_pred_entity)
        print('Number of true', tag, str(len(y_true_entity)))
        print('Number of pred', tag, str(len(y_pred_entity)))
        print_scores(scores_entity, tag)


def get_entities(labels):
    """
    Get list of entities from a list of labels. Each entity should be a set ('entity_tag', start_span, end_span)
    An entity --> ('PER', 2, 3)
    List of entities --> [ ('PER', 2, 3), ('ORG', 1, 1), ('ORG', 3, 3) ... ]

    :param labels: list of labels

    :return: list of entities
    """

    entities = list()

    idx = 0
    while idx < len(labels):
        if labels[idx].startswith('B'):
            tag = labels[idx][2:]  # get the entity tag, not the BI tag
            start_span = idx

            # get end_span from the remaining of the seq
            end_span = get_end_span(labels[idx:], start_span)

            entity = (tag, start_span, end_span)
            entities.append(entity)
            idx = end_span + 1
        else:
            idx += 1

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
    # get the intersection of two sets, which is also TP value
    intersection = set(true_entities) & set(pred_entities)

    tp = len(intersection)
    fp = len(pred_entities) - len(intersection)
    fn = len(true_entities) - len(intersection)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1


def print_scores(scores, value_type):

    print('Scores for', value_type, ': {:.2f}, {:.2f}, {:.2f}\n'.format(scores[0], scores[1], scores[2]))

