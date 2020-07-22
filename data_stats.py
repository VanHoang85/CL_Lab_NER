import os


def process_data_file(path_to_data_dir):

    vocab = set()
    entities = set()
    entity_dict = dict()

    # read file
    files = [file for file in os.listdir(path_to_data_dir) if os.path.isfile(os.path.join(path_to_data_dir, file))]

    for file in files:
        with open(os.path.join(path_to_data_dir, file), 'r', encoding='utf-8') as f:
            for line in f:
                if len(line.split('\t')) == 5:
                    line = line.split('\t')
                    vocab.add(line[0])

                    if 'PER' in line[3] or 'ORG' in line[3] or 'LOC' in line[3]:
                        entities.add(line[0])

                        entity_type = line[3][2:]
                        if entity_type in entity_dict:
                            entity_dict[entity_type].add(line[0])
                        else:
                            entity_dict[entity_type] = {line[0]}

    return vocab, entities, entity_dict


def process_emb_file(path_to_emb_file):

    vocab = set()

    with open(path_to_emb_file, 'r', encoding='utf-8') as file:

        _, dim = file.readline().split(' ')  # first line

        for line in file:
            line = line.split(' ')

            if len(line) == int(dim) + 1:
                vocab.add(line[0])

    return vocab


def main():

    path_to_data = '/Users/vanhoang/PycharmProjects/CL_Lab/data'
    data_dirs = ['NER2016-TrainingData-3-3-2017-txt', 'TestData-16-9-2016']

    path_to_emb_dir = '/Users/vanhoang/PycharmProjects/CL_Lab'
    emb_files = ['FastText_ner.vec', 'glove.vie.25d.txt']

    for emb_file in emb_files:
        for data_dir in data_dirs:
            print('Data Dir', data_dir)
            print('Embedding File', emb_file)

            vocab_emb = process_emb_file(os.path.join(path_to_emb_dir, emb_file))
            vocab_data, entities, entity_dict = process_data_file(os.path.join(path_to_data, data_dir))

            oov = vocab_data - vocab_emb
            ooe = entities - vocab_emb

            print('='*50)
            print('Total tokens in data file', len(vocab_data))
            print('Total tokens in emb file', len(vocab_emb))
            print('Total entities', len(entities))
            print('\nNum of tokens not covered by emb file', len(oov))
            print('Percentage of tokens OOV', len(oov)/len(vocab_data)*100)
            print('\nNum of entities not covered by emb file', len(ooe))
            print('Percentage of entities OOV', len(ooe)/len(entities)*100)
            print('Some entities not covered', list(ooe)[:10])

            print('\nAmong those OOV entities....')
            for entity in entity_dict.keys():
                tokens = ooe & entity_dict[entity]
                print('\t{} belong to {}'.format(len(tokens), entity))

            print('=' * 100)


main()

