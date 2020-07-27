import argparse
import os

import torch
from models import *
import dataset
import evaluation


def get_model(args, use_crf, num_ner_tags, input_dim, emb_weights=None):
    """
    Create model accordingly to the arguments

    :param args: arguments
    :param use_crf: True if use crf layer for prediction
    :param num_ner_tags: number of ner tags
    :param input_dim: the dimension of an input representation
    :param emb_weights: the embedding weights to initialize the embedding layer if used, None if not
    :return: the model
    """

    if args.model_architecture == 'crf':
        model = CRF_NER(num_ner_tags=num_ner_tags, input_dim=input_dim)
        print('Running experiment with CRF')

    elif args.model_architecture == 'lstm':
        model = LSTM_NER(args, use_crf=use_crf, num_ner_tags=num_ner_tags, input_dim=input_dim, emb_weights=emb_weights)
        print('Running experiment with LSTM')

    else:
        # lstm-crf
        model = LSTM_CRF_NER(args, use_crf=use_crf, num_ner_tags=num_ner_tags, input_dim=input_dim, emb_weights=emb_weights)
        print('Running experiment with LSTM CRF')

    return model.to(device)


def load_data_object(args, data_object):
    """
    Load data object if exists. Otherwise, create a new one

    :param args: the arguments
    :param data_object: name of the data object
    :return: the corpus
    """

    if os.path.exists(os.path.join(args.data_path, data_object)):
        print('Loading data file....')
        with open(os.path.join(args.data_path, data_object), 'rb') as file:
            corpus = torch.load(file)
    else:
        print('Creating data file....')
        corpus = dataset.Dataset(no_misc=args.no_misc, path=args.data_path, model_arch='nn', input_type=args.input_type,
                                 emb_file=os.path.join(os.getcwd(), args.path_to_emb_file))

        with open(os.path.join(args.data_path, data_object), 'wb') as file:
            torch.save(corpus, file)

    return corpus


def load_pre_trained_embeddings(path_to_emb_file):
    """
    Load pre-trained embedding file

    :param path_to_emb_file: path to the embedding file
    :return: (1) a dict mapping token to its weights, (2) a dict mapping token to its index, (3) embedding dimension
    """

    word2vector = dict()
    word2idx = dict()

    with open(path_to_emb_file, 'r', encoding='utf-8') as file:
        line = file.readline().strip().split(' ')
        vocab_size, dim = int(line[0]), int(line[1])

        # add PAD token
        word2idx['PAD'] = 0
        word2vector['PAD'] = np.zeros(dim, dtype=float)

        # add UNK token
        word2idx['UNK'] = 1
        word2vector['UNK'] = np.random.normal(loc=0.0, scale=0.6, size=(dim,))

        for line in file:
            line = line.strip().split(' ')

            if len(line) == dim+1:
                vector = [float(score) for score in line[1:]]
                word2vector[line[0]] = np.array(vector)
                word2idx[line[0]] = len(word2idx)

    return word2vector, word2idx, dim


def create_emb_weights(path_to_emb_file, vocab_data):
    """
    Create a matrix of embedding weights for initialization of the embedding layer

    :param path_to_emb_file: path to the embedding file
    :param vocab_data: list of vocab of training data
    :return: (1) the embedding weights, (2) a dict mapping token to its index
    """

    word2vector, word2idx, embedding_dim = load_pre_trained_embeddings(path_to_emb_file)

    # add vocab from data file
    for token in vocab_data:
        if token not in word2idx:
            word2idx[token] = len(word2idx)

    emb_weights = torch.zeros(len(word2idx), embedding_dim)

    for word, idx in word2idx.items():
        if word in word2vector:
            emb_weights[idx] = torch.from_numpy(word2vector[word]).float()
        else:
            emb_weights[idx] = torch.randn(embedding_dim)

    return emb_weights, word2idx


def encode_labels(corpus, labels_to_encode):
    """
    Encode labels to int for training purposes

    :param corpus: the corpus
    :param labels_to_encode: labels to encode
    :return: list of lists of encoded labels
    """

    # add PAD to label dicts
    corpus.label2idx['PAD'] = -1
    corpus.idx2label[-1] = 'PAD'

    encoded_labels = []
    for sent in labels_to_encode:
        sent_encoded = []
        for label in sent:
            sent_encoded.append(corpus.label2idx[label])
        encoded_labels.append(sent_encoded)

    return torch.LongTensor(encoded_labels)


def encode_emb_data(args, max_len, data_to_encode, word2idx, embedding_weights):
    """
    Encode data when input representation is embeddings

    :param args: the arguments
    :param max_len: max sentence length
    :param data_to_encode: data to encode
    :param word2idx: the dict to map a token to its corresponding index in embedding weight matrix
    :param embedding_weights: the embedding weight matrix
    :return: encoded data
    """

    # change tokens into word index
    # for CRF, remember to turn tokens into emb vector
    if 'lstm' in args.model_architecture:
        encoded_data = []
        for sent in data_to_encode:
            sent_encoded = []
            for token in sent:
                if token in word2idx:
                    sent_encoded.append(word2idx[token])
                else:
                    sent_encoded.append(word2idx['UNK'])
            encoded_data.append(sent_encoded)
        encoded_data = torch.LongTensor(encoded_data)

    else:
        # if using crf model
        encoded_data = torch.zeros(len(data_to_encode), max_len, embedding_weights.shape[1])
        for idx_sent, sent in enumerate(data_to_encode):
            for idx_tk, token in enumerate(sent):
                if token in word2idx:
                    encoded_data[idx_sent][idx_tk] = embedding_weights[word2idx[token]]
                else:
                    encoded_data[idx_sent][idx_tk] = embedding_weights[word2idx['UNK']]

    return encoded_data


def perform_final_eval(corpus, predictions):
    """
    Evaluation of model performance at entity level on three metrics, precision, recall, and F1 score

    :param corpus: the corpus
    :param predictions: list of lists of predictions
    :return:
    """

    # get entity tags
    entity_tags = list(corpus.label2idx.keys())
    entity_tags.remove('PAD')

    print('Running evaluation....')
    evaluation.eval_entity(corpus.test_data[1], predictions, entity_tags)


def run_experiment(args):
    args.data_path = os.path.join(os.getcwd(), args.data_path)
    data_object = 'NER_VLSP_2016_nn_' + args.input_type + '.dataset'
    corpus = load_data_object(args, data_object)

    if args.input_type == 'embeddings':
        args.path_to_emb_file = os.path.join(os.getcwd(), args.path_to_emb_file)

        # read embedding file --> embedding_weights, word2idx
        emb_obj = 'embedding_weights.obj'
        if os.path.exists(os.path.join(args.data_path, emb_obj)) and args.more_training:
            print('Load embedding weights')
            with open(os.path.join(args.data_path, emb_obj), 'rb') as file:
                embedding_weights, word2idx = torch.load(file)
        else:
            embedding_weights, word2idx = create_emb_weights(args.path_to_emb_file, corpus.vocab)

            # save for next time
            with open(os.path.join(args.data_path, emb_obj), 'wb') as file:
                torch.save([embedding_weights, word2idx], file)

        train_data = encode_emb_data(args, corpus.max_len, corpus.train_data[0], word2idx, embedding_weights)
        test_data = encode_emb_data(args, corpus.max_len, corpus.test_data[0], word2idx, embedding_weights)
        dim = embedding_weights.shape[1]
    else:
        train_data = corpus.train_data[0]
        test_data = corpus.test_data[0]
        dim = train_data.shape[2]
        embedding_weights = None

    # encoding labels for training
    train_labels = encode_labels(corpus, corpus.train_data[1])
    num_ner_tags = len(corpus.label2idx)

    if 'crf' in args.model_architecture:
        use_crf = True
    else:
        use_crf = False

    model_name = args.model_architecture + '_' + args.input_type + '.model'

    # create/load and run model
    if args.more_training and os.path.exists(os.path.join(args.data_path, model_name)):
        print('Load model for further training....')
        with open(os.path.join(args.data_path, model_name), 'rb') as file:
            model, optimizer, criterion = torch.load(file)
    else:
        model = get_model(args, use_crf, num_ner_tags, input_dim=dim, emb_weights=embedding_weights)
        # define optimizer & get loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = loss_fn

    print('Running experiment with', args.input_type)
    train(args, model, train_data, train_labels, optimizer, criterion, use_crf, model_name)

    print('Perform predictions on the model with best val acc')
    with open(os.path.join(args.data_path, model_name), 'rb') as file:
        model, _, _ = torch.load(file)
    predictions = make_predictions(model, test_data, corpus.test_data[1], corpus.idx2label, use_crf)
    perform_final_eval(corpus, predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch NER experiment on models versus features/embeddings')
    parser.add_argument('--data_path', type=str, default='data')
    # parser.add_argument('--path_to_emb_file', type=str, default='FastText_ner.vec')
    parser.add_argument('--path_to_emb_file', type=str, default='glove.vie.25d.txt')
    parser.add_argument('--no_misc', type=bool, default=True)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lstm_hidden_dim', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--val_batch_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--input_type', type=str, default='features', const='features', nargs='?',
                        choices=['features', 'embeddings', 'stackings'],
                        help='types of input: [features, embeddings, stackings]')
    parser.add_argument('--model_architecture', type=str, default='lstm-crf', const='lstm-crf', nargs='?',
                        choices=['crf', 'lstm', 'lstm-crf'],
                        help='type of architectures: [lstm, crf, lstm-crf]')
    parser.add_argument('--more_training', action='store_true',
                        help='continue training instead of creating new models')
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_experiment(args)
