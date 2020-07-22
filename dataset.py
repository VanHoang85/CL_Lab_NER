import os
import string
import numpy as np
import torch

from nn_experiments import load_pre_trained_embeddings


class Dataset:

    def __init__(self, path, no_misc, model_arch, input_type, emb_file):
        """
        Create Dataset object for train and test data

        :param path: path to data dir
        :param no_misc: if True, convert MISC entities into O labels
        :param model_arch: adjust the data representation depending on model architectures: 'perceptron' and 'nn'
        :param input_type: input representation for the data matrix (perceptron, features, embeddings, stackings)
        :param emb_file: the embedding file used for embeddings and stackings
        """
        self.path = path
        self.no_misc = no_misc
        self.model_architecture = model_arch
        self.input_type = input_type
        self.emb_file = emb_file

        if self.input_type != 'embeddings':
            self.feature2idx = self.init_feature_mapping()

        self.vocab = set()

        self.max_len = 0

        self.label2idx = {'O': 0}
        self.idx2label = {0: 'O'}

        self.train_data = self.make_files(os.path.join(self.path, 'NER2016-TrainingData-3-3-2017-txt'), get_features=True)
        self.test_data = self.make_files(os.path.join(self.path, 'TestData-16-9-2016'), get_features=False)

        # self.train_data = self.make_files(os.path.join(self.path, 'train_small'), get_features=True)
        # self.test_data = self.make_files(os.path.join(self.path, 'test_small'), get_features=False)

    def init_feature_mapping(self):
        """
        Initialize the feature dictionary

        :return: the feature2idx dictionary with some initial features
        """

        mapping = {}

        # for feature if token is capitalized or not
        mapping['init_capital'] = len(mapping)
        mapping['not_init_capital'] = len(mapping)

        # for feature if all char in token is capitalized or not
        mapping['all_cap'] = len(mapping)
        mapping['not_all_cap'] = len(mapping)

        # for feature if the token is special character or not (punctuations or not)
        mapping['special_char'] = len(mapping)
        mapping['not_special_char'] = len(mapping)

        # for feature if the token is digit or not
        mapping['digit'] = len(mapping)
        mapping['not_digit'] = len(mapping)

        return mapping

    def make_files(self, path_to_dir, get_features):
        """
        Create data accordingly to the data files (train/test)

        :param path_to_dir: path to data dir
        :param get_features: If True, while looping through the files, get features (e.g. POS tags) from the files
                             and put them into the mappings (from features to indexes and vice verse)

        :return: a feature matrix representing input data
                 a list of labels for all words (rows) in the matrix
        """

        files = [file for file in os.listdir(path_to_dir) if os.path.isfile(os.path.join(path_to_dir, file))]

        sentences = []

        for file in files:
            with open(os.path.join(path_to_dir, file), 'r', encoding='utf-8') as f:
                for line in f:
                    sent = []
                    if line.startswith('<s>'):
                        is_sent = True
                        while is_sent:
                            line = f.readline().strip()

                            if line.startswith('</s>'):
                                is_sent = False
                                sentences.append(sent)

                                if len(sent) > self.max_len and get_features:
                                    self.max_len = len(sent)

                            else:
                                if len(line.split('\t')) == 5:
                                    sent.append(line)

        # loop through sentence and create features list for each token
        # if perceptron architecture, rows corresponding to tokens and columns to each feature in binary format
        # if neural network architecture, rows corresponding to sentences and columns to tokens in the sentences
        # if using handcrafted features, each token is represented as a list of features in binary format
        # Otherwise the returned data will just be the tokens themselves
        data = []
        labels = []

        # loop through each sentence and get list of features of each token in the sentence
        for sent in sentences:
            as_features = self.sent2features(sent, get_features)

            if self.model_architecture == 'perceptron':
                data.extend(as_features[0])
                labels.extend(as_features[1])
            else:
                # if neural network architecture
                data.append(as_features[0])
                labels.append(as_features[1])

        # create matrices of data
        if self.model_architecture == 'perceptron':
            data_matrix = self.create_token_matrix(data)

            return data_matrix, labels
        else:
            # if neural network architecture
            labels_matrix = self.create_label_matrix(labels)
            data_matrix = self.create_sent_matrix(data)

            return data_matrix, labels_matrix

    def sent2features(self, sent, get_features):
        """
        From token as column information, creature list of features

        Template for feature list of each token [ POS_tag, chunking_tag, init_capital/no_init_capital, etc. ]

        :param sent: the sentence
        :param get_features: if True, make the features and labels encoding

        :return: list of features of all tokens in the sentence
        """
        sent_as_features = []
        labels = []

        for idx, token in enumerate(sent):
            features = token.strip().split('\t')

            # if no training on MISC entity, just ignore it --> convert it to O (out of token)
            if self.no_misc:
                if 'MISC' in features[3]:
                    features[3] = 'O'

            if get_features:
                self.make_encoding(features)
                self.vocab.add(features[0].strip())  # only add training vocab

            # add the token
            token_as_features = list([features[0].strip()])

            # add POS and chunk tags of current token
            token_as_features.append(features[1].strip())
            token_as_features.append(features[2].strip())

            # add POS and chunk tags of previous token if any
            if idx > 0:
                pre_features = sent[idx-1].strip().split('\t')
                token_as_features.append('pre-' + pre_features[1].strip())
                token_as_features.append('pre-' + pre_features[2].strip())

            # check and add features
            token_as_features.append(self.is_init_cap(token))
            token_as_features.append(self.is_all_cap(token))
            token_as_features.append(self.is_punc(token))
            token_as_features.append(self.is_digit(token))

            sent_as_features.append(token_as_features)
            labels.append(features[3].strip())

        return sent_as_features, labels

    def make_encoding(self, features):
        """
        Add feature tags accordingly to the mappings

        :param features: the list of features
        """

        if self.input_type != 'embeddings':
            if features[1] not in self.feature2idx:
                self.feature2idx[features[1].strip()] = len(self.feature2idx)
                self.feature2idx['pre-' + features[1].strip()] = len(self.feature2idx)

            if features[2] not in self.feature2idx:
                self.feature2idx[features[2].strip()] = len(self.feature2idx)
                self.feature2idx['pre-' + features[2].strip()] = len(self.feature2idx)

        if features[3] not in self.label2idx:
            self.label2idx[features[3].strip()] = len(self.label2idx)
            self.idx2label[len(self.idx2label)] = features[3].strip()

    def is_init_cap(self, token):

        if token[0].isupper():
            return 'init_capital'
        else:
            return 'not_init_capital'

    def is_all_cap(self, token):

        if token.isupper():
            return 'all_cap'
        else:
            return 'not_all_cap'

    def is_punc(self, token):

        if token in string.punctuation:
            return 'special_char'
        else:
            return 'not_special_char'

    def is_digit(self, token):

        if token.isdigit():
            return 'digit'
        else:
            return 'not_digit'

    def create_label_matrix(self, labels):
        """
        Create label matrix for all mode architectures

        :param labels: the labels
        :return: a label matrix
        """

        matrix = []
        for sent in labels:
            vector = ['PAD'] * self.max_len
            vector[:len(sent)] = sent
            matrix.append(vector)

        return matrix

    def create_token_matrix(self, data):
        """
        Create a feature matrix for perceptron architecture

        :param data: input data with each data point being a list of that token's features
        :return: a sparse feature matrix
        """

        matrix = []

        for token in data:
            vector = self.create_feature_vector(token)
            matrix.append(vector)

        return np.array(matrix)

    def create_sent_matrix(self, data):
        """
        Create data matrix for neural network architecture depending on data representation type

        :param data: data for representation
        :return: the data matrix corresponding to data representation type
        """

        if self.input_type == 'features':
            return self.create_sent_feature_matrix(data)
        elif self.input_type == 'stackings':
            return self.create_sent_stacking_matrix(data)
        else:
            # embedding
            return self.create_sent_token_matrix(data)

    def create_sent_token_matrix(self, data):
        """
        Create 2d data matrix with dim (data_size, sent_length)

        :param data: data for representation
        :return: the data matrix
        """

        matrix = []
        for sent in data:
            vector = ['PAD'] * self.max_len

            for idx, token in enumerate(sent):
                if idx < len(vector):
                    vector[idx] = token[0]

            matrix.append(vector)

        return matrix

    def create_sent_stacking_matrix(self, data):
        """
        Create 3d data matrix with dim (data_size, max_sent_length, feature + embedding size)

        :param data: data for representation
        :return: the data matrix
        """

        word2vector, _, dim = load_pre_trained_embeddings(self.emb_file)

        matrix = torch.zeros(len(data), self.max_len, dim + len(self.feature2idx))

        for idx_sent, sent in enumerate(data):
            for idx_tk, token in enumerate(sent):
                if idx_tk < self.max_len:

                    if token[0] in word2vector:
                        embedding = word2vector[token[0]]
                    else:
                        embedding = np.random.normal(loc=0.0, scale=1.0, size=(dim,))
                        word2vector[token[0]] = embedding

                    vector = np.array(self.create_feature_vector(token))
                    vector = np.concatenate([embedding, vector])

                    matrix[idx_sent][idx_tk] = torch.LongTensor(vector)

        return matrix

    def create_sent_feature_matrix(self, data):
        """
        Create 3d data matrix with dim (data_size, max_sent_length, feature_size)

        :param data: data for representation
        :return: the data matrix
        """

        # dim: (data_size, max_len, feature_dim)
        matrix = torch.zeros(len(data), self.max_len, len(self.feature2idx))

        for idx_sent, sent in enumerate(data):
            for idx_tk, token in enumerate(sent):
                if idx_tk < self.max_len:
                    vector = self.create_feature_vector(token)
                    matrix[idx_sent][idx_tk] = torch.LongTensor(vector)

        return matrix

    def create_feature_vector(self, token):
        """
        Create a sparse vector representation for an input token from its features

        :param token: the token
        :return: a vector representation
        """

        # for each token, create an array of feature length,
        vector = [0] * len(self.feature2idx)

        # if the token contains a particular feature,
        # change the value of the index in the array corresponding to the feature encoding
        # the first feature in the token is the token itself, ignore it
        for feature in token[1:]:
            if feature in self.feature2idx:
                vector[self.feature2idx[feature]] = 1

        return vector
