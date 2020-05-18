import os
import string
import numpy as np


class Dataset():

    def __init__(self, path, sparse=True):
        """
        Note: currently, no implementation of sparse=False

        :param path: path to data dir
        :param sparse: if True, create a sparse representation of features in the dataset
        """
        self.path = path
        self.sparse = sparse

        self.feature2idx = self.init_feature_mapping()

        self.label2idx = {'O': 0}
        self.idx2label = {0: 'O'}

        self.train_data = self.make_files(os.path.join(self.path, 'NER2016-TrainingData-3-3-2017-txt'), get_features=True)
        self.test_data = self.make_files(os.path.join(self.path, 'TestData-16-9-2016'), get_features=False)

    def init_feature_mapping(self):

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
                            else:
                                if len(line.split('\t')) == 5:
                                    sent.append(line)

        # loop through sentence and create features list for each token
        data = []  # each data point is a features list for each token
        labels = []

        for sent in sentences:
            as_features = self.sent2features(sent, get_features)
            data.extend(as_features[0])
            labels.extend(as_features[1])

        # create matrices of data
        data_matrix = self.create_data_matrix(data)

        return [np.array(data_matrix), labels]

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

            if get_features:
                self.make_encoding(features)

            # add POS and chunk tags of current token
            token_as_features = [features[1].strip(), features[2].strip()]

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

    def create_data_matrix(self, data):
        """
        Create a feature matrix from information about features of each tokens

        :param data: input data with each data point being a list of that token's features

        :return: a sparse feature matrix
        """

        matrix = []

        # for each token, create an array of feature length, if the token contains a particular feature,
        # change the value of the index in the array corresponding to the feature encoding
        for token in data:
            array = np.zeros(len(self.feature2idx))

            for feature in token:
                array[self.feature2idx[feature]] = 1

            matrix.append(array)

        return matrix

