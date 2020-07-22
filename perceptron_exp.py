import numpy as np
import pickle
import os
import argparse

import dataset
import evaluation


class Perceptron(object):
    """
    A perceptron whose job is to do binary classification with positive = 1 and negative = 0
    """

    def __init__(self, num_features, lr, random_state=1):
        self.num_features = num_features
        self.learning_rate = lr
        self.random_state = random_state

        self.bias, self.weights = self.weights_init()

    def weights_init(self):
        """
        Randomly initialize the weights and bias of the perceptron
        """
        generator = np.random.RandomState(self.random_state)

        bias = generator.normal(loc=0.0, scale=0.01, size=1)
        weights = generator.normal(loc=0.0, scale=0.01, size=self.num_features)

        return bias, weights

    def score(self, X):
        """
        Calculate score of the input

        :param X: an input vector to calculate the net input

        :return: the net input
        """
        return np.dot(self.weights, X) + self.bias

    def predict(self, X):
        """
        Predict class of the input vector (aka token)

        :param X: an input vector

        :return: the class label in binary value
        """
        return np.where(self.score(X) >= 0.0, 1, 0)

    def fit(self, X, y_true, y_pred):
        """
        Train the perceptron with an input vector

        :param X: an input vector
        :param y_true: the golden label
        :param y_pred: the predicted label

        :return: the perceptron
        """
        error = (y_true - y_pred) * self.learning_rate
        self.weights += error * X
        self.bias += error

        return self


class MulticlassPerceptron(object):
    """
    A multi-perceptron for NER classification
    """
    def __init__(self, num_classifiers, num_features, lr, num_epochs):
        self.num_classifiers = num_classifiers
        self.num_features = num_features
        self.num_epochs = num_epochs
        self.learning_rate = lr

        self.classifiers = self.init_classifiers(self.learning_rate)

    def init_classifiers(self, lr):
        """
        Create a list of perceptrons / classifiers equal to the number of labels

        :return: a list of  classifiers
        """
        classifiers = []
        for num in range(self.num_classifiers):
            classifiers.append(Perceptron(num_features=self.num_features, lr=lr))

        return classifiers

    def predict(self, X):
        """
        Predict class of the input vector (aka token)

        :param X: an input vector

        :return: the class label in binary value
        """
        scores = []  # to store scores predicted by all classifiers

        for classifier in self.classifiers:
            scores.append(classifier.score(X))

        # predict class labels based on scores given by each perceptron
        # winner (label = 1) is the one with highest score, otherwise label = 0
        scores = np.array(scores)
        return np.where(scores == scores.max(), 1, 0)

    def predict_final_label(self, X):
        """
        Predict class of the input vector (aka token)

        :param X: an input vector

        :return: the index of class label according to the mapping
        """
        # return the idx of non-zero value as int --> same as max value
        return int(np.argmax(self.predict(X)))

    def fit(self, X, y):
        """
        Train the multi-perceptron with an input vector

        :param X: list of input vectors for training
        :param y: list of golden labels for training

        :return: the multi-perceptron
        """

        for epoch in range(self.num_epochs):
            errors = 0

            for xi, y_true in zip(X, y):
                # currently, y_true is an integer reflecting the label index in the mapping
                # we need to convert y_true to suitable values for all classifiers (0 or 1)
                targets = np.zeros(self.num_classifiers)
                targets[y_true] = 1

                predictions = self.predict(xi)

                # update the classifiers accordingly if giving incorrect prediction
                for idx in range(self.num_classifiers):
                    if predictions[idx] != targets[idx]:
                        self.classifiers[idx].fit(xi, targets[idx], predictions[idx])
                        errors += 1

            print(epoch, errors)

        return self


def main():

    # read/load data
    parser = argparse.ArgumentParser('Perceptron for NER task')
    parser.add_argument('--path_to_data_dir', type=str, default='/Users/vanhoang/PycharmProjects/CL_Lab/data')
    parser.add_argument('--data_file', type=str, default='NER_VLSP_2016_perceptron.obj')
    parser.add_argument('--no_misc', type=bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=20)
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.path_to_data_dir, args.data_file)):
        print("Loading data files.....")
        with open(os.path.join(args.path_to_data_dir, args.data_file), 'rb') as file:
            data = pickle.load(file)
    else:
        print("Creating data file....")
        data = dataset.Dataset(no_misc=args.no_misc, path=args.path_to_data_dir,
                               model_arch='perceptron', input_type='perceptron', emb_file=None)

        # save data as object for next time
        file = open(os.path.join(args.path_to_data_dir, args.data_file), 'wb')
        pickle.dump(data, file)

    # encoding labels for training
    labels_matrix = [data.label2idx[label] for label in data.train_data[1]]

    # create multi perceptron classifier
    print('Start training.....')
    multi_classifier = MulticlassPerceptron(num_classifiers=len(data.label2idx), num_features=len(data.feature2idx),
                                            num_epochs=args.num_epochs, lr=args.learning_rate)
    multi_classifier.fit(data.train_data[0], labels_matrix)

    # get label predictions
    print('Making predictions....')
    predictions_train = []
    predictions_test = []

    for xi in data.train_data[0]:
        pred = multi_classifier.predict_final_label(xi)
        predictions_train.append(pred)

    for xi in data.test_data[0]:
        pred = multi_classifier.predict_final_label(xi)
        predictions_test.append(pred)

    train_labels_predicted = [data.idx2label[idx] for idx in predictions_train]
    test_labels_predicted = [data.idx2label[idx] for idx in predictions_test]

    # get entity tags
    entity_tags = data.label2idx.keys()

    # for evaluation, make it list of list
    print('for training')
    evaluation.eval_entity([data.train_data[1]], [train_labels_predicted], entity_tags)

    print('='*50)
    print('for testing')
    evaluation.eval_entity([data.test_data[1]], [test_labels_predicted], entity_tags)


main()
