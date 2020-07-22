import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

from collections import Counter
import numpy as np
import time, os


class LSTM_NER(nn.Module):
    def __init__(self, args, use_crf, num_ner_tags, input_dim, emb_weights):
        """
        Define the neural network LSTM here. It consists of:
            - an embedding layer
            - a LSTM layer
            - a linear layer to map output of lstm layer to probability distributions over 7 NER tags

        :param args: contains info about embedding_dim, vocab_size, hidden_dim, output_dim
        :param use_crf: True if the output will be passed later to a crf layer
        :param num_ner_tags: the number of output tags
        :param input_dim: the dimension of an input representation
        :param emb_weights: the embedding weights to initialize the embedding layer if used, None if not
        """
        super(LSTM_NER, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = args.lstm_hidden_dim
        self.output_dim = num_ner_tags

        # if True, use pre-trained embeddings
        # if False, use handcrafted features / stacking
        self.use_pre_trained_emb = (args.input_type == 'embeddings')
        if self.use_pre_trained_emb:
            self.embedding_vocab = emb_weights.shape[0]  # vocab of the pre-trained embeddings
            self.embedding_weights = emb_weights

        self.use_crf = use_crf

        # embedding layer if any
        if self.use_pre_trained_emb:
            self.embedding = nn.Embedding(num_embeddings=self.embedding_vocab, embedding_dim=self.input_dim)

            # load pre-trained embeddings and freeze it
            if not args.more_training:
                self.embedding.from_pretrained(self.embedding_weights, freeze=True)

            # lstm layer
            self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, batch_first=True)
        else:
            # if using handcrafted features or stackings, just put the inputs directly to the lstm layer
            self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, batch_first=True)

        # linear layer to transform lstm output to ner tags
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim)

    def forward(self, data):

        if self.use_pre_trained_emb:
            data_emb = self.embedding(data)  # dim: (batch_size x batch_max_len x embedding_dim)
        else:
            data_emb = data

        # run lstm cell
        output, _ = self.lstm(data_emb)  # dim: (batch_size x batch_max_len x lstm_hidden_dim)

        # apply linear layer to get ner tags
        # dim: (batch_size * batch_max_len x num_tags)
        predictions = self.fc(output.contiguous().view(output.size(0)*output.size(1), output.size(2)))

        # if using crf later, turn it back to 3d
        # dim (batch_size * batch_max_len x num_tags)
        if self.use_crf:
            predictions = predictions.view(output.size(0), output.size(1), predictions.size(1))

        return predictions


class LSTM_CRF_NER(nn.Module):
    def __init__(self, args, use_crf, num_ner_tags, input_dim, emb_weights):
        super(LSTM_CRF_NER, self).__init__()

        self.input_dim = input_dim
        self.output_dim = num_ner_tags
        self.use_crf = use_crf

        self.encoder = LSTM_NER(args, use_crf=self.use_crf, num_ner_tags=self.output_dim,
                                input_dim=self.input_dim, emb_weights=emb_weights)
        self.crf = CRF(num_tags=self.output_dim, batch_first=True)

    def forward(self, data, labels, mask, reduction):

        lstm_output = self.encoder(data)

        crf_out_loss = self.crf(lstm_output, labels, mask, reduction=reduction)

        return -crf_out_loss

    def predict(self, data):

        lstm_output = self.encoder(data)
        predictions = self.crf.decode(lstm_output)

        return predictions


class CRF_NER(nn.Module):
    def __init__(self, num_ner_tags, input_dim):
        super(CRF_NER, self).__init__()

        self.input_dim = input_dim
        self.output_dim = num_ner_tags

        # input dim = feature dim / emb dim
        # using a Linear layer to convert the feature matrix to suitable dim for crf layer
        # (batch_size x batch_max_len x input_dim) --> (batch_size x batch_max_len x num_tags)
        self.embedding = nn.Linear(in_features=self.input_dim, out_features=self.output_dim)
        self.crf = CRF(num_tags=self.output_dim, batch_first=True)

    def forward(self, data, labels, mask, reduction):

        # first reshape the data dim before using linear layer
        # (batch_size x batch_max_len x input_dim) --> (batch_size*batch_max_len x input_dim)
        data_in = data.contiguous().view(data.size(0)*data.size(1), data.size(2))

        crf_in = self.embedding(data_in)  # dim: (batch_size*batch_max_len x num_tags)

        # convert back to 3d tensor before using crf layer
        # --> dim: (batch_size, batch_max_len, num_tags)
        crf_in = crf_in.view(data.size(0), data.size(1), crf_in.size(1))

        crf_out_loss = self.crf(crf_in, labels, mask, reduction=reduction)

        return -crf_out_loss

    def predict(self, data):

        data_in = data.contiguous().view(data.size(0) * data.size(1), data.size(2))
        crf_in = self.embedding(data_in)
        crf_in = crf_in.view(data.size(0), data.size(1), crf_in.size(1))

        predictions = self.crf.decode(crf_in)

        return predictions


def get_batch(batch_size, batch_idx):
    """
    Get the start and end indexes of the data points of the current batch

    :param batch_size: size of the batch
    :param batch_idx: the current batch to get start and end indexes
    :return: start idx, end idx
    """

    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size

    return start_idx, end_idx


def loss_fn(outputs, labels, mask):
    """
    A custom loss function which do NOT take into account of the losses from PADDING tokens

    :param outputs: the outputs from a LSTM network
    :param labels: the gold labels
    :param mask: list of list of mask tokens (e.g. [False, False, False, True...])
    :return: the loss over all outputs
    """

    # get the number of non-PAD tokens
    num_tokens = int(torch.sum(mask))

    # calculate log softmax over all the tags
    outputs = F.log_softmax(outputs, dim=1)

    # flatten labels shape to dim (batch_size * seq_len)
    labels = labels.view(-1)

    # convert negative PAD idx (-1) into a positive number as indexing with negative values is not supported
    labels = labels % outputs.shape[1]

    # pick the values corresponding to labels
    outputs = outputs[range(outputs.shape[0]), labels] * mask

    # calculate and return cross entropy loss over all non PAD tokens
    return -torch.sum(outputs)/num_tokens


def get_mask_tokens(labels):
    """
    Generate list of list of mask tokens from a label batch
    e.g. [[False, False, False, True, True...], [False, False, True...], ....]

    :param labels: list of list of labels for conversion
    :return: list of list of mask tokens
    """

    # reshape labels from dim (batch_size, seq_len) to dim (batch_size*seq_len)
    labels = labels.view(-1)

    # mask out PAD tokens
    # 1.0 for non-PAD tokens, 0.0 for PAD tokens
    # e.g. tensor([[1., 1., 1., 1., 0., 0.],
    #         [1., 1., 1., 0., 0., 0.]])
    mask = (labels >= 0).float()

    return mask


def get_val_info(val_labels):
    """
    Print out entity distribution in validation set

    :param val_labels: labels in validaion set
    :return:
    """

    label_counter = Counter()

    for sent in val_labels.tolist():
        for label_idx in sent:
            label_counter.update([str(label_idx)])

    print('val labels info')
    print(label_counter)


def train(args, model, train_data, train_labels, optimizer, criterion, use_crf, model_name):

    print('Start training....')
    start_training_time = time.time()

    # divide data into train and val sets
    # val set containing the last 200 batches of size val batch from train set = 2k sentences
    val_size = args.val_batch_size * 200
    val_data = train_data[-val_size:]
    val_labels = train_labels[-val_size:]

    # to get NER distribution on validation set
    # get_val_info(val_labels)

    train_data = train_data[:-val_size]
    train_labels = train_labels[:-val_size]

    # calculate number of batches for the train set
    num_batches = train_data.shape[0] // args.batch_size

    best_val_acc = None

    for epoch in range(args.num_epochs):

        print('\tTraining epoch', epoch+1, '.....')
        start_epoch_time = time.time()

        # turn on training mode
        model.train()

        total_loss = 0.  # per epoch

        for batch_idx in range(num_batches):

            start_idx, end_idx = get_batch(batch_size=args.batch_size, batch_idx=batch_idx)

            # get the train data for the current batch
            data_batch = train_data[start_idx:end_idx]
            labels_batch = train_labels[start_idx:end_idx]

            # get mask tokens
            mask_batch = get_mask_tokens(labels_batch)

            # compute model output and loss
            if use_crf:
                # make sure to return the loss averaged over all (non-PAD) tokens
                mask_batch = mask_batch.view(labels_batch.shape[0], labels_batch.shape[1]).to(dtype=torch.uint8)
                loss = model(data_batch, labels_batch, mask=mask_batch, reduction='token_mean')
            else:
                output = model(data_batch)
                loss = criterion(output, labels_batch, mask_batch)

            total_loss += loss.item()

            # clear existing gradients
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()  # update the weights accordingly

        # perform evaluation on validation set
        val_loss, val_acc = evaluate(args, model, val_data, val_labels, criterion, use_crf)

        # print train info
        print('\tEpoch {} | training time {:5.2f} minutes | train loss: {:5.2f} | val loss: {:5.2f} | val acc: {:5.7f}'.
              format(epoch+1, (time.time() - start_epoch_time)/60, total_loss, val_loss, val_acc*100))

        # save the model if it's the best val acc so far
        if not best_val_acc or val_acc >= best_val_acc:
            with open(os.path.join(args.data_path, model_name), 'wb') as file:
                torch.save([model, optimizer, criterion], file)
            best_val_acc = val_acc
        else:
            # if no improvement on val acc
            # retrieve the best val acc model and reduce learning rate
            with open(os.path.join(args.data_path, model_name), 'rb') as file:
                model, optimizer, criterion = torch.load(file)
            optimizer.param_groups[0]['lr'] /= 2.0

    print('=' * 50)
    print('Training time in minutes:', (time.time() - start_training_time) / 60)
    print('=' * 50)


def evaluate(args, model, val_data, val_labels, criterion, use_crf):

    # turn on evaluation mode
    model.eval()

    total_loss = 0.
    total_labels = 0.
    correct_labels = 0.
    num_batches = val_data.shape[0] // args.val_batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx, end_idx = get_batch(batch_size=args.val_batch_size, batch_idx=batch_idx)

            data_batch = val_data[start_idx:end_idx]
            label_batch = val_labels[start_idx:end_idx]

            mask_batch = get_mask_tokens(label_batch)

            if use_crf:
                mask_batch_crf = mask_batch.view(label_batch.size(0), label_batch.size(1)).to(dtype=torch.uint8)
                loss_batch = model(data_batch, label_batch, mask_batch_crf, reduction='token_mean')

                output_batch = np.array(model.predict(data_batch))
                output_batch = output_batch.reshape(output_batch.shape[0]*output_batch.shape[1])
            else:
                output_batch = model(data_batch)
                loss_batch = criterion(output_batch, label_batch, mask_batch)

                output_batch = F.log_softmax(output_batch, dim=1)
                output_batch = output_batch.data.cpu().numpy()
                output_batch = np.argmax(output_batch, axis=1)  # class predicted for each token

            label_batch = label_batch.view(-1).data.cpu().numpy()  # turn it to flat vector
            mask_batch = mask_batch.data.cpu().numpy()

            total_labels += np.sum(mask_batch)
            correct_labels += np.sum(output_batch == label_batch)

            total_loss += loss_batch.item()

    # compute accuracy at token level
    accuracy = correct_labels / total_labels

    return total_loss, accuracy


def make_predictions_lstm(model, test_data, test_labels, idx2label):

    # set model on eval mode
    model.eval()

    outputs_decoded = []
    # loop over each sentence in test set and make predictions
    for idx in range(test_data.shape[0]):

        output = model(test_data[idx:idx+1])  # dim: (batch_size * batch_max_len x num_tags)

        # convert to log softmax
        output = F.log_softmax(output, dim=1)

        # extract data from torch, move to cpu, convert to numpy arrays
        output = output.data.cpu().numpy()

        # find the class predicted for each token
        output = np.argmax(output, axis=1)

        output_decoded = []

        # append to outputs_decoded but make sure to check its length with the corresponding sentence in test labels
        for label_idx in output[:len(test_labels[idx])]:
            if label_idx in idx2label:
                output_decoded.append(idx2label[label_idx])
            else:
                output_decoded.append('O')

        outputs_decoded.append(output_decoded)

    return outputs_decoded


def make_predictions_crf(model, test_data, test_labels, idx2label):

    # set model on eval mode
    model.eval()

    outputs_decoded = []
    # loop over each sentence in test set and make predictions
    for idx in range(test_data.shape[0]):
        # List of list with the best tag sequence (int) for each batch
        # but since we feed one sentence at a time, only get the first list in output
        output = model.predict(test_data[idx:idx+1])[0]

        output_decoded = []
        for label_idx in output[:len(test_labels[idx])]:
            if label_idx in idx2label:
                output_decoded.append(idx2label[label_idx])
            else:
                output_decoded.append('O')
        outputs_decoded.append(output_decoded)

    return outputs_decoded


def make_predictions(model, test_data, test_labels, idx2label, use_crf):

    if use_crf:
        return make_predictions_crf(model, test_data, test_labels, idx2label)
    else:
        return make_predictions_lstm(model, test_data, test_labels, idx2label)
