# Which Is Better For Vietnamese Named Entity Recognition: Features or Embeddings?

With the move from traditional machine learning to neural network models, the emphasis on feature engineering has been diminished accordingly. It is often claimed that the neural networks can learn directly from the dataset the information needed for the task at hand.
In this paper, we seek to understand the power of the neural networks by exploring whether handcrafted features still have a role in the area of deep learning through Named Entity Recognition task for Vietnamese language. The results show that, though the model architecture is indeed crucial, the word embeddings are not necessarily superior to handcrafted features.

-----------------------------

### Dependencies
You need to install [Pytorch](https://pytorch.org/) library. 

### Data and pre-trained embeddings

For the NER dataset, please see [VLSP web page](https://vlsp.org.vn/resources-vlsp2016).

For pre-trained embeddings, you can download fastTex at [here](https://github.com/vietnlp/etnlp) and GloVe at [here](https://github.com/minhpqn/vietner).

If you wish to see data distribution on training and test set, type:
```
python data_stats.py
```

### Models

To run experiments on **_the perceptron_**, type:
```
python perceptron_exp.py
```

The perceptron accepts these arguments:
```
--path_to_data_dir  path to data directory
--data_file         name of data file/corpus to load or create
--no_misc           True if do no recognition for MISC entity, default=True
--learning_rate     learning rate, default=0.2
--num_epochs        number of training epochs, default=20
```

To run experiments on **_neural network models_** (crf, lstm, lstm-crf), type:
```
python nn_experiments.py
```

It accepts these arguments:
```
--data_path           path to data directory
--path_to_emb_file    path to embedding file
--no_misc             True if do no recognition for MISC entity, default=True
--num_epochs          number of training epochs, default=20
--lstm_hidden_dim     size of hidden layer, default=100
--batch_size          size of training batch, default=30
--val_batch_size      size of validation batch, default=10
--learning_rate       learning rate, default=0.001
--input_type          type of input representation, choices=['features', 'embeddings', 'stackings'], default='features'
--model_architecture  the model for NER task, choices=['crf', 'lstm', 'lstm-crf'], default='lstm-crf'
--more_training       continue training for more epoches
```

For example, assuming you have the data available, if you wish to perform an experiment with stacking (features + embeddings) representation on LSTM, type:
```
python nn_experiments.py --input_type stackings --model_architecture lstm
```

To continue training, simply type:
```
python nn_experiments.py --more_training
```
You can also specify the number of training epochs with argument --num_epochs.

