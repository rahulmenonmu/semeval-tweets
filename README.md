# Semantic Evaluation of Tweets
Semantic Evaluation of Tweets using Naive Bayes, SVM and LSTM models



This project makes use of the  "SemEval 2017 task 4" corpus and focuses on Subtask A i.e, classifying the overal sentiment of a tweet as positive, negative or neutral.

## Data Pre-Processing

The aim of pre processing the tweets is to remove as much of the data as possible that does not add value to the prediction ability of a model. Precisely, removing all occurrences in the text that do not add any value to the sentiment of the text. The following are the pre processing tasks done in this project.

1. URL removal
2. Removal of the RT keyword ( stands for Re-Tweet ).
3. Convert all words to lower case for standardisation and ease of comparison.
4. Remove all user mentions as they do not contribute to the sentiment.
5. Remove all hashtags that do not contribute to the sentiment.
6. Remove small words ( less than 3 characters ) since they are less likely to contribute to the sentiment. Most fo these will also be handled by stop word removal.
7. Remove all non alpha numeric characters
8. Lemmatise the corpus using the nltk WordNetLemmatizer
9. Remove stop words from the corpus

## Feature Extraction

The following methods were tried and models evaluated to help decide which algorithm is to be used to extract features from the corpus to use as the training data for the sentiment classiﬁers.

• CountVectorizer
• TﬁdfTransformer

The TﬁdfTransformer works by incorporating the Term frequency and the Inverse term frequency of the words in the sentence while creating a vector for the sentence. This helps in capturing the semantic information in the corpus as opposed to a bag of words approach where the relationship between words is not captured while creating the vectors. This diﬀerence makes TﬁdfTransformer a better feature extractor for NLP tasks. This is also conformed by the fact that the accuracies of the traditional models increased on the test set on moving from a CountVectorizer to the TﬁdfTransformer.

Also, the accuracy seemed to increase with the usage of unigrams, bigrams and trigrams with the help for the ngram_range = ( 1, 3 )

## Classifiers

### Naive Bayes
The Naive Bayes classiﬁer from the scikit learn library is trained with the extracted feature vectors. The X vectors, calculated from the ﬁnal feature extraction step is then used to ﬁt a Multinomial Naive Bayes model. The model gives a 60% accuracy on the test data and macroF1 score of 0.45.


### Support Vector Machine
The scikit learn library is used for the SVM too. sklearn.LinearSVC is used. The same training and testing input vectors created using tﬁdf extraction used for the Naive Bayes is used here as well. The SVM model performs better than the Naive Bayes with an accuracy on the test set of 62.38% and a macroF1 score of 0.550


### LSTM

The Long Short Term Memory networks are a special type of Recurrent Neural Networks that have the capability to learn and remember the order dependence in problems with sequential input. These are widely used in NLP for problems such as speech recognition, translation etc. 
The implemented LSTM classiﬁer makes use of the PyTorch and TorchText libraries from python. A pre trained word embedding with 100 dimensions ( glove.6b.100d ) is used as the embedding layer for the neural network. 

The text ( processed tweets) and the labels ( sentiment ) are represented using the data classes ‘Field’ and ‘LabelField’ from the TorchText library. The ﬁeld is created using the space English tokeniser. 

The cleaned train data and test data are loaded using the torchtext.data.TabularDataSet 
The input vocabulary is then created using the build_vocab function using the glove embeddings as the vectors, setting the max_size of the vocabulary to be 5000.

Then the BucketIterator module used to create the training data iterator in batches of 100. This is to be used for training the model. 
The model class is built to take in the dimensions of the input, output ,hidden and the embedding layers along with the number of layers and the dropout proportion.

2 linear neural networks as the fully connected hidden layers have been used. The activation function used is ReLU. The dropout is speciﬁed to prevent overﬁtting on the data and I have chose the dropout ratio to be 0.25. 

The 2 hidden layers have a dimension of 32, the input and output sizes having the dimensions of the corresponding built vocabularies. The embedding layer inevitably has the dimensions in the size of the glove vectors which is 100.

On training the LSTM with the above mentioned hyper parameters, the loss has been minimised to 0.43 in 20 epochs with the loss graph showing a steady decrease. The loss criterion used is cross entropy loss and the learning rate is setup by the scheduler with the Adams optimiser. 




