# train.py
import io
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import dataset
import vdcnn

import tensorflow as tf
from sklearn import metrics
import config

def train(data_loader, model, optimizer, device):
     """
    This is the main training function that trains model
     for one epoch
     :param data_loader: torch dataloader
     :param model: model (vdcnn or lstm)
     :param optimizer: torch optimizer (adam, sgd, etc.)
     :param device: this can be "cuda" or "cpu"
     """

     # set model to training mode
     model.train()
     # go through batches of data in data loader
     for data in data_loader:
         # fetch review and target from the dict
         reviews = data["review"]
         targets = data["target"]
         # move the data to device that we want to use
         reviews = reviews.to(device, dtype=torch.long)
         targets = targets.to(device, dtype=torch.long)
         # clear the gradients
         optimizer.zero_grad()
         # make predictions from the model
         predictions = model(reviews)

         # calculate the loss
         loss = nn.CrossEntropyLoss()(
         predictions,
         targets
         )
         # compute gradient of loss w.r.t. all parameters of the model that are trainable
         loss.backward()
         # optimization step
         optimizer.step()

def evaluate(data_loader, model, device):
     # initialize empty lists to store predictions and targets
     final_predictions = []
     final_targets = []
     # put the model in eval mode
     model.eval()
     # disable gradient calculation
     with torch.no_grad():
         for data in data_loader:
             reviews = data["review"]
             targets = data["target"]
             reviews = reviews.to(device, dtype=torch.long)
             targets = targets.to(device, dtype=torch.float)
             # make predictions
             predictions = model(reviews)
             # move predictions and targets to list
             # move predictions and targets to cpu too
             predictions = predictions.cpu().numpy().tolist()
             targets = data["target"].cpu().numpy().tolist()
             final_predictions.extend(predictions)
             final_targets.extend(targets)
     # return final predictions and targets
     return final_predictions, final_targets

def create_embedding_matrix(word_index, embedding_dict):
    """
    This function creates the embedding matrix.
    :param word_index: a dictionary with word:index_value
    :param embedding_dict: a dictionary with word:embedding_vector
    :return: a numpy array with embedding vectors for all known words
    """
    # initialize matrix
    embedding_matrix = []
    # loop over all the words
    for word, i in word_index.items():
        # if word is found in pre-trained embeddings,
        # update the matrix. if the word is not found,
        # the vector is zeros
        if word in embedding_dict:
            embedding_matrix.append(embedding_dict[word])
        # return embedding matrix
    return np.array(embedding_matrix)


def tokenizing(df, char_index_dict):
    """
    Tokenizing the amino acids of the sequences in the dataset
    :param df: datframe containing the sequences
    :param char_index_dict: a dictionary with amino-acids:index value
    :return: df where the sequences are updated
    """
    final_string = []
    for sentence in df['sequence']:
        sequence = []
        for word in sentence:
            x = char_index_dict[word]
            sequence.append(str(x))
        final_string.append(sequence)
    df['clean_seq'] = final_string

    return df

def one_hot():
    '''
    One hot encoding of a sequence of index
    :return: one hot encoded embedding
    '''
    nb_classes = len(config.CHAR_INDEX_DICT)
    one_hot_embedding_mat = np.eye(nb_classes)
    one_hot_embedding = {}
    for key, value in enumerate(config.CHAR_INDEX_DICT):
        one_hot_embedding[value] = one_hot_embedding_mat[key]
    return one_hot_embedding

def run(df, fold):
    """
    Run training and validation for a given fold
    and dataset
    :param df: pandas dataframe with kfold column
    :param fold: current fold, int
    """
    # fetch training dataframe
    train_df = df[df.kfold != fold].reset_index(drop=True)
    # fetch validation dataframe
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    print("Fitting tokenizer")
    # convert training data to sequences
    # for example : "ag" gets converted to
    # [24, 27] where 24 is the index for a and 27 is the index for g
    xtrain = list(tokenizing(train_df,config.CHAR_INDEX_DICT)['clean_seq'])
    # similarly convert validation data to sequences
    xtest = list(tokenizing(valid_df,config.CHAR_INDEX_DICT)['clean_seq'])
    # zero pad the training sequences given the maximum length
    # this padding is done on left hand side
    # if sequence is > MAX_LEN, it is truncated on left hand side too
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(
        xtrain, maxlen=config.MAX_LEN
    )
    # zero pad the validation sequences
    xtest = tf.keras.preprocessing.sequence.pad_sequences(
        xtest, maxlen=config.MAX_LEN
    )
    # initialize dataset class for training
    train_dataset = dataset.PFAMDataset(
        reviews=xtrain,
        targets=train_df.family_id.values
    )
    # create torch dataloader for training
    # torch dataloader loads the data using dataset
    # class in batches specified by batch size
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=2
    )
    # initialize dataset class for validation
    valid_dataset = dataset.PFAMDataset(
        reviews=xtest,
        targets=valid_df.family_id.values
    )

    # create torch dataloader for validation
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )
    print("Loading embeddings")
    # load embeddings as shown previously
    embedding_dict = one_hot()
    embedding_matrix = create_embedding_matrix(
        config.CHAR_INDEX_DICT, embedding_dict
    )
    # create torch device, since we use gpu, we are using cuda
    print(torch.cuda.is_available())
    device = torch.device("cuda")
    # fetch our model
    # model = LSTM(embedding_matrix)
    model = vdcnn.VDCNN(embedding_matrix)
    # send model to device
    model.to(device)

    # initialize Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Training Model")
    # set best accuracy to zero
    best_accuracy = 0
    # set early stopping counter to zero
    early_stopping_counter = 0
    # train and validate for all epochs
    for epoch in range(config.EPOCHS):
        # train one epoch
        train(train_data_loader, model, optimizer, device)
        # validate
        outputs, targets = evaluate(
            valid_data_loader, model, device
        )
        outputs = np.array(outputs)
        # calculate accuracy
        accuracy = metrics.accuracy_score(targets, np.argmax(outputs,axis=1))
        precision = metrics.precision_score(targets, np.argmax(outputs,axis=1),average='macro')
        recall = metrics.recall_score(targets, np.argmax(outputs,axis=1),average='macro')

        print(
            f"FOLD:{fold}, Epoch: {epoch}, Accuracy Score = {accuracy}, Precision Score = {precision}, Recall Score = {recall}"
        )
        # simple early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        else:
            early_stopping_counter += 1
        if early_stopping_counter > 2:
            break
    return accuracy,precision, recall

if __name__ == "__main__":
     # load data
     df = pd.read_csv("/content/pfam_736_folds.csv")
     # train for all folds
     accuracy_0,precision_0, recall_0 = run(df, fold=0)
     accuracy_1,precision_1, recall_1 = run(df, fold=1)
     accuracy_2,precision_2, recall_2 = run(df, fold=2)
     accuracy_3,precision_3, recall_3 = run(df, fold=3)
     accuracy_4,precision_4, recall_4 = run(df, fold=4)
     print('Average accuracy : ' + str(np.mean(np.array([accuracy_0,accuracy_1,accuracy_2,accuracy_3,accuracy_4]))))
     print('Average precision : ' + str(np.mean(np.array([precision_0,precision_1,precision_2,precision_3,precision_4]))))
     print('Average recall : ' + str(np.mean(np.array([recall_0,recall_1,recall_2,recall_3,recall_4]))))
