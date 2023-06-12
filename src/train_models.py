"""
This file is used to train three different models:
1. training three versions of bert-base-uncased on the three different 
data augmentation setups (datase A, B, and C).
2. training RoBERTa-large and DeBERTa-large with our best data augmentation
setup (dataset B).
"""

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.optimizers import Adam
from keras.metrics import binary_accuracy
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

tf.random.set_seed(1234)


def create_dataset(df_labels, df_text):
    # create dataset by combining labels and text 
    # into one single pandas Dataframe
    labels_dic = {}
    for index, row in df_labels.iterrows():
        values = []
        for k, v in row.items():
            if v == 1:
                values.append(k)
            labels_dic[row[0]] = values
    
    df_text['Labels'] = df_text['Argument ID'].map(labels_dic)

    text = preprocess_text(list(df_text['Premise']))
    labels = df_text['Labels']
    return text, labels, df_text


def preprocess_text(text):
    # Preprocessing text string by stripping trailing whitespace
    preproc_text = []
    for x in text:
        x = x.strip()
        preproc_text.append(x)
    return preproc_text


def train_model(lm, epoch, bs, lr, sl, X_train, Y_train, X_dev, Y_dev):
    print("Training model: {}\nWith parameters:\nLearn rate: {}, "
          "Batch size: {}\nEpochs: {}, Sequence length: {}"
          .format(lm, lr, bs, epoch, sl))

    # Selecting the correct tokenizer for the model, and loading the model
    tokenizer = AutoTokenizer.from_pretrained(lm, use_fast=False)
    model = TFAutoModelForSequenceClassification.from_pretrained(lm,
                                                                 num_labels=20,
                                                                 problem_type="multi_label_classification")

    # Tokenzing train and dev texts
    x_train_tok = tokenizer(X_train, padding=True, max_length=sl,
                             truncation=True, return_tensors="np").data
    x_dev_tok = tokenizer(X_dev, padding=True, max_length=sl,
                           truncation=True, return_tensors="np").data

    # Setting the loss and optimization function
    loss_function = BinaryCrossentropy(from_logits=True)
    optim = Adam(learning_rate=lr)

    # Implement early stopping
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor='val_f1_score', patience=2, restore_best_weights=True,
        mode="max")
    
    # Encoding the labels with sklearns MultiLabelBinarizer
    encoder = MultiLabelBinarizer()
    encoder.fit(Y_train)
    y_train_enc = encoder.fit_transform(Y_train)
    y_dev_enc = encoder.fit_transform(Y_dev)
    
    # Compiling the model and training it with the given parameter settings
    model.compile(loss=loss_function, optimizer=optim, 
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5), 
             tfa.metrics.F1Score(average='macro', threshold=0.5, num_classes=20)])
    model.fit(x_train_tok, y_train_enc, verbose=1, epochs=epoch,
              batch_size=bs, validation_data=(x_dev_tok, y_dev_enc),
              callbacks=[early_stopper])
    return model


def test_model(lm, epoch, bs, lr, sl, model, X_test, Y_test, labels):
    print("Testing model: {}\nWith parameters:\nLearn rate: {}, "
        "Batch size: {}\nEpochs: {}, Sequence length: {}"
        .format(lm, lr, bs, epoch, sl))

    # Selecting the correct tokenizer for the model.
    tokenizer = AutoTokenizer.from_pretrained(lm, use_fast=False)
    tokens_test = tokenizer(X_test, padding=True, max_length=sl,
                            truncation=True, return_tensors="np").data

    # Encoding the labels with sklearns MultiLabelBinarizer 
    encoder = MultiLabelBinarizer()
    encoder.fit(labels)
    y_test = encoder.fit_transform(Y_test)

    # Get predictions and convert to numerical
    pred = model.predict(tokens_test)["logits"]
    prob = tf.nn.sigmoid(pred)
    predictions = np.zeros(prob.shape)
    predictions[np.where(prob >= 0.5)] = 1

    # Printing clasification report
    print(classification_report(y_test, predictions, zero_division=True, 
                                digits=4, target_names=encoder.classes_))


# labels 
train_labels = pd.read_table('data/labels-training.tsv')
dev_labels = pd.read_table('data/labels-validation.tsv')
test_labels = pd.read_table('data/labels-test.tsv')

# Original data
df_train_A = pd.read_table('data/arguments-training.tsv')
df_dev_A = pd.read_table('data/arguments-validation.tsv')
df_test_A = pd.read_table('data/arguments-test.tsv')

# Paraphrased data
df_train_B = pd.read_csv('data/B_training.csv')
df_dev_B = pd.read_csv('data/B_validation.csv')
df_test_B = pd.read_csv('data/B_test.csv')

# Original + paraphrased data
df_train_C = pd.read_csv('data/C_training.csv')
df_dev_C = pd.read_csv('data/C_validation.csv')
df_test_C = pd.read_csv('data/C_test.csv')

# Original train, dev, and test text and labels
x_train_A, y_train_A, df_A1 = create_dataset(train_labels, df_train_A)
x_dev_A, y_dev_A, df_A2 = create_dataset(dev_labels, df_dev_A)
x_test_A, y_test_A, df_A3 = create_dataset(test_labels, df_test_A)

# Paraphrased train, dev, and test text and labels
x_train_B, y_train_B, df_B1 = create_dataset(train_labels, df_train_B)
x_dev_B, y_dev_B, df_B2 = create_dataset(dev_labels, df_dev_B)
x_test_B, y_test_B, df_B3 = create_dataset(test_labels, df_test_B)

# Original + paraphrased train, dev, and test text and labels
x_train_C, y_train_C, df_C1 = create_dataset(train_labels, df_train_C)
x_dev_C, y_dev_C, df_C2 = create_dataset(dev_labels, df_dev_C)
x_test_C, y_test_C, df_C3 = create_dataset(test_labels, df_test_C)



print('Testing best data augmentation setup')
model_A = train_model('bert-base-uncased', 20, 8, 2e-5, 160, x_train_A, y_train_A, x_dev_A, y_dev_A)
test_model('bert-base-uncased', 20, 8, 2e-5, 160, model_A, x_test_A, y_test_A, y_train_A)

model_B = train_model('bert-base-uncased', 20, 8, 2e-5, 120, x_train_B, y_train_B, x_dev_B, y_dev_B)
test_model('bert-base-uncased', 20, 8, 2e-5, 120, model_B, x_test_B, y_test_B, y_train_A)
test_model('bert-base-uncased', 20, 8, 2e-5, 120, model_B, x_test_A, y_test_A, y_train_A)

model_C = train_model('bert-base-uncased', 20, 8, 2e-5, 160, x_train_C, y_train_C, x_dev_C, y_dev_C)
test_model('bert-base-uncased', 20, 8, 2e-5, 160, model_C, x_test_C, y_test_C, y_train_A)
test_model('bert-base-uncased', 20, 8, 2e-5, 160, model_C, x_test_A, y_test_A, y_train_A)


print('Training RoBERTa-large with best data setup')
ROBERTA = train_model('roberta-large', 20, 8, 1e-5, 120, x_train_B, y_train_B, x_dev_B, y_dev_B)
test_model('roberta-large', 20, 8, 1e-5, 120, ROBERTA, x_test_B, y_test_B, y_train_A)
test_model('roberta-large', 20, 8, 1e-5, 120, ROBERTA, x_test_A, y_test_A, y_train_A)
ROBERTA.save_weights('LTP_ROBERTA_weights.h5')


print('Training DeBERTa-large with best data setup')
DEBERTA = train_model('microsoft/deberta-large', 20, 8, 1e-5, 120, x_train_B, y_train_B, x_dev_B, y_dev_B)
test_model('microsoft/deberta-large', 20, 8, 1e-5, 120, DEBERTA, x_test_B, y_test_B, y_train_A)
test_model('microsoft/deberta-large', 20, 8, 1e-5, 120, DEBERTA, x_test_A, y_test_A, y_train_A)
DEBERTA.save_weights('LTP_DEBERTA_weights.h5')