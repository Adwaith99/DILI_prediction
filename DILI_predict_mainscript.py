# This script can be used to predcit the DILI status of a molecule(from its smiles) using our
# trained model. 

# The requirements to run this script are
# - tensorflow==2.2.0
# - pandas==1.1.3
# - numpy==1.19.2
# - padelpy=0.1.7

# The script takes the follwing as input:
# - a model directory(in the Savedmodel format of tensorflow)(The link is addded in the input form) - This model was trained using GPU in CDAC virtual tool room
# - a .smi file with the list of canonical smiles on which you wish to predict DILI status

import tensorflow as tf
import pandas as pd
import numpy as np
import csv
from urllib.request import urlopen
from padelpy import padeldescriptor
import pickle
from tensorflow.keras.metrics import AUC
from tensorflow.keras import backend as K

#   This function processes the given smiles to a numpy array of the inputs to be fed to the model
def process_smiles_input(file_location):

    padeldescriptor(mol_dir=file_location, d_file="padel_out.csv", fingerprints=True, d_2d=True, d_3d=False, threads=-1)
    
    with open(file_location) as file:
        smiles = [line for line in file]

    padel_out = pd.read_csv("padel_out.csv", index_col=0).to_numpy()
    print("Total molecules to predict : {}".format(len(padel_out)))

    descriptors = padel_out[:, :1444]
    fingerprints = padel_out[:, 1444:]
    
    input = np.concatenate((fingerprints, descriptors), axis=1)

    return [smiles, input]

#   This function return the prediction and confidence scores of individual molecules
def dili_predict(model, predict_input):
    predict_input = np.reshape(predict_input, (1, 2325))
    prediction = model.predict(predict_input)
    if np.argmax(prediction[0, :]) == 0:
        label = "positive"
        confidence = (prediction[0, 0]/([prediction.sum()]))*100
    elif np.argmax(prediction[0, :]) == 1:
        label = "negative"
        confidence = (prediction[0, 1]/([prediction.sum()]))*100

    return [label, confidence]

# The four functions given below are the custom metric functions during our training. This is necessary to load our
# model

def sensitivity(y_true, y_pred):
    true_positives = K.sum(
        K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(
        K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(
        K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(
        K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(
        K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(
        K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(
        K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / \
        (predicted_positives + K.epsilon())
    recall = true_positives / \
        (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall) / \
        (precision+recall+K.epsilon())
    return f1_val


def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + K.epsilon())




if __name__ == "__main__":
    #Load the model
    model_file_location = input("Please enter the path to the model directory:- \n")
    model = tf.keras.models.load_model(model_file_location, custom_objects={'sensitivity':sensitivity, 'specificity': specificity, 'f1': f1, 'matthews_correlation':matthews_correlation}, compile=False)

    #Prepare the input for the model
    smiles_file_location = input("Please enter the smiles input file name. Ensure that the input is a .smi file with each Canonical SMILES string in a new line:- \n")
    predict_smiles, predict_input = process_smiles_input(smiles_file_location)

    for molecule in range(len(predict_input)):
        label, prob = dili_predict(model, predict_input[molecule, :])
        smile = predict_smiles[molecule]

        print("Smiles : {}, DILI_status : {}, Confidence : {}".format(smile, label, prob))
    








