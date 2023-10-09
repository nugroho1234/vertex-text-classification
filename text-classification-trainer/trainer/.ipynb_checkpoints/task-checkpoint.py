#standard librares
import re
import pickle
import numpy as np
import pandas as pd
import random
import argparse
import logging
from tqdm import tqdm

#bigquery and storage
from google.cloud import bigquery, storage

#pytorch libraries
import torch
import torch.nn as nn
import torch.optim

#transformers libraries
from transformers import BertModel
from transformers import BertTokenizer

#scikit-learn libraries
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#helper functions
from trainer import helper
from trainer import data
from trainer.models import BertClassifier
from trainer import optimization
from trainer import training

#initializing random seed
random.seed(42)
torch.cuda.manual_seed(42)

#initializing variables
target_col = 'product'
text_col = 'consumer_complaint_narrative'

if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    #input arguments
    parser.add_argument(
        "--model_dir",
        help = "Directory to output best weights",
        type = str,
        default = "output/bert_pre_trained.pth"
    )
    parser.add_argument(
        "--project_id",
        help = "Project ID of this training job",
        type = str
    )
    parser.add_argument(
        "--bucket_name",
        help = "Bucket name of this training job to save and load models",
        type = str
    )
    parser.add_argument(
        "--tokens_dir",
        help = "Directory where the tokens are stored",
        type = str,
        default = "output/tokens.pkl"
    )
    parser.add_argument(
        "--labels_dir",
        help = "Directory where the encoded labels are stored",
        type = str,
        default = "output/labels.pkl"
    )
    parser.add_argument(
        "--label_encoder_dir",
        help = "Directory where the label encoder is stored",
        type = str,
        default = "output/label_encoder.pkl"
    )
    parser.add_argument(
        "--bq_uri",
        help = "BigQuery URI where the data is stored",
        type = str,
        default = "bq://optimum-pier-401103.bert_chat_dataset.bert-text-use"
    )
    parser.add_argument(
        "--optimizer_choice",
        help = "Choice of optimizer",
        choices = ['sgd', 'adam'],
        type = str,
        default = 'adam'
    )
    parser.add_argument(
        "--num_epochs",
        help = "Number of training epochs",
        type = int
    )
    parser.add_argument(
        "-v", "--verbose", 
        help = "Increase output verbosity",
        action = "store_true"
    )
    args = parser.parse_args()
    arguments = args.__dict__
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    #read data from the URI and store it in a pandas dataframe
    project_id = arguments['project_id']
    bucket_name = arguments['bucket_name']
    logging.info('Reading data from BigQuery')
    df = helper.read_data_from_bq(arguments['bq_uri'])
    logging.info('Preprocessing data')
    df = helper.preprocess_data(df, text_col, target_col)
    
    logging.info('Loading the tokenizer')
    #loading pre-trained tokenizer
    helper.load_from_gcs(project_id, bucket_name, arguments['tokens_dir'])
    logging.info(f"Loaded the tokens to {bucket_name}.{arguments['tokens_dir']}")
    
    logging.info('Loading the labels and label encoder')
    #encoding label
    label_encoder = helper.load_from_gcs(project_id, bucket_name, arguments['label_encoder_dir'])
    labels = helper.load_from_gcs(project_id, bucket_name, arguments['labels_dir'])
    
    num_classes = len(label_encoder.classes_)
    
    logging.info('Creating PyTorch Dataloaders...')
    #create dataloaders
    data_loaders = data.create_dataloaders(tokens, labels,
                                       test_size=0.2, 
                                       valid_size=0.25, 
                                       random_state=42, 
                                       batch_size=16, 
                                       shuffle=False, 
                                       drop_last=False)
    
    logging.info("Getting the loss function") 
    #get loss function
    loss = optimization.get_loss()
    
    logging.info("Initializing the BERT Classifier")
    #initialize model
    dropout = 0.5
    bert_model = BertClassifier(dropout, num_classes=num_classes, 
                                bert_model_name='bert-base-uncased')
    
    logging.info("Getting the optimizer")
    #get optimizer
    learning_rate = 1e-3
    optimizer = arguments['optimizer_choice']
    optimizer = optimization.get_optimizer(bert_model, optimizer, learning_rate)
    num_epochs = arguments['num_epochs']
    
    logging.info("Training the model")
    train_loss, valid_loss = training.optimize(
        data_loaders,
        bert_model,
        optimizer,
        loss,
        n_epochs=num_epochs,
        model_path=model_path,
        project_id=project_id,
        bucket_name=bucket_name,
        interactive_tracking=False
    )
    logging.info(f"Training loss is {train_loss} and validation loss is {valid_loss}")
    
    logging.info("Testing the model")
    test_accuracy, test_loss = training.one_epoch_test(data_loaders['test'], bert_model, loss)
    logging.info(f"Testing loss is {test_loss} with the accuracy of {test_accuracy:.2f}")
    
    logging.info("Training job successful, exiting...")
    
    
    