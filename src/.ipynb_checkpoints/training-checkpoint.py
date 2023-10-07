import torch
from tqdm import tqdm
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from helper import after_subplot
from google.cloud import bigquery, storage
import pickle

def train_one_epoch(train_dataloader, model, optimizer, loss, mode='bert'):
    """
    Performs model training for one epoch
    
    INPUT
    :train_dataloader: data loader for training
    :model: the BERT model
    :optimizer: the optimizer used in training
    :loss: the loss function
    :mode: either BERT or DistilBERT
    
    OUTPUT
    :train_loss: average training loss
    """

    if torch.cuda.is_available():
        model.cuda()
        loss.cuda()

    # YOUR CODE HERE: set the model to training mode
    model.train()
    
    train_loss = 0.0

    for batch_idx, (batch_labels, batch_data) in tqdm(enumerate(train_dataloader), desc="Training",
                                                    total=len(train_dataloader),
                                                    leave=True,
                                                    ncols=80
                                                    ):
        #get input ids and attention mask
        input_ids = batch_data["input_ids"]
        attention_mask = batch_data["attention_mask"]
        
        # move data to GPU if available
        if torch.cuda.is_available():
            batch_labels = batch_labels.cuda()
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        
        #clear gradients of all optimized parameters
        optimizer.zero_grad()
        
        #match the dimension to bert's
        input_ids = torch.squeeze(input_ids, 1)
        
        #forward pass
        batch_output = model(input_ids, attention_mask)
        batch_output = torch.squeeze(batch_output)
        
        #calculate loss
        #convert batch_labels to LongTensor if it's currently an IntTensor
        batch_labels = batch_labels.long()
        loss_value = loss(batch_output, batch_labels)
        
        #backward pass
        loss_value.backward()
        
        #single optimization step -> updating parameter
        optimizer.step()
        
        # update average training loss
        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )
        
    return train_loss

def valid_one_epoch(valid_dataloader, model, loss, mode='bert'):
    """
    Function to validate at the end of one epoch
    
    INPUT
    :valid_dataloader: data loader for validation
    :model: the BERT model
    :loss: the loss function
    
    OUTPUT
    :valid_loss: average validation loss
    """

    with torch.no_grad():

        # set the model to evaluation mode
        model.eval()

        if torch.cuda.is_available():
            model.cuda()
            loss.cuda()

        valid_loss = 0.0
        for batch_idx, (batch_labels, batch_data) in tqdm(enumerate(valid_dataloader), desc="Validating",
                                                        total=len(valid_dataloader),
                                                        leave=True,
                                                        ncols=80):
            #get input ids and attention mask
            input_ids = batch_data["input_ids"]
            attention_mask = batch_data["attention_mask"]
            
            # move data to GPU if available
            if torch.cuda.is_available():
                batch_labels = batch_labels.cuda()
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            
            #match the dimension to bert's
            input_ids = torch.squeeze(input_ids, 1)
            
            #forward pass
            batch_output = model(input_ids, attention_mask)
            batch_output = torch.squeeze(batch_output)
            
            #calculate loss
            #convert batch_labels to LongTensor if it's currently an IntTensor
            batch_labels = batch_labels.long()
            loss_value = loss(batch_output, batch_labels)
                    
            #calculate average validation loss
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )    
    return valid_loss

def optimize(data_loaders, model, optimizer, loss, n_epochs, model_path, interactive_tracking=False):
    """
    Function to train BERT model for a specific epochs
    
    INPUT
    :data_loaders: a dictionary of data loaders containing the keys 'train', 'valid', 'test'
    :model: the BERT model
    :optimizer: the optimizer used in training
    :loss: the loss function
    :n_epochs: the number of epochs, datatype int
    :model_path: the path to save model weights, datatype string
    :interactive_tracking: whether the loss is graphed or not, default is False
    
    OUTPUT
    None
    
    The function doesn't output anything. It saves the weights to be loaded later during prediction 
    """
    # initialize tracker for minimum validation loss
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
        
    else:
        liveloss = None
    
    #initializing variables
    project_id = 'optimum-pier-401103'
    bucket_name = 'text-classification-bucket'
    valid_loss_min = None
    logs = {}
    
    #initializing gcs
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.get_bucket(bucket_name)
    
    #learning rate scheduler: setup a learning rate scheduler that
    #reduces the learning rate when the validation loss reaches a plateau
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.01) # YOUR CODE HERE
    
    for epoch in range(1, n_epochs + 1):

        train_loss = train_one_epoch(data_loaders["train"], model, optimizer, loss)

        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)

        #get max loss value between train and valid loss for live plotting purpose
        max_loss_value = max(train_loss, valid_loss)
        
        # print training/validation statistics
        print(f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}")

        # If the validation loss decreases by more than 1%, save the model
        if valid_loss_min is None or (
                (valid_loss_min - valid_loss) / valid_loss_min > 0.01
        ):
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")

            #save the weights to save_path
            with open('/tmp/model.pth', 'wb') as model_file:
                torch.save(model.state_dict(), model_file)
                
            #save the weights to gcs
            blob = bucket.blob(model_path)
            blob.upload_from_filename('/tmp/model.pth')
            
            #update valid_loss_min to valid_loss
            valid_loss_min = valid_loss

        #update learning rate if necessary
        scheduler.step(valid_loss)

        #log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["lr"] = optimizer.param_groups[0]["lr"]
            liveloss.update(logs)
            liveloss.send()

def one_epoch_test(test_dataloader, model, loss):
    """
    Function to validate at the end of one epoch
    
    INPUT
    :test_dataloader: data loader for testing
    :model: the BERT model
    :loss: the loss function
    
    OUTPUT
    :test_loss: average testing loss
    """
    
    #initialize variables, including correct for total correct predictions and total for total data
    
    test_loss = 0.0
    correct = 0.0
    total = 0.0

    
    with torch.no_grad():

        
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        for batch_idx, (batch_labels, batch_data) in tqdm(enumerate(test_dataloader), desc="Testing", 
                total=len(test_dataloader),
                leave=True,
                ncols=80):
            input_ids = batch_data["input_ids"]
            attention_mask = batch_data["attention_mask"]
            # move data to GPU
            if torch.cuda.is_available():
                batch_labels = batch_labels.cuda()
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            #match the dimension into bert
            input_ids = torch.squeeze(input_ids, 1)

            #forward pass
            batch_output = model(input_ids, attention_mask)
            batch_output = torch.squeeze(batch_output)
            
            #calculate loss
            #convert batch_labels to LongTensor if it's currently an IntTensor
            batch_labels = batch_labels.long()
            loss_value = loss(batch_output, batch_labels)

            batch_preds = torch.argmax(batch_output, axis=1)
            # Move predictions to CPU
            if torch.cuda.is_available():
                batch_labels = batch_labels.cpu()
                batch_preds = batch_preds.cpu()
            
            # update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss))

            # Calculate correct predictions
            correct += (batch_preds == batch_labels).sum().item()
            total += batch_labels.size(0)
        
        test_accuracy = 100. * correct / total

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print(f'\nTest Accuracy: {test_accuracy:.2f}% ({correct}/{total})')
