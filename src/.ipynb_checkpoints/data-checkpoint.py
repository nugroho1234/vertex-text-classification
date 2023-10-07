from sklearn.model_selection import train_test_split
from models import TextDataset
import torch

def create_dataloaders(tokens, labels, test_size=0.2, valid_size=0.25, random_state=42, batch_size=16, shuffle=False, drop_last=False):
    """
    Function to create pytorch dataloaders.
    
    INPUT
    :tokens: a list of tokenized text columns from BertTokenizer
    :labels: an array of labels from the target column from LabelEncoder (already fitted and transformed)
    :test_size: the size of test dataset from all dataset
    :valid_size: the size of validation dataset from temporary training data
    :random_state: random state
    :batch_size: batch size for dataloaders
    :shuffle: whether there is shuffling done in the dataloaders. Use True for training dataloader
    :drop_last: whether to drop data if the total number of samples is not divisible by batch_size
    
    OUTPUT
    :data_loaders: a dictionary containing the keys:
        :train: dataloader for training data
        :valid: dataloader for validation data
        :test: dataloader for test data
    """
    
    #dataloaders variable
    data_loaders = {}
    
    #split the dataset
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(tokens, labels,
                                                   test_size=test_size, random_state=random_state)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_temp, 
                                                          y_train_temp,
                                                         test_size=valid_size, random_state=random_state)

    print(f'There are {len(X_train)} training data, and {len(X_valid)} validation data, and {len(X_test)} testing data')
    
    #create pytorch dataset
    train_dataset = TextDataset(X_train, y_train)
    valid_dataset = TextDataset(X_valid, y_valid)
    test_dataset = TextDataset(X_test, y_test)
    
    #create dataloaders
    #create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=drop_last)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                             batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=drop_last)
    
    data_loaders['train'] = train_loader
    data_loaders['valid'] = valid_loader
    data_loaders['test'] = test_loader
    return data_loaders
    