from transformers import BertModel, DistilBertModel
import torch.nn as nn
import torch

class BertClassifier(nn.Module):
    
    def __init__(self, dropout, num_classes, bert_model_name='bert-base-cased'):
        super(BertClassifier, self).__init__()
        
        #loading the model
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        #freezing the loaded model parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        
        #initializing dropout, linear, and activation layers
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)
        self.activation = nn.GELU()
    
    def forward(self, input_ids, attention_mask):
        #forward pass
        _, bert_output = self.bert(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  return_dict=False)
        dropout_output = self.activation(self.dropout(bert_output))
        
        #apply linear layer
        final_output = self.linear(dropout_output)
        return final_output

class DistilBertClassifier(nn.Module):
    
    def __init__(self, dropout, num_classes, bert_model_name='distilbert-base-uncased'):
        super(DistilBertClassifier, self).__init__()
        
        #loading the model
        self.bert = DistilBertModel.from_pretrained(bert_model_name)
        
        #freezing the loaded model parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        
        #initializing dropout, linear, and activation layers
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)
        self.activation = nn.GELU()
    
    def forward(self, input_ids, attention_mask):
        #forward pass
        bert_output = self.bert(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  return_dict=False)
        
        #extract the [CLS] token representation (first token)
        cls_token_embedding = bert_output[0][:, 0, :]
        
        #apply dropout to the hidden states
        dropout_output = self.dropout(cls_token_embedding)
        
        #apply gelu activation function
        gelu_output = self.activation(dropout_output)
        
        #apply linear layer
        final_output = self.linear(gelu_output)
        return final_output

class TextDataset(torch.utils.data.Dataset):
    
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels
        
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        return self.labels[idx], self.tokens[idx]
