# Adapted from tutorial at https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import scikitplot as skplt


# Load data
data = pd.read_csv('sarcasm_train.csv',engine='python')
val = pd.read_csv('sarcasm_valid.csv',engine='python')
test = pd.read_csv('sarcasm_test.csv',engine='python')

# Drop useless columns
data = data[['label', 'comment']]
val = val[['label', 'comment']]
test_data = test[['label', 'comment']]

# Display 5 random samples
'''print(data.sample(5))
print(test_data.sample(5))'''

X_train = data.comment.values
y_train = data.label.values
X_val = val.comment.values
y_val = val.label.values
X_test = test_data.comment.values
y_test = test_data.label.values

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Create a function to tokenize a set of texts
def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text = sent,
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            padding='max_length',         # Pad sentence to max length
            truncation = True,
            return_attention_mask=True      # Return attention mask
            )
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

'''
# Concatenate train data and test data
all_comments = np.concatenate([data.comment.values, test_data.comment.values])

# Encode our concatenated data
encoded_comments = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_comments]

# Find the maximum length
max_len = max([len(sent) for sent in encoded_comments])
print('Max length: ', max_len)
# Max len is 9827'''

# Specify `MAX_LEN`
MAX_LEN = 512

# Run function `preprocessing_for_bert` on the train set and the validation set
print('Tokenizing training data...')
train_inputs, train_masks = preprocessing_for_bert(X_train)
torch.save(train_inputs, 'train_inputs.pt')
torch.save(train_masks, 'train_masks.pt')

#train_inputs = torch.load('train_inputs.pt')
#train_masks = torch.load('train_masks.pt')

print('Tokenizing validation data...')
val_inputs, val_masks = preprocessing_for_bert(X_val)
torch.save(val_inputs, 'val_inputs.pt')
torch.save(val_masks, 'val_masks.pt')

#val_inputs = torch.load('val_inputs.pt')
#val_masks = torch.load('val_masks.pt')

print('Tokenizing done.')

# Convert other data types to torch.Tensor
train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_val)

# For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
batch_size = 32

print('Creating Dataloaders...')

# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits
    
def initialize_model(epochs=4):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-6,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

# Specify loss function
loss_fn = nn.CrossEntropyLoss()

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    
    print("Training complete!")


def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

# Train model on training set
#set_seed(42)    # Set seed for reproducibility
#bert_classifier, optimizer, scheduler = initialize_model(epochs=2)
#print(len(train_dataloader))
#print('Training model...')
#train(bert_classifier, train_dataloader, val_dataloader, epochs=2, evaluation=True)



def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs

# Compute predicted probabilities on the validation set
#print('Testing model on validation set...')
#probs = bert_predict(bert_classifier, val_dataloader)

def evaluate_roc(probs, y_true, thres=0.5):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    preds = probs[:, 1]

    # Get accuracy over the test set
    y_pred = np.where(preds >= thres, 1, 0)
    print("-----------")
    print(y_pred.size)
    results = pd.DataFrame(y_pred)
    print(results.shape)
    results.to_csv('bertFF_output.csv', index=False, header = None)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')
    
    # Print confusion matrix
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    print("Confusion Matrix:")
    print(cm)
    
    # Print auc and save ROC AUC graph
    myauc = roc_auc_score(y_true, preds)
    print('auc: ', myauc)
    sns.set(style='darkgrid')
    y_prob = np.stack((1-np.array(preds), np.array(preds)),axis=1)
    skplt.metrics.plot_roc(y_true, y_prob, classes_to_plot=[1], title="ROC Curve", plot_macro=False, plot_micro=False)
    plt.savefig('test_auc.png', bbox_inches='tight')
    plt.cla()
    plt.clf()
    
    # save confusion matrix
    sns.set(style='darkgrid')
    print("-----------")
    print(y_pred.size)
    print(y_true.size)
    skplt.metrics.plot_confusion_matrix(y_true, y_pred)
    plt.savefig('test_confusion_matrix.png', bbox_inches='tight')

    # Get optimal threshold
    fpr, tpr, threshold = roc_curve(y_true, preds)
    optimal_threshold = threshold[np.argmax(tpr-fpr)]

    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')
    print('Optimal threshold: ', optimal_threshold)
    
    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.savefig('ROCAUC.png')
    
# Evaluate the Bert classifier on validation set
#print('Evaluating model...')
#evaluate_roc(probs, y_val)

# Concatenate the train set and the validation set
full_train_data = torch.utils.data.ConcatDataset([train_data, val_data])
full_train_sampler = RandomSampler(full_train_data)
full_train_dataloader = DataLoader(full_train_data, sampler=full_train_sampler, batch_size=32)

# Train the Bert Classifier on the entire training data
print('Training model on entire training data...')
set_seed(42)
bert_classifier, optimizer, scheduler = initialize_model(epochs=2)
train(bert_classifier, full_train_dataloader, epochs=1)
print('Training done...')

# save the model weight in the checkpoint variable
# and dump it to system on the model_path
# tip: the checkpoint can contain more than just the model
checkpoint = {'model_state_dict': bert_classifier.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()}
torch.save(checkpoint, 'checkpoint.pt')
print("Model saved!")

# Load pre-existing model
#print("Loading pre-existing model...")
#checkpoint = torch.load('checkpoint.pt')
#bert_classifier.load_state_dict(checkpoint['model_state_dict'], strict = False)
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Run `preprocessing_for_bert` on the test set
print('Tokenizing test data...')
test_inputs, test_masks = preprocessing_for_bert(test_data.comment)
torch.save(test_inputs, 'test_inputs.pt')
torch.save(test_masks, 'test_masks.pt')

#test_inputs = torch.load('test_inputs.pt')
#test_masks = torch.load('test_masks.pt')

# Create the DataLoader for our test set
test_dataset = TensorDataset(test_inputs, test_masks)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)

# Compute predicted probabilities on the test set
print('Testing model on test set...')
probs = bert_predict(bert_classifier, test_dataloader)

# Get predictions from the probabilities
threshold = 0.5345398
preds = np.where(probs[:, 1] > threshold, 1, 0)

# Number of comments predicted non-negative
print("Number of comments predicted non-negative: ", preds.sum())

# Evaluate the Bert classifier
print('Evaluating model with threshold of %f...' % threshold)
evaluate_roc(probs, y_test, threshold)