import os
# Libraries
import torch

# Preliminaries
from torchtext.legacy.data import Field, TabularDataset, BucketIterator

# Models
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import ExponentialLR

# Training
import torch.optim as optim

from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix

import numpy as np

import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.manual_seed(0)


class LSTM(nn.Module):

    def __init__(self, dimension=512):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(len(text_field.vocab), 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2*dimension, 2*dimension)
        self.out = nn.Linear(2*dimension, 1)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        encoding_forward = output[range(len(output)), text_len - 1, :self.dimension]
        encoding_reverse = output[:, 0, self.dimension:]
        encoding_reduced = torch.cat((encoding_forward, encoding_reverse), 1)
        encoding = self.drop(encoding_reduced)

        hidden1 = self.fc1(encoding)
        hidden1 = self.relu(hidden1)
        hidden1 = self.drop(hidden1)
        
        out = self.out(hidden1)
        out = torch.squeeze(out, 1)

        text_out = torch.sigmoid(out)

        return text_out

def train(model, optimizer, scheduler, train_loader, track_every, # validate every how many batches
          criterion = nn.BCELoss(), num_epochs = 1):
  
    start = datetime.datetime.now()
    # initialize running values
    running_loss = 0.0
    local_batch_num = 0
    global_batch_num = 0

    # training loop
    model.train()
    for epoch in range(num_epochs):
        local_batch_num = 0
        running_loss = 0

        for (labels, (comment, comment_len)), _ in train_loader:
            labels = labels.to(device)
            comment = comment.to(device)
            comment_len = comment_len.to(device)
            output = model(comment, comment_len)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            local_batch_num += 1
            global_batch_num += 1

            if local_batch_num % track_every == 0:
                # print progress
                print('Epoch [{}/{}], Batch [{}/{}]'.format(epoch+1, num_epochs, local_batch_num, len(train_loader)))

        scheduler.step()

    end = datetime.datetime.now()
    torch.save(model.state_dict(), source_folder + '/lstm_model_final.pt')

    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))
    print('Model weights saved.')

def evaluate(model, eval_loader):
    y_pred = []
    y_true = []
    y_pred_raw = []

    model.eval()
    with torch.no_grad():
        for (labels, (comment, comment_len)), _ in eval_loader:           
            labels = labels.to(device)
            comment = comment.to(device)
            comment_len = comment_len.to(device)
            output = model(comment, comment_len)
            prediction = (output > 0.5).int()
            y_pred_raw.extend(output.tolist())
            y_pred.extend(prediction.tolist())
            y_true.extend(labels.tolist())
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], target_names=['Positive', 'Negative'], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    print('Confusion Matrix')
    print(cm)

    auc = roc_auc_score(y_true, y_pred_raw)
    print('auc: ', auc)
    sns.set(style='darkgrid')
    y_prob = np.stack((1-np.array(y_pred_raw), np.array(y_pred_raw)),axis=1)
    skplt.metrics.plot_roc(y_true, y_prob, classes_to_plot=[1], title="ROC Curve", plot_macro=False, plot_micro=False)
    plt.savefig('test_auc.png', bbox_inches='tight')
    plt.cla()
    plt.clf()

    sns.set(style='darkgrid')
    skplt.metrics.plot_confusion_matrix(y_true, y_pred)
    plt.savefig('test_confusion_matrix.png', bbox_inches='tight')


    state_dict = {'y_pred': y_pred,
                  'y_true': y_true,
                  'y_pred_raw': y_pred_raw}
    torch.save(state_dict, source_folder + '/final_preds.pt')


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(sequential=True, tokenize='spacy', lower=True, include_lengths=True, batch_first=True)
    fields = [('label', label_field), ('comment', text_field )]

    source_folder = '/home/j/joonjie/project'
    # TabularDataset
    train_split, test_split = TabularDataset.splits(path=source_folder, train='sarcasm_train_valid.csv', test='sarcasm_test.csv',
                                           format='CSV', fields=fields, skip_header=True)

    # Iterators
    train_iter = BucketIterator(train_split, batch_size=512,shuffle=True, sort_key=lambda x: len(x.comment),
                            device=device, sort=None, sort_within_batch=None)
    test_iter = BucketIterator(test_split, batch_size=512, sort_key=lambda x: len(x.comment),
                            device=device, sort=True, sort_within_batch=True)
    
    
    # Vocabulary
    text_field.build_vocab(train_split, min_freq=3)

    # Train
    model = LSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.80)
    train(model=model, optimizer=optimizer, scheduler=scheduler, 
    train_loader=train_iter, num_epochs=2, track_every=len(train_iter)//10)

    # Evaluation
    trained_model = LSTM().to(device)
    trained_model.load_state_dict(torch.load(source_folder + '/lstm_model_final.pt'))
    evaluate(trained_model, test_iter)

