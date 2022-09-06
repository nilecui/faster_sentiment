#########################################################################
#Author: nilecui
# Date: 2022-09-06 10:49:54
#LastEditors: nilecui
#LastEditTime: 2022-09-06 14:09:31
#FilePath: /faster_sentiment/faster_sentiment/faster_sentiment.py
# Description: This class is only responsible for processing English text.
#           Create fasttext object to train, evaluate and predict, use
#           dataset imdb or custom local file to process.
# Details do not determine success or failure!
# Copyright (c) 2022 by nilecui, All Rights Reserved.
#########################################################################

from concurrent.futures.process import _ExceptionWithTraceback
from functools import partial
import torch
from torchtext.utils import download_from_url, unicode_csv_reader
import io
import glob
import torch.nn.functional as F
import torch.nn as nn
import json
import os
import time
import torch
import random

from torchtext.legacy import data, datasets

import torch.optim as optim
import spacy

#from torchtext import datasets
from faster_sentiment.utils import generate_bigrams, Example
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        embedded = self.embedding(text)
                
        #embedded = [sent len, batch size, emb dim]
        
        embedded = embedded.permute(1, 0, 2)
        
        #embedded = [batch size, sent len, emb dim]
        
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        
        #pooled = [batch size, embedding_dim]
                
        return self.fc(pooled)


class FasterSent:
    def __init__(self, abs_dataset_path="fast_datasets", epochs=5):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        self.BATCH_SIZE = 64
        self.epochs = epochs
        self.abs_dataset_path = abs_dataset_path
        self.MAX_VOCAB_SIZE = 25_000

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(1)

        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None
        # 初始化数据集
        self.TEXT = None
        self.generate_dateset()
        self.INPUT_DIM = len(self.TEXT.vocab)
        self.EMBEDDING_DIM = 100
        self.OUTPUT_DIM = 1
        self.PAD_IDX = self.TEXT.vocab.stoi[self.TEXT.pad_token]

        self.model = self.create_model()
        self.optimizer = None
        self.criterion = None
        self.create_optimizer()

        self.update_embedding()
        self.EPOCHS = 5

        self.predict_nlp = None

    def create_model(self):
        # INPUT_DIM = len(self.TEXT.vocab)
        # EMBEDDING_DIM = 100
        # OUTPUT_DIM = 1
        # PAD_IDX = self.TEXT.vocab.stoi[self.TEXT.pad_token]

        self.model = FastText(
            self.INPUT_DIM, self.EMBEDDING_DIM, self.OUTPUT_DIM, self.PAD_IDX)
        return self.model

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def update_embedding(self):
        pretrained_embeddings = self.TEXT.vocab.vectors
        self.model.embedding.weight.data.copy_(pretrained_embeddings)

        UNK_IDX = self.TEXT.vocab.stoi[self.TEXT.unk_token]
        self.model.embedding.weight.data[UNK_IDX] = torch.zeros(
            self.EMBEDDING_DIM)
        self.model.embedding.weight.data[self.PAD_IDX] = torch.zeros(
            self.EMBEDDING_DIM)

    def create_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.BCEWithLogitsLoss()
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

    def binary_accuracy(self, preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """

        # round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        # convert into float for division
        correct = (rounded_preds == y).float()
        acc = correct.sum() / len(correct)
        return acc

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def generate_dateset(self):
        SEED = 1234
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

        self.TEXT = data.Field(tokenize='spacy',
                               tokenizer_language='en_core_web_sm',
                               preprocessing=generate_bigrams)

        self.LABEL = data.LabelField(dtype=torch.float)

        # self.train_data, self.test_data = datasets.IMDB.splits(self.TEXT, self.LABEL)
        # self.train_data, self.valid_data = self.train_data.split(random_state = random.seed(SEED)) # 切分训练集和验证集

        self.fields = {
            'text': ('text', self.TEXT),
            'label': ('label', self.LABEL),
        }

        # 加载自定义数据集
        data.dataset.sort = self.sort_key
        train_ds = data.TabularDataset.splits(
            path=self.abs_dataset_path,
            train='train.json',
            validation='test.json',
            test='test.json',
            format='json',
            fields=self.fields)[0]
        print(train_ds)
        print(f"train_ds=>{len(train_ds)}")

        self.train_data, self.valid_data = train_ds.split(
            random_state=random.seed(SEED))
        self.test_data = self.valid_data

        self.TEXT.build_vocab(self.train_data,
                              max_size=self.MAX_VOCAB_SIZE,
                              vectors="glove.6B.100d",
                              unk_init=torch.Tensor.normal_)

        self.LABEL.build_vocab(self.train_data)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size=self.BATCH_SIZE,
            device=device,
            sort_key=lambda x: len(x.text))

    def evaluate(self, model, iterator, criterion):

        epoch_loss = 0
        epoch_acc = 0

        self.model.eval()

        with torch.no_grad():

            for batch in iterator:

                predictions = self.model(batch.text).squeeze(1)

                loss = criterion(predictions, batch.label)

                acc = self.binary_accuracy(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def train(self, iterator, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0

        self.model.train()

        for batch in iterator:

            optimizer.zero_grad()

            predictions = self.model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = self.binary_accuracy(predictions, batch.label)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def start_train(self):
        best_valid_loss = float('inf')

        for epoch in range(self.epochs):

            start_time = time.time()

            train_loss, train_acc = self.train(
                self.train_iterator, self.optimizer, self.criterion)
            valid_loss, valid_acc = self.evaluate(
                self.model, self.valid_iterator, self.criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), 'tut3-model.pt')

            print(
                f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(
                f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(
                f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    def start_eval(self, model_path):
        self.model.load_state_dict(torch.load('tut3-model.pt'))
        test_loss, test_acc = self.evaluate(
            self.model, self.test_iterator, self.criterion)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

    def predict(self, text):
        self.predict_nlp = spacy.load('en_core_web_sm')

        self.model.eval()
        tokenized = generate_bigrams(
            [tok.text for tok in self.predict_nlp.tokenizer(text)])
        indexed = [self.TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(self.device)
        tensor = tensor.unsqueeze(1)
        prediction = torch.sigmoid(self.model(tensor))
        res = prediction.item()
        return res

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def generate_bigrams(self, x):
        n_grams = set(zip(*[x[i:] for i in range(2)]))
        for n_gram in n_grams:
            x.append(' '.join(n_gram))
        return x
