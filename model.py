import math
import os
import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# HuggingFace
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from accelerate import Accelerator
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl, IntervalStrategy
from transformers import AdamW, get_linear_schedule_with_warmup
# from datasets import load_metric
import evaluate

import numpy as np

import constants
import utils


class CLDataset(Dataset):
    def __init__(self, encodings, targets):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dimension):
        super().__init__()
        self.in_features = in_features
        self.W = nn.Linear(in_features, hidden_dimension)
        self.V = nn.Linear(hidden_dimension, 1)

    def forward(self, x):
        att = torch.tanh(self.W(x))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = torch.sum(attention_weights * x, dim=1)
        return context_vector


class CLModel(nn.Module):
    def __init__(self, path):
        super(CLModel, self).__init__()
        self.automodel = AutoModel.from_pretrained(path)
        self.config = AutoConfig.from_pretrained(path)
        self.attention_head = AttentionHead(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.config.hidden_size, 1)

    def forward(self, **xb):
        x = self.roberta(**xb)[0]
        x = self.head(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


def loss_fn(y_pred, y_gt):
    y_pred = y_pred.view(-1)
    y_gt = y_gt.view(-1)
    out = torch.sqrt(nn.MSELoss()(y_pred, y_gt))
    return out


def update(model, data, target, optimizer, accelerator, scheduler):

    device = accelerator.device
    model.train()
    data = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out, target)
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
    return loss.item()


def validate(model, validation_data_loader, device):
    model.eval()
    validation_loss = 0

    with torch.no_grad():
        for x_val, y_gt in validation_data_loader:
            x_val = x_val.to(device)
            y_gt = y_gt.to(device)
            y_val = model(x_val)
            validation_loss += loss_fn(y_val, y_gt)

    validation_loss /= len(validation_data_loader.dataset)
    return validation_loss


def train_one_fold(train_data_loader, validation_data_loader, hyperparams, number_of_epochs, validation_step):

    accelerator = Accelerator()
    device = accelerator.device

    model = CLModel().to(device) # TODO
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=hyperparams[constants.LEARNING_RATE],
                                  weight_decay=hyperparams[constants.WEIGHT_DECAY])

    number_of_training_steps = int(len(train_data_loader) * hyperparams[constants.NUM_EPOCH])
    number_of_warmup_steps = int(0.05 * number_of_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_training_steps=number_of_training_steps,
                                                num_warmup_steps=number_of_warmup_steps)
    model, optimizer, training_dataloader, scheduler = accelerator.prepare(model, optimizer,
                                                                           train_data_loader, scheduler)

    for epoch in range(hyperparams[constants.NUM_EPOCH]):

        training_loss = 0

        for batch_idx, (data, target) in enumerate(train_data_loader):

            training_loss += update(model=model, data=data, target=target,
                                    optimizer=optimizer, accelerator=accelerator, scheduler=scheduler) # TODO

            if (batch_idx % validation_step == 0) or (batch_idx == len(train_data_loader) - 1):
                validation_loss = validate(model=model, validation_data_loader=validation_data_loader, device=device)

                # TODO: Save the model

        training_loss /= len(training_dataloader.dataset)







def predict(input_model_name, test_data, test_targets, hyperparams, config):
    pass


def train(hyperparams, nfolds=1):

    for f in range(nfolds):




