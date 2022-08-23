import math
import os
import datetime

import torch
from torch.utils.data import Dataset, DataLoader

# HuggingFace
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl, IntervalStrategy
from transformers import AdamW, get_linear_schedule_with_warmup
# from datasets import load_metric
import evaluate

import numpy as np

import constants
import utils


class CommonLitDataset(Dataset):
    def __init__(self, encodings, targets):
        self.encodings = encodings
        self.targets = targets

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.targets[idx])
        return item

    def __len__(self):
        return len(self.targets)


class CommonLitCallback(TrainerCallback):

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(f'Training Start: {datetime.datetime.now()}')
        print('--')

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(f'Training End: {datetime.datetime.now()}')
        print('--')

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(f'Epoch Start: {state.epoch} / {args.num_train_epochs} @ {datetime.datetime.now()}')
        print('--')

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(f'Epoch Finish: {state.epoch} / {args.num_train_epochs} @ {datetime.datetime.now()}')
        print('--')


def train(input_model_name, output_model_name,
          train_data, train_targets,
          hyperparams, config,
          validation_data=None, validation_targets=None):
    model_folder = utils.generate_local_model_folder_path(input_model_name)
    output_folder = utils.generate_local_model_folder_path(output_model_name)

    if not os.path.exists(model_folder):
        model_folder = input_model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    print('--')

    tokenizer = AutoTokenizer.from_pretrained(model_folder)

    train_encodings = tokenizer(train_data, truncation=True, padding=True, max_length=constants.MAX_LENGTH)
    train_dataset = CommonLitDataset(train_encodings, train_targets)
    train_dl = DataLoader(train_dataset, shuffle=True, batch_size=hyperparams[constants.BATCH_SIZE])

    if (validation_data is not None) and (validation_targets is not None):
        validation_encodings = tokenizer(validation_data, truncation=True, padding=True, max_length=constants.MAX_LENGTH)
        validation_dataset = CommonLitDataset(validation_encodings, validation_targets)
    else:
        validation_dataset = None

    training_steps = len(train_dl) * hyperparams[constants.NUM_EPOCH]
    warmup_steps = math.ceil(training_steps * 0.06)

    training_args = TrainingArguments(output_dir=output_folder,
                                      num_train_epochs=hyperparams[constants.NUM_EPOCH],
                                      per_device_train_batch_size=hyperparams[constants.BATCH_SIZE],
                                      per_device_eval_batch_size=1,
                                      logging_dir=constants.OUT_FOLDER,
                                      logging_steps=config[constants.LOGGING_STEPS],
                                      seed=constants.RANDOM_SEED,
                                      weight_decay=hyperparams[constants.WEIGHT_DECAY],
                                      learning_rate=hyperparams[constants.LEARNING_RATE],
                                      save_strategy=IntervalStrategy.EPOCH,
                                      evaluation_strategy=IntervalStrategy.EPOCH)

    model = AutoModelForSequenceClassification.from_pretrained(model_folder, num_labels=config[constants.NUM_LABELS])
    model.config = AutoConfig.from_pretrained(model_folder, num_labels=config[constants.NUM_LABELS])
    # model = model.to(device)

    optimizer = AdamW(model.parameters(),
                      correct_bias=hyperparams[constants.BIAS],
                      lr=hyperparams[constants.LEARNING_RATE])
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_training_steps=training_steps,
                                                num_warmup_steps=warmup_steps)
    metric = evaluate.load(constants.ACCURACY)
    # metric = load_metric(constants.ACCURACY)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=validation_dataset,
                      optimizers=(optimizer, scheduler),
                      callbacks=[CommonLitCallback],
                      compute_metrics=compute_metrics)

    trainer.train(),


    dummy = -32


def train_with_cross_validation():
    pass
