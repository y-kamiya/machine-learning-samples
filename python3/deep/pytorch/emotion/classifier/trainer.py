import os
import io
import csv
import time
import uuid
import argparse
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from logzero import setup_logger

class EmotionDataset(Dataset):
    label_index_map = {
        'anger': 0,
        'disgust': 1,
        'fear': 2,
        'joy': 3,
        'sadness': 4,
        'surprise': 5,
    }

    def __init__(self, config, phase):
        self.config = config

        filepath = os.path.join(config.dataroot, config.dataset_name, f'{phase}.txt')
        self.texts = []
        self.labels = []
        with io.open(filepath, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for text, label_name in reader:
                if label_name not in self.label_index_map:
                    self.config.logger.warn(f'{label_name} is invalid label name, skipped')
                    continue
                self.texts.append(text)
                self.labels.append(self.label_index_map[label_name])

        # self.dataset = data.TabularDataset(
        #     path=os.path.join(config.dataroot, config.dataset_name, f'{phase}.txt'),
        #     format='tsv',
        #     fields=[
        #         ('data', data.Field(tokenize=tokenizer)),
        #         ('label', data.Field(sequential=False)),
        #     ]
        # )

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def __len__(self):
        return len(self.texts)


class Trainer:
    def __init__(self, config):
        self.config = config

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', padding=True)
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=config.n_labels, return_dict=True)
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)

        data_train= EmotionDataset(self.config, 'train')
        self.dataloader_train = DataLoader(data_train, batch_size=self.config.batch_size, shuffle=True)

        data_eval= EmotionDataset(self.config, 'eval')
        self.dataloader_eval = DataLoader(data_eval, batch_size=self.config.batch_size, shuffle=True)

        self.writer = SummaryWriter(log_dir=config.tensorboard_log_dir)

    def train(self, epoch):
        self.model.train()

        for i, (texts, labels) in enumerate(self.dataloader_train):
            start_time = time.time()
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True)
            outputs = self.model(**inputs, labels=labels)

            self.optimizer.zero_grad()
            outputs.loss.backward()
            self.optimizer.step()

            elapsed_time = time.time() - start_time
            self.config.logger.info('train epoch: {}, step: {}, loss: {:.2f}, time: {:.2f}'.format(epoch, i, outputs.loss, elapsed_time))

            self.writer.add_scalar('loss/train', outputs.loss, epoch, start_time)

    @torch.no_grad()
    def eval(self, epoch):
        self.model.eval()

        n_correct = 0
        losses = []
        start_time = time.time()

        for i, (texts, labels) in enumerate(self.dataloader_eval):
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True)
            outputs = self.model(**inputs, labels=labels)

            preds = torch.argmax(outputs.logits, dim=1)
            n_correct += (preds == labels).sum().item()
            losses.append(outputs.loss)

        elapsed_time = time.time() - start_time
        n_all = len(self.dataloader_eval.dataset)
        accuracy = n_correct / n_all
        self.config.logger.info('eval epoch: {}, accuracy: {:.3f} ({}/{}), time: {:.2f}'.format(epoch, accuracy, n_correct, n_all, elapsed_time))

        self.writer.add_scalar('loss/eval', sum(losses)/len(losses), epoch, start_time)
        self.writer.add_scalar('loss/acc', accuracy, epoch, start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--loglevel', default='DEBUG')
    parser.add_argument('--n_labels', type=int, default=6, help='number of classes to train')
    parser.add_argument('--dataroot', default='data', help='path to data directory')
    parser.add_argument('--dataset_name', default='default', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=64, help='size of batch')
    parser.add_argument('--epochs', type=int, default=10, help='epoch count')
    args = parser.parse_args()

    logger = setup_logger(name=__name__, level=args.loglevel)
    logger.info(args)
    args.logger = logger

    args.tensorboard_log_dir = f'{args.dataroot}/runs/{args.dataset_name}_{str(uuid.uuid4())[:8]}'

    trainer = Trainer(args)

    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.eval(epoch)
