import sys
import os
import io
import csv
import time
import uuid
import argparse
import torch
import apex
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
        'joy': 2,
        'sadness': 3,
        'surprise': 4,
    }

    def __init__(self, config, phase):
        self.config = config

        filepath = os.path.join(config.dataroot, f'{phase}.txt')
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

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def __len__(self):
        return len(self.texts)


class Trainer:
    def __init__(self, config):
        self.config = config

        model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking' if config.lang == 'ja' else 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(model_name, padding=True)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=config.n_labels, return_dict=True).to(config.device)
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)

        data_train= EmotionDataset(self.config, 'train')
        self.dataloader_train = DataLoader(data_train, batch_size=self.config.batch_size, shuffle=True)

        data_eval= EmotionDataset(self.config, 'eval')
        self.dataloader_eval = DataLoader(data_eval, batch_size=self.config.batch_size, shuffle=True)

        self.writer = SummaryWriter(log_dir=config.tensorboard_log_dir)
        self.best_accuracy = 0.0

        if config.fp16:
            self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, 'O1')

        self.load(self.config.model_path)

        if self.config.freeze_base:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

    def train(self, epoch):
        self.model.train()

        for i, (texts, labels) in enumerate(self.dataloader_train):
            start_time = time.time()
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True).to(self.config.device)
            labels = labels.to(self.config.device)

            outputs = self.model(**inputs, labels=labels)

            if self.config.fp16:
                with apex.amp.scale_loss(outputs.loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                outputs.loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if i % self.config.log_interval == 0:
                elapsed_time = time.time() - start_time
                self.config.logger.info('train epoch: {}, step: {}, loss: {:.2f}, time: {:.2f}'.format(epoch, i, outputs.loss, elapsed_time))

            self.writer.add_scalar('loss/train', outputs.loss, epoch, start_time)

        self.save(self.config.model_path)

    @torch.no_grad()
    def eval(self, epoch):
        self.model.eval()

        n_correct = 0
        losses = []
        start_time = time.time()

        for i, (texts, labels) in enumerate(self.dataloader_eval):
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True).to(self.config.device)
            labels = labels.to(self.config.device)

            outputs = self.model(**inputs, labels=labels)

            preds = torch.argmax(outputs.logits, dim=1)
            n_correct += (preds == labels).sum().cpu().item()
            losses.append(outputs.loss)

        elapsed_time = time.time() - start_time
        n_all = len(self.dataloader_eval.dataset)
        accuracy = n_correct / n_all
        average_loss = sum(losses)/len(losses)
        self.config.logger.info('eval epoch: {}, loss: {:.2f}, accuracy: {:.3f} ({}/{}), time: {:.2f}'.format(epoch, average_loss, accuracy, n_correct, n_all, elapsed_time))

        if not self.config.eval_only:
            self.writer.add_scalar('loss/eval', average_loss, epoch, start_time)
            self.writer.add_scalar('loss/acc', accuracy, epoch, start_time)

            if self.best_accuracy < accuracy:
                self.best_accuracy = accuracy
                self.save(self.config.best_model_path)

    def save(self, model_path):
        data = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'amp': apex.amp.state_dict() if self.config.fp16 else None,
            'batch_size': self.config.batch_size,
            'fp16': self.config.fp16,
        }
        torch.save(data, model_path)
        self.config.logger.info(f'save model to {model_path}')

    def load(self, model_path):
        if not os.path.isfile(model_path):
            return

        data = torch.load(model_path, map_location=self.config.device_name)
        self.model.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['optimizer'])
        if self.config.fp16:
            apex.amp.load_state_dict(data['amp'])
        self.config.logger.info(f'load model from {model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--loglevel', default='DEBUG')
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--n_labels', type=int, default=6, help='number of classes to train')
    parser.add_argument('--dataroot', default='data', help='path to data directory')
    parser.add_argument('--batch_size', type=int, default=64, help='size of batch')
    parser.add_argument('--epochs', type=int, default=10, help='epoch count')
    parser.add_argument('--fp16', action='store_true', help='run model with float16')
    parser.add_argument('--lang', default='ja', choices=['en', 'ja'])
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--name', default=None)
    parser.add_argument('--freeze_base', action='store_true')
    args = parser.parse_args()

    is_cpu = args.cpu or not torch.cuda.is_available()
    args.device_name = "cpu" if is_cpu else "cuda:0"
    args.device = torch.device(args.device_name)

    logger = setup_logger(name=__name__, level=args.loglevel)
    logger.info(args)
    args.logger = logger

    if args.name is None:
        args.name = str(uuid.uuid4())[:8]

    args.tensorboard_log_dir = f'{args.dataroot}/runs/{args.name}'

    args.model_path = f'{args.dataroot}/{args.name}.pth'
    args.best_model_path = f'{args.dataroot}/{args.name}.best.pth'

    trainer = Trainer(args)

    if args.eval_only:
        trainer.eval(0)
        sys.exit()

    for epoch in range(args.epochs):
        trainer.train(epoch)
        if epoch % args.eval_interval == 0:
            trainer.eval(epoch)

    trainer.eval(epoch)
