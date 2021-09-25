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
from sklearn import metrics
import pycm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tabulate import tabulate


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
            for row in reader:
                text = row[0]
                label_name = 'none' if len(row) == 1 else row[1]
                if label_name not in self.label_index_map and phase != 'predict':
                    self.config.logger.warn(f'{label_name} is invalid label name, skipped')
                    continue

                self.texts.append(text)

                label = -1 if label_name == 'none' else self.label_index_map[label_name]
                self.labels.append(label)

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
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr)

        if not self.config.predict:
            data_train= EmotionDataset(self.config, 'train')
            self.dataloader_train = DataLoader(data_train, batch_size=self.config.batch_size, shuffle=True)

            data_eval= EmotionDataset(self.config, 'eval')
            self.dataloader_eval = DataLoader(data_eval, batch_size=self.config.batch_size, shuffle=False)

        self.writer = SummaryWriter(log_dir=config.tensorboard_log_dir)
        self.best_f1_score = 0.0

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

        all_labels = torch.empty(0)
        all_preds = torch.empty(0)
        losses = []
        start_time = time.time()

        for i, (texts, labels) in enumerate(self.dataloader_eval):
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True).to(self.config.device)
            labels = labels.to(self.config.device)

            outputs = self.model(**inputs, labels=labels)

            preds = torch.argmax(outputs.logits, dim=1)
            losses.append(outputs.loss)

            all_labels = torch.cat([all_labels, labels.cpu()])
            all_preds = torch.cat([all_preds, preds.cpu()])

        elapsed_time = time.time() - start_time
        average_loss = sum(losses)/len(losses)
        self.config.logger.info('eval epoch: {}, loss: {:.2f}, time: {:.2f}'.format(epoch, average_loss, elapsed_time))

        self.__log_confusion_matrix(all_preds, all_labels, epoch)

        columns = EmotionDataset.label_index_map.keys()
        df = pd.DataFrame(metrics.classification_report(all_labels, all_preds, output_dict=True))
        print(tabulate(df, headers='keys', tablefmt="github", floatfmt='.2f'))

        if not self.config.eval_only:
            f1_score = df.loc['f1-score', 'macro avg']
            self.writer.add_scalar('loss/eval', average_loss, epoch, start_time)
            self.writer.add_scalar('loss/f1_score', f1_score, epoch, start_time)

            if self.best_f1_score < f1_score:
                self.best_f1_score = f1_score
                self.save(self.config.best_model_path)

    @torch.no_grad()
    def predict(self):
        label_map = {value: key for key, value in EmotionDataset.label_index_map.items()}
        label_map[-1] = 'none'
        np.set_printoptions(precision=0)

        dataset = EmotionDataset(self.config, 'predict')
        loader = DataLoader(dataset, batch_size=self.config.batch_size)

        output_path = os.path.join(self.config.dataroot, 'predict_result')
        with open(output_path, 'w') as f:
            for i, (texts, labels) in enumerate(loader):
                inputs = self.tokenizer(texts, return_tensors='pt', padding=True).to(self.config.device)
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                probs = F.softmax(outputs.logits, dim=1) * 100
                for j in range(len(texts)):
                    pred_label_name = label_map[preds[j].item()]
                    true_label_name = label_map[labels[j].item()]
                    prob = probs[j].cpu().numpy()
                    f.write(f'{pred_label_name}\t{prob}\t{true_label_name}\t{texts[j]}\n')

    def __log_confusion_matrix(self, all_preds, all_labels, epoch):
        label_map = {value: key for key, value in EmotionDataset.label_index_map.items()}
        cm = metrics.confusion_matrix(y_pred=all_preds.numpy(), y_true=all_labels.numpy(), normalize='true')
        display = metrics.ConfusionMatrixDisplay(cm, display_labels=label_map.values())
        display.plot(cmap=plt.cm.Blues)
        
        buf = io.BytesIO()
        display.figure_.savefig(buf, format="png", dpi=180)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()

        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.writer.add_image('confusion_maatrix', img, epoch, dataformats='HWC')
        
        cm = pycm.ConfusionMatrix(actual_vector=all_labels.numpy(), predict_vector=all_preds.numpy())
        cm.relabel(mapping=label_map)
        cm.print_normalized_matrix()

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
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--name', default=None)
    parser.add_argument('--freeze_base', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()

    pd.options.display.precision = 3
    pd.options.display.max_columns = 30

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

    if args.predict:
        trainer.predict()
        sys.exit()

    for epoch in range(args.epochs):
        trainer.train(epoch)
        if epoch % args.eval_interval == 0:
            trainer.eval(epoch)

    trainer.eval(epoch)
