import os
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

BOS_ID = 1
EOS_ID = 2
PAD_ID = 3

class SimpleAttention(nn.Module):
    def __init__(self, dim):
        super(SimpleAttention, self).__init__()

        self.dim = dim

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)

    def forward(self, x, memory, attension_mask):
        query = self.q_lin(x)
        key = self.k_lin(memory)
        value = self.v_lin(memory)

        logit = torch.matmul(query, torch.transpose(key, 1, 2))
        attension_weight = F.softmax(logit, dim=2)

        output = torch.matmul(attension_weight, value)

        return self.out_lin(output)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.dropout = dropout

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)

    def forward(self, x, memory, mask):
        batch_size, length, dim = x.shape
        assert dim == self.dim, 'dimension mismatched'

        dim_per_head = dim // self.n_heads

        def split(x):
            x = x.view(batch_size, -1, self.n_heads, dim_per_head)
            return x.transpose(1, 2)

        def combine(x):
            x = x.transpose(1, 2)
            return x.contiguous().view(batch_size, -1, dim)

        q = split(self.q_lin(x))
        k = split(self.k_lin(memory))
        v = split(self.v_lin(memory))

        q = q / math.sqrt(dim_per_head)

        logit = torch.matmul(q, k.transpose(2, 3))
        logit.masked_fill_(mask.view(-1, 1, 1, length), -float('inf'))

        weights = F.softmax(logit, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)

        output = torch.matmul(weights, v)
        output = combine(output)

        return self.out_lin(output)

class FeedForward(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, dropout):
        super(FeedForward, self).__init__()

        self.dropout = dropout
        self.hidden = nn.Linear(dim_in, dim_hidden)
        self.out = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.out(x)
        return F.dropout(x, p=self.dropout, training=self.training)

class ResidualNormalizationWrapper(nn.Module):
    def __init__(self, dim, layer, dropout):
        super(ResidualNormalizationWrapper, self).__init__()

        self.layer = layer
        self.dropout = dropout
        self.normal = nn.LayerNorm(dim)

    def forward(self, input, *args):
        x = self.normal(input)
        x = self.layer(x, *args)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return input + x

class TransformerModel(nn.Module):
    def __init__(self, config, is_decoder):
        super(TransformerModel, self).__init__()

        self.config = config
        self.is_decoder = is_decoder
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dim_hidden = config.dim * 4
        self.dropout = config.dropout

        self.token_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.position_embeddings = nn.Embedding(config.n_words, config.dim)

        self.layer_norm_emb = nn.LayerNorm(self.dim)

        self.attentions = nn.ModuleList()
        self.source_attentions = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(self.n_layers):
            attention = MultiHeadAttention(self.n_heads, self.dim, self.dropout)
            self.attentions.append(ResidualNormalizationWrapper(self.dim, attention, self.dropout))

            if (is_decoder):
                source_attention = MultiHeadAttention(self.n_heads, self.dim, self.dropout)
                self.source_attentions.append(ResidualNormalizationWrapper(self.dim, source_attention, self.dropout))

            ffn = FeedForward(self.dim, self.dim_hidden, self.dim, self.dropout)
            self.ffns.append(ResidualNormalizationWrapper(self.dim, ffn, self.dropout))

    def _get_mask(self, input):
        pad_tensor = torch.empty(1).fill_(PAD_ID).expand_as(input).to(torch.long)
        mask = input == pad_tensor

        shape = input.size()
        source_mask_np = np.triu(np.ones(shape), k=1).astype('uint8')
        source_mask = torch.from_numpy(source_mask_np) == 0

        return (mask, source_mask)

    def forward(self, input, src_enc=None):
        batch_size, n_sentences = input.size()
        (mask, att_mask) = self._get_mask(input)

        positions = torch.arange(n_sentences).long().unsqueeze(0)
        x = self.token_embeddings(input)
        x = x + self.position_embeddings(positions).expand_as(x)

        x = self.layer_norm_emb(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x[mask] = 0

        for i in range(self.n_layers):
            x = self.attentions[i](x, x, att_mask)

            if self.is_decoder:
                x = self.source_attentions[i](x, src_enc, mask)

            x = self.ffns[i](x)
            x[mask] = 0

        return x

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.encoder = TransformerModel(config, is_decoder=False)
        self.decoder = TransformerModel(config, is_decoder=True)
        self.generator = nn.Linear(config.dim, config.vocab_size)

        self.optimizer_enc = self._get_optimizer(self.encoder)
        self.scheduler_enc = self._get_scheduler(self.optimizer_enc)

        self.optimizer_dec = self._get_optimizer(self.decoder)
        self.scheduler_dec = self._get_scheduler(self.optimizer_dec)

        self.criterion = nn.KLDivLoss(size_average=True)

        if os.path.isfile(config.model):
            data = torch.load(config.model)
            self.encoder.load_state_dict(data['encoder'])
            self.decoder.load_state_dict(data['decoder'])
            self.generator.load_state_dict(data['generator'])
            self.optimizer_enc.load_state_dict(data['optimizer_enc'])
            self.optimizer_dec.load_state_dict(data['optimizer_dec'])

        self.start_time = time.time()
        self.steps = 0
        self.stats = {
            'sentences': 0,
            'words': 0,
            'loss': 0.0,
        }

    def save(self):
        data = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'generator': self.generator.state_dict(),
            'optimizer_enc': self.optimizer_enc.state_dict(),
            'optimizer_dec': self.optimizer_dec.state_dict(),
        }
        torch.save(data, self.config.model)
        print('save model to {}'.format(self.config.model))

    def _get_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    def _get_scheduler(self, optimizer):
        dim = self.config.dim
        warmup = self.config.warmup_steps

        def update(step):
            current_step = step + 1
            return 2 * dim ** -0.5 * min(current_step ** -0.5, current_step * warmup ** -1.5)

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=update)

    def _get_batch(self):
        vocab_size = self.config.vocab_size
        batch_size = self.config.batch_size
        n_sentences = 5
        for i in range(200):
            data = np.random.randint(PAD_ID+1, vocab_size, size=(batch_size, n_sentences))
            data[:, 0] = BOS_ID
            data = torch.from_numpy(data).requires_grad_(False)
            return (data.clone(), data)

    def _generate(self, x):
        return F.log_softmax(self.generator(x), dim=-1)

    def step(self):
        self.encoder.train()
        self.decoder.train()

        x, y = self._get_batch()

        enc_output = self.encoder(x)
        dec_output = self.decoder(x, enc_output)

        gen_output = self._generate(dec_output)

        target = torch.zeros_like(gen_output)
        target.scatter_(2, y.unsqueeze(-1), 1)

        self.optimizer_enc.zero_grad()
        self.optimizer_dec.zero_grad()

        loss = self.criterion(gen_output, target)
        loss.backward()

        self.optimizer_enc.step()
        self.optimizer_dec.step()

        self.stats['loss'] = loss.item()
        self.stats['sentences'] += x.size(0)
        self.stats['words'] += (y != PAD_ID).sum().item()

    def step_end(self, step):
        self.steps += 1
        self._print_log()

        self.scheduler_enc.step()
        self.scheduler_dec.step()

    def _print_log(self):
        # if self.steps % args.log_interval != 0:
            # return

        current_time = time.time()
        elapsed_time = current_time - self.start_time
        lr = self.optimizer_enc.param_groups[0]['lr']
        print('step: {}, loss: {:.2f}, tokens/sec: {:.1f}, lr: {:.6f}'.format(self.steps, self.stats['loss'], self.stats['words'] / elapsed_time, lr))

        self.start_time = current_time
        self.stats['sentences'] = 0
        self.stats['words'] = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--epochs', type=int, default=10, help='epoch count')
    parser.add_argument('--batch_size', type=int, default=2, help='size of batch')
    parser.add_argument('--log_interval', type=int, default=5, help='step num to display log')
    parser.add_argument('--vocab_size', type=int, default=8, help='vocabulary size for copy task')
    parser.add_argument('--n_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--n_heads', type=int, default=8, help='number of heads for multi head attention')
    parser.add_argument('--n_words', type=int, default=10, help='number of words max')
    parser.add_argument('--dim', type=int, default=8, help='dimention of word embeddings')
    parser.add_argument('--dropout', type=int, default=0.1, help='rate of dropout')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='adam lr increases until this steps have passed')
    parser.add_argument('--model', default='model.pth', help='file to save model parameters')
    args = parser.parse_args()
    print(args)

    is_cpu = args.cpu or not torch.cuda.is_available()
    device_name = "cpu" if is_cpu else "cuda:0"
    device = torch.device(device_name)

    trainer = Trainer(args)

    # dataset = Dataset(args)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size)

    for epoch in range(args.epochs):
        start_time = time.time()

        # for step, data in enumerate(dataloader):
        for i in range(200):
            trainer.step()
            trainer.step_end(i)
                      
        trainer.save()


