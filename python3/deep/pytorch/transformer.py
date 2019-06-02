import time
import argparse
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

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
        logit.masked_fill_(mask, -float('inf'))

        weights = F.softmax(logit, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)

        output = torch.matmul(attension_weight, v)
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

    def forward(self, input, *args, **kwargs):
        x = self.normal(input)
        x = self.layer(x, args, kwargs)
        x = self.dropout(x, training=self.training)
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

        attentions = nn.ModuleList()
        source_attentions = nn.ModuleList()
        ffns = nn.ModuleList()
        for _ in self.n_layers:
            attention = MultiHeadAttention(self.n_heads, self.dim, self.dropout)
            attentions.append(ResidualNormalizationWrapper(self.dim, attention, self.dropout))

            if (is_decoder):
                source_attention = MultiHeadAttention(self.n_heads, self.dim, self.dropout)
                source_attentions.append(ResidualNormalizationWrapper(self.dim, source_attention, self.dropout))

            ffn = FeedForward(self.dim, self.dim_hidden, self.dim, self.dropout)
            ffns.append(ResidualNormalizationWrapper(self.dim, ffn, self.dropout))

    def _get_mask(self, input):
        pad_tensor = torch.empty(1).fill_(PAD_ID).expand_as(input)
        mask = input == pad_tensor

        source_mask_np = np.triu(np.ones(), k=1).astype('uint8')
        source_mask = torch.from_numpy(source_mask_np) == 0

        return (mask, source_mask)

    def forward(self, input, src_enc=None):
        batch_size, n_sentences = input.size()
        (mask, att_mask) = self._get_mask(input)

        positions = torch.arange(n_sentences).unsqueeze(0)
        x = self.token_embeddings(input)
        x = x + self.position_embeddings(positions).expand_as(x)

        x = self.layer_norm_emb(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x *= mask.unsqueeze(-1).to(x.dtype)

        for i in self.n_layers:
            x = self.attentions[i](x, x, att_mask)

            if self.is_decoder:
                x = self.source_attention[i](x, src_enc, mask)

            x = self.ffns[i](x)
            x *= mask.unsqueeze(-1).to(x.dtype)

        return x

class Trainer():
    def __init__(self, config):
        self.config = config
        self.encoder = TransformerModel(config, is_decoder=False)
        self.decoder = TransformerModel(config, is_decoder=True)

        self.optimizer_enc = self._get_optimizer(self.encoder)
        self.scheduler_enc = self._get_scheduler(self.optimizer_enc)

        self.optimizer_dec = self._get_optimizer(self.decoder)
        self.scheduler_dec = self._get_scheduler(self.optimizer_dec)

    def _get_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)

    def _get_scheduler(self, optimizer):
        dim = self.config.dim
        warmup = self.config.warmup_steps
        return optim.LambdaLR(optimizer, lr_lambda=lambda step: dim ** -0.5 * min(step ** -0.5, warmup ** -1.5))

    def _get_batch(self):
        vocab_size = self.config.vocab_size
        batch_size = self.config.batch_size
        for i in range(batch_size):
            data = np.random.randint(1, vocab_size, size=(batch_size, 10))
            data[:, 0] = 1
            data = torch.from_numpy(data, requires_grad=False)
            return (data.clone(), data)

    def step(self):
        self.encoder.train()
        self.decoder.train()

        x, y = self._get_batch()

        enc_output = self.encoder(x)
        dec_output = self.decoder(x, enc_output)

        gen_output = self.generator(dec_output)
        loss = self.criterion(gen_output, y)
        loss.backward()

        self.optimizer_enc.step()
        self.optimizer_dec.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--epochs', type=int, default=10, help='epoch count')
    parser.add_argument('--batch_size', type=int, default=32, help='size of batch')
    parser.add_argument('--log_interval', type=int, default=5, help='step num to display log')
    parser.add_argument('--vocab_size', type=int, default=10, help='vocabulary size for copy task')
    parser.add_argument('--n_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--dim', type=int, default=512, help='dimention of word embeddings')
    parser.add_argument('--dropout', type=int, default=0.1, help='rate of dropout')
    args = parser.parse_args()
    print(args)

    is_cpu = args.cpu or not torch.cuda.is_available()
    device_name = "cpu" if is_cpu else "cuda:0"
    device = torch.device(device_name)

    trainer = Trainer(args)

    dataset = Dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    for epoch in range(args.epochs):
        start_time = time.time()

        for step, data in enumerate(dataloader):
            trainer.step()

            if step % args.log_interval == 0:
                elapsed_time = time.time() - start_time
                print('epoch: {:%.1f}, step: {}, loss: {:%.2f}, tokens/sec: {:%.1f}'.format(epoch, step, 0, 0))
                # trainer.print_loss(step)
                      


