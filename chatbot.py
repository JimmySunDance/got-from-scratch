import argparse
import math
import mmap
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


dev = 'mps' if torch.backends.mps.is_available()\
    else 'cuda' if torch.backends.cuda.is_available() else 'cpu'
print(f'You are using: {dev}')


chars = ''
with open('data/vocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))


batch_size = 64
block_size = 128
max_iters = 200
eval_iters = 50
n_embd = 384
n_head = 8
n_layer = 8
dropout = 0.2

vocab_size = len(chars)


string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [ string_to_int[c] for c in s ]
decode = lambda l: ''.join([ int_to_string[i] for i in l])


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            'tril', 
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim = -1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) 
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    '''Simple linear layer followed by a non-linearity'''
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    '''Transformer block: communication followed by computation'''
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head = n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd) # Final layer normalisation
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self.__init__weights)
    
    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)

    def forward(self, index, targets=None):
        B, T = index.shape

        # idx and targets are both (B, T) tensor of ints
        tok_embd = self.token_embedding_table(index) # (B, T, C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=dev)) 
        x = tok_embd + pos_embd # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens): 
            index_cond = index[:, -block_size:]
            logits, loss = self.forward(index_cond) # get the predictions
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim = -1) # (B, C)
            index_next = torch.multinomial(probs, num_samples = 1) # (B, 1)
            index = torch.cat((index, index_next), dim = -1) # (B, T+1)
        return index
    

print('Loading model parameters...')
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)
print('Load successful')

m = model.to(dev)

while True:
    prompt = input('Prompt: \n')
    if prompt == 'quit()':
        print('Quitting chatbot...')
        exit(0)
    context = torch.tensor(encode(prompt), dtype=torch.long, device=dev)
    generated_chars = decode(
        m.generate(
            context.unsqueeze(0), 
            max_new_tokens= 150
        )[0].tolist()
    )
    print(f'Completion:\n{generated_chars}')