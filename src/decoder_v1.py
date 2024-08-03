import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder_utils import get_device, read_text, encoder_decoder, get_train_val, get_batch, eval_model

N_EMBD = 384#384 # embedding space
BLOCK_SIZE = 256 # context length
BATCH_SIZE = 64 #64
N_HEAD  = 6
N_LAYER = 6

DROPOUT = 0.2

LEARNING_RATE = 3e-4
MAX_ITERS = 1
EVAL_INTERVAL = 5000

DATA_SIZE = 1000

SAVEMODEL_FOLDER='../models'
SAVERESULT_FOLDER='../results'
DATA_FOLDER='../data'

DATA_PATH = f'{DATA_FOLDER}/python_{DATA_SIZE}.txt'
SAVERESULT_PATH = f'{SAVERESULT_FOLDER}/decoder_v1_py_{DATA_SIZE}.txt'
SAVEMODEL_PATH = f'{SAVEMODEL_FOLDER}/decoder_v1_py_P{DATA_SIZE}.pt'

# def attention1(x):
#     B, T, C= x.shape
#     xbow = torch.zeros((B, T, C))
#     w= torch.tril(torch.ones(T, T))
#     w = w/w.sum(1, keepdim=True)
#     xbow = w @x # (T, T) @ (B, T, C)  -> (B, T, T) @ (B, T, C) -> (B, T, C)

class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.val= nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))) # will not be considered as model params

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x) # roughly the info that this token contains ->  B, T, head_size
        q = self.query(x) # roughly what the token is interested in -> B, T, head_size

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # link query vs key (B, T, head_size) * (B, head_size, T) -> B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        v = self.val(x)
        out = wei @ v
        return out


class MultiHeadedAttention(nn.Module):
    """
    Using multiple attention head
    """

    def __init__(self, num_head, head_size):
        super().__init__()
        self.num_head = num_head
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(num_head * head_size, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim =-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4* n_embd),
            nn.ReLU(),
            nn.Linear(4* n_embd, n_embd),
            nn.Dropout (0.2)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd// n_head
        self.sa = MultiHeadedAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class GPTDecoderOnly(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embd = nn.Embedding(vocab_size, N_EMBD)
        self.pos_embd = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embd(idx)  # (B,T,C)
        pos_emb = self.pos_embd(torch.arange(T, device=device))  # generate array from 0-> T-1 -> (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -BLOCK_SIZE:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


text = read_text(DATA_PATH)
chars = sorted(list(set(text)))
vocab_size = len(chars)

encode, decode = encoder_decoder(chars)
device = get_device()
data = torch.tensor(encode(text), dtype=torch.long, device=device)
train_data, val_data = get_train_val(data)

model = GPTDecoderOnly(vocab_size)
device = get_device()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
print (f"vocal size {vocab_size}")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

with open(SAVERESULT_PATH, 'w') as f:
    num_step = len(train_data) // BATCH_SIZE
    for iter in range(MAX_ITERS):
        for step in range(num_step):
            # every once in a while evaluate the loss on train and val sets
            if step % EVAL_INTERVAL == 0 or step == num_step - 1:
                train_loss, val_loss = eval_model(model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE)
                print(f"iter: {iter}/{MAX_ITERS}, step: {step}/{num_step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
                # generate from the model
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                gen_text = decode(m.generate(context, max_new_tokens=1000)[0].tolist())
                print (gen_text)

                # save every 1000 steps
                if step % 10000 == 0:
                    torch.save(m.state_dict(), f'{SAVEMODEL_PATH}/decoder_v1_py_{DATA_SIZE}_{iter}_{step}.pt')
                f.write(f"iter: {iter}/{MAX_ITERS}, step: {step}/{num_step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}\n")
                f.write(gen_text)
                f.write('================================================================ \n ')
                f.flush()

            # sample a batch of data
            xb, yb = get_batch('train', train_data, val_data, BLOCK_SIZE, BATCH_SIZE)

            # evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    gen_text = decode(m.generate(context, max_new_tokens=2000)[0].tolist())
    #open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
    f.write(gen_text)
    f.close()

    # save model
    torch.save(m.state_dict(), SAVEMODEL_PATH)


# # load weight
# model.load_state_dict(torch.load(SAVEMODEL))
# text = f"""
# class SinglyLinkedList:
#
#     def __init__(self, *items):
#         if items:
#             self.tail = None
#             for each in items:
#                 self.append(each)
#         else:
#             self.head = None
#             self.tail = None
#
# """
#
# context = torch.tensor(encode(text), dtype=torch.long, device=device).reshape(1, -1)
# print (context.shape)
# gen_text = decode(m.generate(context, max_new_tokens=2000)[0].tolist())
# print (gen_text)