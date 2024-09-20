import math
import os
import time
from contextlib import nullcontext

import numpy as np
import tiktoken
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt2_model import GPT, GPTConfig
from gpt2_dataloader import DataLoaderLite
import torch.nn.functional as F

############ CONFIG #############
eval_interval = 100
save_interval = 2000
log_interval = 1
eval_iters = 50
eval_only = True  # if True, script exits right after the first eval
is_compile =False # default is True
always_save_checkpoint = True  # if True, always save a checkpoint after each eval

batch_size = 4  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
total_batch_size = 524288

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
vocab_size = 50304

learning_rate = 6e-4  # max learning rate
# max_iters = 600000  # total number of training iterations
max_iters = 19073

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 500  # how many steps to warm up for
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# weight and bias
wandb_log = False  # disabled by default
wandb_project = 'codegen'
wandb_run_name = 'gpt2' + str(time.time()) # 'run' + str(time.time())

init_from = "resume"  # "scratch", "resume
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
master_process = True
seed_offset = 0

# DDP settings
ddp = True
backend = 'nccl'  # 'nccl', 'gloo', etc.

out_dir = '../models/gpt2'
os.makedirs(out_dir, exist_ok=True)

############## END CONFIG ##############


ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    print(f"ddp rank: {ddp_rank}, ddp local rank: {ddp_local_rank}")
    device = f'cuda:{ddp_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed

else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

gradient_accumulation_steps = total_batch_size // (batch_size * block_size * ddp_world_size)
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

# not using distributed training
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337 + seed_offset)

# optimization 1: tf16 precision
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast

# note: float16 data type will automatically use a GradScaler
dtype = 'bfloat16'  # 'float32', 'bfloat16', 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
###############

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
# init a new model from scratch
print("Initializing a new model from scratch")

# determine the vocab size we'll use for from-scratch training
meta_vocab_size = None  # TODO: init meta vocal size here
if meta_vocab_size is None:
    print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")

# init model
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)  # start with model_args from command line
model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304

if init_from == 'scratch':
    model = GPT(GPTConfig(**model_args))
    iter_num = 0
    best_val_loss = 10 ** 9

elif init_from == 'resume':

    checkpoint = torch.load("../models/gpt2/models_gpt2_ckpt_8000.pt", map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    state_dict = checkpoint['model']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model = GPT(GPTConfig(**model_args))
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size  # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory

# optimization 2: compile the model
uncompiled_model = model

if is_compile:
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
enc = tiktoken.get_encoding("gpt2")


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)

            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# code_starter = f"""
#       def _broadcast_indexes_vectorized(self, key):
#             variables = []
#             out_dims_set = OrderedSet()
#             for dim, value in zip(self.dims, key):
#                 if isinstance(value, slice):
#                     out_dims_set.add(dim)
#                 else:
#                     variable = (
#                         value
#                         if isinstance(value, Variable)
#                         else as_variable(value, name=dim)
#                     )
# """


code_starter = '''
    def to_index(self):
        # Convert this variable to a pandas.Index
        return self.to_index_variable().to_index()

    def to_dict(self, data=True):
        # Dictionary representation of variable.
        item = {"dims": self.dims, "attrs": decode_numpy_dict_values(self.attrs)}
'''

@torch.no_grad()
def generate():
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    model.eval()
    num_return_sequences = 1
    max_length = 1024
    # print (code_starter)
    tokens = enc.encode(code_starter)
    # print(len(tokens))
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)  # ddp rank
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with ctx:
                logits, loss = model(xgen)  # (B, T, vocab_size)
            # take the logits at the last position
            # logits = logits[:, -1, :]  # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 10, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        try:
            decoded = enc.decode(tokens)
        except BaseException as e:
            decoded = f"error: {e}"

        # print(f"rank {'0'} sample {i}: {decoded}")
        print("######### Generated code ##########")
        print(decoded)
        print("###################################")
    torch._dynamo.config.suppress_errors = False


data_dir = os.path.join('../data', "pythoncode")
if ddp:
    train_loader = DataLoaderLite(data_root=data_dir, B=batch_size, T=block_size, process_rank=ddp_rank,
                                  num_processes=ddp_world_size, split="train", device=device)
    val_loader = DataLoaderLite(data_root=data_dir, B=batch_size, T=block_size, process_rank=ddp_rank,
                                num_processes=ddp_world_size, split="val", device=device)
else:
    train_loader = DataLoaderLite(data_root=data_dir, B=batch_size, T=block_size, process_rank=0,
                                  num_processes=1, split="train", device=device)
    val_loader = DataLoaderLite(data_root=data_dir, B=batch_size, T=block_size, process_rank=0,
                                num_processes=1, split="val", device=device)


def get_batch(split):
    if split == 'train':
        return train_loader.next_batch()
    else:
        return val_loader.next_batch()


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

running_mfu = -1.0
raw_model = model if not ddp else model.module

local_iter_num = 0  # number of iterations in the lifetime of this process
X, Y = get_batch('train')  # fetch the very first batch
iter_num = 0

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    is_last_iter = iter_num == max_iters - 1
    # evaluate the loss on train/val sets and write checkpoints
    if ( iter_num % eval_interval == 0 or is_last_iter) and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
            })


        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if (iter_num > 0 and iter_num % save_interval == 0) or is_last_iter:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{str(iter_num)}.pt'))
    if iter_num == 0 and eval_only:
        for i in range(5):
            generate()
        break

    t0 = time.time()
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps  # scale the loss
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # time.sleep(10)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    if 'cuda' in device:
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

        tokens_processed = train_loader.B * train_loader.T * gradient_accumulation_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}, "
              f"tok/sec: {tokens_per_sec :.2f}")
        # once in a while generate from the model (except step 0, which is noise)

    if iter_num % (eval_interval*5) == 0 and master_process:
        generate()
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
if ddp:
    destroy_process_group()