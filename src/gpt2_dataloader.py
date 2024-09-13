import os

import numpy as np
import torch

# def get_batch(split, data_dir, block_size, batch_size, device, device_type):
#     if split == 'train':
#         data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
#     else:
#         data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
#     y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
#     if device_type == 'cuda':
#         # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
#         x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
#     else:
#         x, y = x.to(device), y.to(device)
#     return x, y

def load_tokens(filename):
    npt = np.load(filename, mmap_mode='r')
    print(filename)
    # npt = npt.astype(np.int16) # np.int16
    # ptt = torch.tensor(npt, dtype=torch.long)
    # return ptt
    return npt

class DataLoaderLite:
    def __init__(self, data_root, B, T, process_rank, num_processes, split, master_process= True, device=None):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.device = device
        assert split in {'train', 'val'}

        # get the shard filenames
        shards = os.listdir(data_root)

        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens_stream = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        # print("start load")
        B, T = self.B, self.T
        buf = self.tokens_stream[self.current_position : self.current_position+B*T+1]
        # print (len(buf))
        buf = buf.astype(np.int64)
        buf = torch.from_numpy(buf)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        # print ("token stream len: ",len(self.tokens_stream))
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens_stream):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens_stream = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank


        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        # print("end load")
        return x, y
