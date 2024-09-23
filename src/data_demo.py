import os

import pandas as pd
import tiktoken

# from gpt2_dataloader import DataLoaderLite, load_tokens
#
# enc = tiktoken.get_encoding("gpt2")
# eot = enc._special_tokens['<|endoftext|>']  # end of text token
#
# data = DataLoaderLite(data_root='../data/pythoncode', B=1, T=1024*10, process_rank=0, num_processes=1, split='val', master_process=True, device='cpu')
#
# for i in range(1):
#     x, y = data.next_batch()
#     print(enc.decode(x[0].tolist()))
#
# # npt = load_tokens("data/pythoncode_train_000000.npy")

# ------------------------------------------------------------

# from datasets import load_dataset, Dataset # pip install datasets
# # download the dataset
# ds = load_dataset("bigcode/the-stack", split='train', data_dir="data/python", streaming=True)
# subset = []
# for idx, sample in enumerate(ds):
#     if idx < 10000:
#         subset.append(sample)
#     else:
#         break
#
#
# # Convert the subset to Hugging Face Dataset
# fw = Dataset.from_dict({key: [example[key] for example in subset] for key in subset[0].keys()})
#
# # fw.set_format(type='pandas')
# df = fw.to_pandas()
# # save datafrane
# df.to_pickle('data/pythoncode3.pkl')
# --- ---------------------------------------------
# # loop through the dataset, get a map of line length to file length
long_count, total_count = 0, 0
# for doc in fw:
#     total_count += 1
#     print(
#         f"avg_line_len: {doc['avg_line_length']}, max_line_len: {doc['max_line_length']}, "
#         f"content_len: {len(doc['content'])}, alpha_num: {doc['alphanum_fraction']}")
#     if len(doc['content']) > 20000:
#         print(doc['content'])
#         print("#######################################################")
#         print("-------------------------------------------------------")
#         print()
#         long_count += 1
#     if total_count > 10000:
#         print(f"long_count: {long_count}, total_count: {total_count}")
#         break

# ------------------------------------------------------------
df = pd.read_pickle('data/pythoncode3.pkl')
# df = df[(df['max_line_length'] > 1000) |  (df['content'].str.len()/df['avg_line_length'] > 2000) |  (df['alphanum_fraction']<0.25)]
# df = df[(df['content'].str.len()/df['avg_line_length'] > 2000)]
# print (df['content'].str.len()/df['avg_line_length'])
# print (len(df))

df = df[(df['max_line_length'] > 250)]
print (df['max_line_length'])
print (len(df))
