from datasets import load_dataset

NUM_FILE = 10000

# If the dataset is gated/private, make sure you have run huggingface-cli login
ds = load_dataset("bigcode/the-stack-dedup",split='train',data_dir="data/python", streaming=True)

# dump this to a file
with open(f"../data/python_{NUM_FILE}.txt", "w") as f:
    for idx, sample in enumerate(ds):

        if idx< NUM_FILE:
            f.write(sample['content'])
            f.write("\n")
            # print(sample["content"])
        else:
            break
