import torch


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def read_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def get_train_val(data):
    n = int(len(data) * 0.9)
    train_data, val_data = data[:n], data[n:]
    return train_data, val_data


def get_batch(split, train_data, val_data, block_size, batch_size):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i: i + block_size] for i in idx])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in idx])

    device = get_device()
    return x.to(device), y.to(device)


def encoder_decoder(chars):
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[c] for c in l])
    return encode, decode


@torch.no_grad()
def eval_model(model, train_data, val_data, block_size, batch_size):
    model.eval()
    xtrain, ytrain = get_batch('train', train_data, val_data, block_size, batch_size)
    xval, yval = get_batch('val', train_data, val_data, block_size, batch_size)
    _, train_loss = model.forward(xtrain, ytrain)
    _, val_loss = model.forward(xval, yval)
    # print("Train Loss: {:.4f}".format(train_loss), "Val Loss: {:.4f}".format(val_loss))
    model.train()
    return train_loss, val_loss
