import torch
from torch.nn.utils.rnn import pad_sequence # pad a list of variable length Tensors with padding values

# pad sequence stack a list of tensors along a new dim, and pads them to equal length.
# For example, if the input is list of sequences with size L x * and if batch_first is False, and T x B x * otherwise.
# B is batch size. It is equal to the number of elements in sequences. T is length of the longest sequence. L is length of the sequence.

from torch.utils.data import DataLoader

# from rnn.text_translation.torchtext.torchtextpackage.data_preprocessing import de_vocab, en_vocab, train_data, val_data, test_data
from utils.data_preprocessing import de_vocab, en_vocab, train_data, val_data, test_data


# Sets up device
device = ""
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Fields for german:
BATCH_SIZE = 128
PAD_IDX = de_vocab["<pad>"]
BOS_IDX = de_vocab["<bos>"]
EOS_IDX = de_vocab["<eos>"]


from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def generate_batch(data_batch):
  de_batch, en_batch = [], []
  for (de_item, en_item) in data_batch:
    de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
    en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
  de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
  en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
  return de_batch, en_batch

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=generate_batch)

if __name__ == "__main__":
    generate_batch()
