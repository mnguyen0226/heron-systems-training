# from requests.models import encode_multipart_formdata
# from rnn.text_translation.torchtext.all import train
# from rnn.text_translation.torchtext.all import train
import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive, validate_file
import io
import numpy as np

url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ('train.de.gz', 'train.en.gz')
val_urls = ('val.de.gz', 'val.en.gz')
test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

# Gets Tokenizer from Spacy
de_tokenizer = get_tokenizer('spacy', language='de')
en_tokenizer = get_tokenizer('spacy', language='en')

def build_vocab(filepath, tokenizer):
  """ Buids vocabs from tokenizer from filepath """
  counter = Counter() # counter create a dictionary of key and pair
  with io.open(filepath, encoding = "utf8") as f: # need encoding to read in the filepath name
    for string_ in f: # Read in each line in the file 
      # print(f"TESTING + {string_}")
      counter.update(tokenizer(string_)) # for every line read in file, update counter
  return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>']) # list of special paddings before the words

# Builds vocab from tokenizer and training file path => for data_process
de_vocab = build_vocab(train_filepaths[0], de_tokenizer) # german vocab
en_vocab = build_vocab(train_filepaths[1], en_tokenizer) # english vocab

def data_process(filepaths):
  """ Iterates thru vocab and convert to tensor for pytorch """
  raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
  raw_en_iter = iter(io.open(filepaths[0], encoding='utf8'))
  data = []
  for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
    de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de)], dtype=torch.long) # torch.long = 64-bit signed interger
    en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)], dtype=torch.long)

    data.append((de_tensor_, en_tensor_))
  return data

# Creates training, testing, validating dataset => Cross Entropy Loss calculation
train_data = data_process(train_filepaths)
val_data = data_process(val_filepaths)
test_data = data_process(test_filepaths)


def test():
  """ Tests code """
  print("Running")
  print((train_data)[0])

if __name__ == "__main__":
  test()