import pandas as pd
import string
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import DistilBertModel, DistilBertTokenizer

device = torch.device("cuda")

df = pd.read_csv("good.csv")

text = df["text"]
sentiment = df["sentiment"]

plt.hist(sentiment)
#plt.show()

vocab_ids = set()
vocab_ids.add("[CLS]")
vocab_ids.add("[UNK]")
vocab_ids.add("[PAD]")
max_size = 0
for tweet in list(text):
    tweet = tweet.translate(str.maketrans('', '', string.punctuation)).replace("¿", "").replace("¡", "").lower()
    for word in tweet.split(" "):
      if len(tweet.split(" ")) > max_size:
        max_size = len(tweet.split(" "))
      vocab_ids.add(word)

def save_vocab():
    ids = []
    words = []
    for id, word in enumerate(list(vocab_ids)[1:]):
        ids.append(id)
        words.append(word)
    df_aux = pd.DataFrame(columns=["ids", "words"])
    df_aux["ids"] = ids
    df_aux["words"] = words
    df_aux.to_csv("words_id.csv", index=False)

word_ids = {key: id for id, key in enumerate(list(vocab_ids)[1:])}

def encode_text(tweet, word_ids):
    tweet = tweet.translate(str.maketrans('', '', string.punctuation)).replace("¿", "").replace("¡", "").lower()
    out = [word_ids["[CLS]"]]
    for word in tweet.split(" "):
      if word != "":
        try:
          out.append(word_ids[word])
        except:
          out.append(word_ids["[UNK]"])

    if len(out) < max_size + 1:
      out = out + [word_ids["[PAD]"]] * (max_size + 1 - len(out))
    return torch.tensor(out).to(device)

def decode_text(tweet, word_ids):
  tweet = tweet.flatten()
  out = ""
  for id in tweet:
    value = list(word_ids.keys())[id]
    if value == "[PAD]":
      return out
    out = out + " " + list(word_ids.keys())[id]
  return out

class MyData(Dataset):
  def __init__(self, text, sentiment, word_ids):
    super().__init__()
    self.text = text
    self.sentiment = sentiment
    self.word_ids = word_ids
  def __len__(self):
    return len(self.text)

  def __getitem__(self, index):
    tweet = encode_text(self.text[index], self.word_ids)
    label = self.sentiment[index]
    match label:
      case "positive":
        label = torch.tensor(0).to(device)
      case "neutral":
        label = torch.tensor(1).to(device)
      case "negative":
        label = torch.tensor(2).to(device)
    return tweet, label

data = MyData(list(text), list(sentiment), word_ids)
train_data, test_data = torch.utils.data.random_split(data, [0.8, 0.2])

batch_size = 150
train_set = DataLoader(
    train_data,
    batch_size,
    True
)
test_set = DataLoader(
    test_data,
    batch_size,
    True
)

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.attention = nn.MultiheadAttention(512, 8, 0.1, batch_first=True)
    self.fc = nn.Sequential(
        nn.BatchNorm1d(max_size + 1),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512)
    )
    self.norm = nn.BatchNorm1d(max_size + 1)

  def forward(self, x):
    out, _ = self.attention(x, x, x)
    out = out + x
    x = self.fc(out)
    return self.norm(x + out)

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = nn.Embedding(len(word_ids), 512)
    self.encoders = nn.ModuleList([Encoder() for _ in range(8)])
    self.out = nn.Sequential(
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, 3)
    )
    self.position = nn.Parameter(torch.rand((1, max_size + 1, 512)))

  def forward(self, x):
    x = self.emb(x)
    x = x + self.position
    for encoder in self.encoders:
      x = encoder(x)
    x = self.out(x[:, 0, :])
    return x

model = Model().to(device)
optim = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
loss_f = nn.CrossEntropyLoss()

def train(text, label):
  optim.zero_grad()

  out = model(text)
  loss = loss_f(out, label)

  loss.backward()
  optim.step()

  return loss

def test_epoch():
  with torch.no_grad():
    trues = 0
    falses = 0
    for i, (text, label) in enumerate(test_set):
      out = model(text)
      trues = trues + (torch.argmax(out, dim=1) == label).sum()
      falses = falses + (torch.argmax(out, dim=1) != label).sum()
    print(trues/(trues + falses))
    return trues/(trues + falses)


def train_model():
    save_vocab()
    epochs = 10
    for epoch in range(epochs):
      epoch_loss = 0.0
      for i, (text, label) in tqdm(enumerate(train_set), total=len(train_set)):
        epoch_loss += train(text, label)
      print(f"Epoch: {epoch} ------ Loss: {epoch_loss.item()/i}")
      test_epoch()

    torch.save(model, "sentiment.pt")

def test():
    df = pd.read_csv("words_id.csv")
    word_ids = dict(zip(df["words"], df["ids"]))
    with torch.no_grad():
        model = torch.load("sentiment.pt")
        query = str(input("Text here: "))
        model.eval()
        while query != "":
            query = encode_text(query, word_ids)
            out = model(query[None, :])
            max_value = torch.argmax(out, dim=1)
            if max_value == 0:
                print("Positive")
            elif max_value == 1:
                print("neutral")
            else:
                print("Negative")

            query = str(input("Text here: "))


test()