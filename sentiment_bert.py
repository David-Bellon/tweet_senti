import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer

device = torch.device("cuda")

df = pd.read_csv("good.csv")

text = df["text"]
sentiment = df["sentiment"]

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")

class MyData(Dataset):
  def __init__(self, text, sentiment):
    super().__init__()
    self.text = tokenizer(list(text), padding="longest", truncation=True)
    self.sentiment = sentiment
  def __len__(self):
    return len(self.sentiment)

  def __getitem__(self, index):
    ids = torch.tensor(self.text["input_ids"][index]).to(device)
    mask = torch.tensor(self.text["attention_mask"][index]).to(device)
    label = self.sentiment[index]
    match label:
      case "positive":
        label = torch.tensor(0).to(device)
      case "neutral":
        label = torch.tensor(1).to(device)
      case "negative":
        label = torch.tensor(2).to(device)
    return ids, mask, label

data = MyData(text, list(sentiment))
train_data, test_data = torch.utils.data.random_split(data, [0.8, 0.2])

batch_size = 64
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

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.bert = distilbert
    #for p in self.bert.parameters():
        #p.requires_grad = False

    self.out = nn.Sequential(
        nn.Linear(768, 512),
        nn.GELU(),
        nn.LayerNorm(512),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.LayerNorm(256),
        nn.Dropout(0.4),
        nn.Linear(256, 3),
    )

  def forward(self, input_ids, attention_mask):
    x = self.bert(input_ids, attention_mask)
    out = x.last_hidden_state[:, 0, :]
    out = self.out(out)
    return out

model = Model().to(device)
optim = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
loss_f = nn.CrossEntropyLoss()

def train(input_ids, attention_mask, label):
  optim.zero_grad()

  out = model(input_ids, attention_mask)
  loss = loss_f(out, label)

  loss.backward()
  optim.step()

  return loss

def test_epoch():
  with torch.no_grad():
    trues = 0
    falses = 0
    for i, (input_ids, attention_mask, label) in enumerate(test_set):
      out = model(input_ids, attention_mask)
      trues = trues + (torch.argmax(out, dim=1) == label).sum()
      falses = falses + (torch.argmax(out, dim=1) != label).sum()
    print(trues/(trues + falses))
    return trues/(trues + falses)


def train_model():
    epochs = 10
    for epoch in range(epochs):
      epoch_loss = 0.0
      for i, (input_ids, attention_mask, label) in tqdm(enumerate(train_set), total=len(train_set)):
        epoch_loss += train(input_ids, attention_mask, label)
      print(f"Epoch: {epoch} ------ Loss: {epoch_loss.item()/i}")
      test_epoch()

    torch.save(model, "sentiment_fine.pt")

def test():
    with torch.no_grad():
        model = torch.load("sentiment_fine.pt")
        query = str(input("Text here: "))
        model.eval()
        while query != "":
            out = tokenizer(query, truncation=True)
            ids = torch.tensor([out["input_ids"]]).to(device)
            mask = torch.tensor([out["attention_mask"]]).to(device)
            out = model(ids, mask)
            max_value = torch.argmax(out, dim=1)
            if max_value == 0:
                print("Positive")
            elif max_value == 1:
                print("neutral")
            else:
                print("Negative")

            query = str(input("Text here: "))


test()