from flask import Flask, render_template, request
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast
import numpy as np


app = Flask(__name__)

class BERT_Arch(nn.Module):
    def __init__(self, bert):  
      super(BERT_Arch, self).__init__()
      self.bert = bert   
      self.dropout = nn.Dropout(0.1)            # dropout layer
      self.relu =  nn.ReLU()                    # relu activation function
      self.fc1 = nn.Linear(768,512)             # dense layer 1
      self.fc2 = nn.Linear(512,2)               # dense layer 2 (Output layer)
      self.softmax = nn.LogSoftmax(dim=1)       # softmax activation function
    def forward(self, sent_id, mask):           # define the forward pass  
      cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
                                                # pass the inputs to the model
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)                           # output layer
      x = self.softmax(x)                       # apply softmax activation
      return x
    
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BERT_Arch(bert)


@app.route('/about', methods=['GET'])
def about():
  return render_template("about.html")

@app.route('/', methods=['GET'])
def index_a():
  #news=request.form['input']
  return render_template("index_a.html")

@app.route('/generated', methods=['POST'])
def generate():
  news=request.form['input']
  path = '/home/sunil/WebScraper/coppellisd/schools/c3_new_model_weights.pt'
  model.load_state_dict(torch.load(path))
  MAX_LENGHT = 15
  tokens_unseen = tokenizer.batch_encode_plus(
        [news],
        max_length = MAX_LENGHT,
        pad_to_max_length=True,
        truncation=True
    )
  unseen_seq = torch.tensor(tokens_unseen['input_ids'])
  unseen_mask = torch.tensor(tokens_unseen['attention_mask'])

  with torch.no_grad():
    preds = model(unseen_seq, unseen_mask)
    preds = preds.detach().cpu().numpy()
  preds = np.argmax(preds, axis = 1)
  print(news)
  return render_template("index_a.html", output = preds)

@app.route('/team', methods=['GET'])
def team():
  return render_template("team.html")

@app.route('/chatbot', methods=['GET'])
def chatbot():
  return render_template("chatbot.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
