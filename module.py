import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizerFast as BertTokenizer, BertModel, BertJapaneseTokenizer
import unicodedata,neologdn
import re, os
#import fugashi, ipadic

os.environ["HF_DATASETS_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

TOKENIZER_PATH = "./tokenizer/"
MODEL_PATH = './bert_base_tohoku/'
TRAINED_WEIGHTS = "./trained_weights/epoch9_step1990_bert512.pth"
LABEL_COLUMNS = ['topic-news','sports-watch','smax', 'peachy','movie-enter','livedoor-homme','kaden-channel','it-life-hack','dokujo-tsushin']
MAX_TOKEN_COUNT = 512

# 分類層の追加
class TextsTagger(nn.Module):
  def __init__(self, model_path,n_classes: int, LABEL=None):
    super().__init__()
    self.bert = BertModel.from_pretrained(model_path, return_dict=True, local_files_only = True)
    self.classifier = nn.Linear(768*4, n_classes)
    self.labels = LABEL
    self.criterion = nn.BCELoss()
    
    for param in self.bert.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask, labels=None):
    output = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    #最終4層の重みを結合
    output = self.classifier(torch.cat([output.hidden_states[-1*i][:, 0, :].view(-1, 768) for i in range(1, 5)],dim=1))
    output = torch.sigmoid(output)
    return output

def normalize_text(text):
    text = text.replace("\n","").replace("\t","")
    text = unicodedata.normalize('NFKC', text)
    text = neologdn.normalize(text)
    text = re.sub(r"[a-zA-Z0-9_]+", "N",text)
    text = re.sub(r"[ -/:-@\[-~]+", "",text)
    #text = re.sub(r"[ -~]+", "",text)
    return text

def predict(texts):
    #トークナイザ読み込み
    tokenizer = BertJapaneseTokenizer.from_pretrained(TOKENIZER_PATH)
    #モデル読み込み
    model = TextsTagger(MODEL_PATH, n_classes = len(LABEL_COLUMNS), LABEL = LABEL_COLUMNS)
    #gpuかcpuの選択
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #学習済の重み読み込み
    state_dict = torch.load(TRAINED_WEIGHTS, map_location = device)
    model.load_state_dict(state_dict)
    
    texts = normalize_text(texts)
    encoding = tokenizer.encode_plus(texts,add_special_tokens = True,
                                        max_length = MAX_TOKEN_COUNT,
                                        return_token_type_ids = False,
                                        padding = "max_length", truncation = True,
                                        return_attention_mask = True, return_tensors = "pt",)
    prediction = model(encoding["input_ids"].flatten().unsqueeze(dim=0).to(device), encoding["attention_mask"].flatten().unsqueeze(dim=0).to(device))
    prediction = prediction.flatten()
    THRESHOLD = 0.5

    outputs = []
    for label, pred in zip(LABEL_COLUMNS, prediction):
        if pred< THRESHOLD:
            continue
        outputs.append(label)
        #print(f"{label}:{pred}")
    return outputs

