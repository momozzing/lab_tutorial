"""
python inference.py
"""
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from kobert_tokenizer import KoBERTTokenizer


ckpt_name = "model_save/BERT_fintuing_NSMC-1.pt"
model = AutoModelForSequenceClassification.from_pretrained('skt/kobert-base-v1')
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')


model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()

eval_data = pd.read_csv("data/ratings_test.txt", delimiter="\t")
eval_data = eval_data.dropna(axis=0)
# eval_data = eval_data[:300]
eval_text, eval_labels = (
    eval_data["document"].values,
    eval_data["label"].values,
)

dataset = [
    {"data": tokenizer.cls_token + t, "label": l}
    for t, l in zip(eval_text, eval_labels)
]

acc = 0
for data in tqdm(dataset):
    text, label = data["data"], data["label"]
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    input_ids = tokens.input_ids.cuda()
    attention_mask = tokens.attention_mask.cuda()
    # label = torch.tensor(label).cuda()

    output = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    classification_results = output.logits.argmax(-1)
    if classification_results == label:
        acc += 1


print(f"acc: {acc / len(dataset)}")
