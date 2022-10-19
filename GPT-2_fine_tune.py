from argparse import ArgumentParser
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead
from torch.optim import AdamW


os.environ["TOKENIZERS_PARALLELISM"] = "true"

model_name = "skt/kogpt2-base-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "sep_token": "<seq>"
    }
SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>", "<seq>"]
tokenizer.add_special_tokens(SPECIAL_TOKENS)

model = AutoModelWithLMHead.from_pretrained(
    model_name
).cuda()

model.resize_token_embeddings(len(tokenizer)) 

parser = ArgumentParser()
parser.add_argument("--epoch", default=30, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--sep_token", default=tokenizer.sep_token, type=str)
parser.add_argument("--bos_token", default=tokenizer.bos_token, type=str)
parser.add_argument("--eos_token", default=tokenizer.eos_token, type=str)
args = parser.parse_args()

# wandb.init(project="mobot", name=f"mobot-{model_name}")
train_data = pd.read_csv("data/ChatbotData.csv", delimiter=",")
train_data = train_data[:10000]
train_text, train_labels = (
    train_data["Q"].values,
    train_data["A"].values,
)
dataset = [
    {"data": "<usr>" + t + "<sys>" + l + str(tokenizer.eos_token), "label": l}
    for t, l in zip(train_text, train_labels)
]
train_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

eval_data = pd.read_csv("data/ChatbotData.csv", delimiter=",")
eval_data = eval_data[10000:]

eval_text, eval_labels = (
    eval_data["Q"].values,
    eval_data["A"].values,
)

dataset = [
    {"data": "<usr>" + t + "<sys>" + l + str(tokenizer.eos_token), "label": l}
    for t, l in zip(eval_text, eval_labels)
]
eval_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

optimizer = AdamW(params=model.parameters(),
    lr=3e-5, weight_decay=3e-7
)

epochs = 10
for epoch in range(epochs):
    model.train()
    for train in tqdm(train_loader):
        optimizer.zero_grad()
        text, label = train["data"], train["label"]
        text_tokens = tokenizer(
            text,
            return_tensors="pt",
            max_length=80,
            truncation=True,
            padding=True,
        )

        input_ids = text_tokens.input_ids.cuda()
        # print(input_ids)
        # print(tokenizer.convert_ids_to_tokens(input_ids[0]))         ## 토크나이저가 어떻게 자르고 어떻게 구성되있는지 보기. 
        attention_mask = text_tokens.attention_mask.cuda()            ## <pad>토큰들은 자동으로 masking되어서 실제 단어만 들어있는것만 학습 진행. 

        output = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        loss = output.loss
        loss.backward()        
        optimizer.step()

    print({"loss": loss.item()})

    with torch.no_grad():
        model.eval()
        for eval in tqdm(eval_loader):
            eval_text, eval_label = eval["data"], eval["label"]
            eval_text_tokens = tokenizer(
                eval_text,
                return_tensors="pt",
                max_length=80,
                truncation=True,
                padding=True,
            )

            input_ids = eval_text_tokens.input_ids.cuda()
            attention_mask = eval_text_tokens.attention_mask.cuda()

            eval_out = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            eval_loss = eval_out.loss

        print({"eval_loss": eval_loss.item()})          
        print({"epoch": epoch+1})
        torch.save(model.state_dict(), f"model_save/GPT-2_fintuing-{epoch+1}.pt")