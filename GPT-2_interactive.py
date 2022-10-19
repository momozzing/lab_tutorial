"""
python interactive.py
"""
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer

ckpt_name = "model_save/GPT-2_fintuing-30.pt"
model = AutoModelWithLMHead.from_pretrained("skt/kogpt2-base-v2")
tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "sep_token": "<seq>"
    }
SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>", "<seq>"]
tokenizer.add_special_tokens(SPECIAL_TOKENS)
model.resize_token_embeddings(len(tokenizer)) 

model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()

with torch.no_grad():
    while True:
        t = input("\nUser: ")
        tokens = tokenizer(
            "<usr>" + t,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=80
        )

        input_ids = tokens.input_ids.cuda()
        attention_mask = tokens.attention_mask.cuda()
        sample_output = model.generate(
            input_ids, 
            max_length=80, 
            num_beams=5, 
            early_stopping=True
        )
        gen = sample_output[0]
        print("System: " + tokenizer.decode(gen[len(input_ids[0]):-1], skip_special_tokens=True))