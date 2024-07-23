import json
import os
import urllib
import tensorflow as tf # type: ignore

#code
def download_and_load_file(file_path, url):

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r") as file:
        data = json.load(file)

    return data


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

file_path = "/root/Speech_diarisation/GPT/GPT/Datasets/GPT-Generated-Dataset-fixed-with-output.json"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"

data = download_and_load_file(file_path, url)
# print("Number of entries:", len(data))

train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)   # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]

val_data = data[train_portion + test_portion:]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))


import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

    # print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

# Explain index masking

targets = torch.tensor([0, 1])
inputs = torch.tensor(
    [[-1., 1.],
     [-0.5, 1.5]]
)

torch.nn.functional.cross_entropy(inputs, targets)

targets = torch.tensor([0, 1, 1])
inputs = torch.tensor(
    [[-1., 1.],
     [-0.5, 1.5],
     [-0.5, 1.5]]
)
torch.nn.functional.cross_entropy(inputs, targets)

targets = torch.tensor([0, 1, -100])
inputs = torch.tensor(
    [[-1., 1.],
     [-0.5, 1.5],
     [-0.5, 1.5]]
)
torch.nn.functional.cross_entropy(inputs, targets)

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []
    for item in batch:
        # Add an <|endoftext|> token
        item += [pad_token_id]
        # Pad sequences to max_length
        padded = item + [pad_token_id] * (batch_max_length - len(item))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


inputs_1 = [0, 1, 2, 3, 4, 5, 6]
inputs_2 = [7, 8, 9]

batch = (
    inputs_1,
    inputs_2
)

custom_collate_fn(batch)

from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

customized_collate_fn = partial(custom_collate_fn, device=device)

from torch.utils.data import DataLoader


num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False
)

# loading data from gpt we need to replace this with our custom trained generative fill GPT. 
from gpt_download import download_and_load_gpt2
from gpt_download import load_gpt2_params_from_tf_ckpt
from previous_chapters import GPTModel, load_weights_into_gpt


BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True,      # Query-key-value bias
    "emb_dim": 768,          # Embedding dimension (adjust based on your model)
    "n_heads": 12,           # Number of attention heads (adjust based on your model)
    "n_layers": 12   
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_args = {
    "vocab_size": BASE_CONFIG["vocab_size"],
    "block_size": BASE_CONFIG["context_length"],
    "dropout": BASE_CONFIG["drop_rate"],
    "bias": BASE_CONFIG["qkv_bias"],
    "n_embd": BASE_CONFIG["emb_dim"],
    "n_layer": BASE_CONFIG["n_layers"],
    "n_head": BASE_CONFIG["n_heads"]
}

# def save_checkpoint(model, optimizer, epoch, loss, out_dir, model_args):
#     checkpoint = {
#         'model': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#         'epoch': epoch,
#         'loss': loss,
#         'model_args': model_args  # Include model_args here
#     }
#     torch.save(checkpoint, os.path.join(out_dir, 'OrigionalGPTx.pth'))



CHOOSE_MODEL = "gpt2-small (124M)"

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2_nxt")



# settings = json.load(open(os.path.join("custom_ckpt", "hparams.json")))
# params = load_gpt2_params_from_tf_ckpt(tf.train.latest_checkpoint("custom_ckpt"), settings)


model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

from previous_chapters import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

torch.manual_seed(123)

token_ids = generate(
    model=model,
    idx=text_to_token_ids(format_input(val_data[0]), tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
)


# end of temporary gpt 

# finetuning GPT on natural language dataset-----------------------------------------------------------

from previous_chapters import (
    calc_loss_loader,
    train_model_simple
)

model.to(device)

torch.manual_seed(123)  # For reproducibility due to the shuffling in the data loader

with torch.no_grad():  # Disable gradient tracking for efficiency because we are not training, yet
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)

import time

start_time = time.time()

torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

num_epochs = 3

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

# save_checkpoint(model, optimizer, num_epochs, val_losses[-1], "saveCP", model_args)


end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

from previous_chapters import plot_losses

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

def extract_response(response):
    return response[response.find("\n### Response")+len("\n### Response:")+1:]
torch.manual_seed(123)

for entry in test_data[:3]:
    print("THIS IS ENTRY: ", entry)

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    response = token_ids_to_text(token_ids, tokenizer)
    response_text = extract_response(response)

    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")

from tqdm import tqdm

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    response = token_ids_to_text(token_ids, tokenizer)
    response_text = extract_response(response)

    test_data[i]["model_response"] = response_text


with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)  # "indent" for pretty-printing

import re

file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")
