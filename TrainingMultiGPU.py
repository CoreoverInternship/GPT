import torch
import time
from tqdm import tqdm
import json
import re
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import tiktoken  # Import tiktoken for tokenization
from gpt_download import download_and_load_gpt2, load_gpt2_params_from_tf_ckpt
from AboutModel import GPTModel, load_weights_into_gpt
from AboutModel import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
    calc_loss_batch,
    evaluate_model,
    # format_input,
    generate_and_print_sample
)
import accelerate
accelerator = accelerate.Accelerator()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True,        # Query-key-value bias
    "emb_dim": 768,          # Embedding dimension (adjust based on your model)
    "n_heads": 12,           # Number of attention heads (adjust based on your model)
    "n_layers": 12   
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12}
}

model_args = {
    "vocab_size": BASE_CONFIG["vocab_size"],
    "context_length": BASE_CONFIG["context_length"],
    "drop_rate": BASE_CONFIG["drop_rate"],
    "qkv_bias": BASE_CONFIG["qkv_bias"],
    "emb_dim": BASE_CONFIG["emb_dim"],
    "n_layers": BASE_CONFIG["n_layers"],
    "n_heads": BASE_CONFIG["n_heads"]
}

CHOOSE_MODEL = "gpt2-small (124M)"

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

checkpoint_path = "/root/Speech_diarisation/GPT/GPT/Creating-And-Finetuneing-LLM/gpt2-small124M-sft.pth"
tokenizer = tiktoken.get_encoding("gpt2")

model = GPTModel(BASE_CONFIG).to(device)
model.load_state_dict(torch.load(checkpoint_path))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item[0]) for item in batch) + 1

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []
    for item in batch:
        # Add an end token
        item = list(item[0]) + [pad_token_id]
        # Pad sequences to max_length
        padded = item + [pad_token_id] * (batch_max_length - len(item))
        inputs = torch.tensor(padded[:-1], dtype=torch.long)  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:], dtype=torch.long)  # Shift +1 to the right for targets

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

# Define the InstructionDataset class
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                self.tokenizer.encode(full_text)
            )

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        encoded_text = self.encoded_texts[idx]
        return torch.tensor(encoded_text, dtype=torch.long), torch.tensor(encoded_text, dtype=torch.long)

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    if 'input' in entry:  
        input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    else:
         input_text = ""

    return instruction_text + input_text


# Load data from JSON files
def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


knowledge_data = load_data('/root/Speech_diarisation/GPT/GPT/Datasets/help.json')

# Create datasets and data loaders for the knowledge data
knowledge_dataset = InstructionDataset(knowledge_data, tokenizer)
knowledge_loader = DataLoader(knowledge_dataset, batch_size=1, collate_fn=custom_collate_fn, shuffle=True)

# Use Accelerate to prepare the model, optimizer, and data loaders
model, optimizer = accelerator.prepare(model, optimizer)
knowledge_loader = accelerator.prepare(knowledge_loader)

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            
            accelerator.backward(loss)
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen

# Move model to device
model.to(device)

# Set manual seed for reproducibility
torch.manual_seed(123)

# Initial loss calculation without tracking gradients for efficiency
with torch.no_grad():
    train_loss = calc_loss_loader(knowledge_loader, model, device, num_batches=5)

print("Training loss:", train_loss)

# Start timer
start_time = time.time()

# Set manual seed for reproducibility
torch.manual_seed(123)

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

# Define number of epochs
num_epochs = 3

# Fine-tune the model with additional knowledge data
train_losses, val_losses, tokens_seen = train_model_simple(
    model, knowledge_loader, knowledge_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(knowledge_data[0]), tokenizer=tokenizer
)

# End timer
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# Save model state dictionary
file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-GPT-Generated-Dataset.pth"
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")
