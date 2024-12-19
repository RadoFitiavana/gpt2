import wandb
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from tqdm import tqdm
from utils import TextDataset, setup_model, train_and_save_tokenizer

# Initialize Weights & Biases
wandb.init(project="gpt2-training", name="gpt2_from_scratch")

# Configurations
DATASET_PATH = "/content/drive/MyDrive/rakitra.txt"
# TOKENIZER_DIR = "tokenizer_output"
BATCH_SIZE = 4
BLOCK_SIZE = 128
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
SAVE_DIR = "gpt2_model"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Train tokenizer dynamically and get vocab size
tokenizer, vocab_size = train_and_save_tokenizer(DATASET_PATH)

# Step 2: Prepare dataset and dataloader
train_dataset = TextDataset(DATASET_PATH, tokenizer, block_size=BLOCK_SIZE)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Step 3: Initialize GPT-2 model
model = setup_model(vocab_size=vocab_size)
model.to(device)

# Step 4: Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * NUM_EPOCHS
)

# Step 5: Training Loop
model.train()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for batch in progress_bar:
        batch = batch.to(device)
        
        # Shift inputs for causal language modeling
        outputs = model(input_ids=batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        # Log loss
        epoch_loss += loss.item()
        wandb.log({"batch_loss": loss.item()})

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}")
    wandb.log({f"epoch_loss": avg_epoch_loss})

# Save model and tokenizer
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Model and tokenizer saved to {SAVE_DIR}")

wandb.finish()
