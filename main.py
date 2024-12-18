import torch
from transformers import (GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, Trainer, 
                          TrainingArguments, DataCollatorForLanguageModeling)
import wandb
from torch.utils.data import Dataset
from tqdm.auto import tqdm  # Import tqdm for progress bars

# Initialize Weights & Biases (wandb) for tracking experiments
wandb.init(project="gpt2-from-scratch", name="train-small-gpt2")

# Paths to your data
train_data_path = "/content/drive/MyDrive/rakitra.txt"  # Path to your cleaned training text file
output_dir = "./gpt2-small-trained"  # Directory to save model checkpoints

# Step 1: Load GPT-2 Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token matches EOS token

# Step 2: Create Configuration for Small GPT-2
config = GPT2Config.from_pretrained("gpt2", vocab_size=len(tokenizer))

# Step 3: Initialize GPT-2 Model
model = GPT2LMHeadModel(config)

# Step 4: Custom Dataset Class
class CustomTextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        """
        Custom dataset that tokenizes the text and splits it into blocks of a given size.
        Args:
            file_path (str): Path to the text file.
            tokenizer (transformers.GPT2Tokenizer): Tokenizer for converting text to tokens.
            block_size (int): Length of text chunks (sequence length).
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Tokenize the entire text and split it into blocks of `block_size`
        self.examples = tokenizer(
            text, 
            truncation=True, 
            padding=False, 
            max_length=block_size, 
            return_tensors="pt", 
            add_special_tokens=True
        )["input_ids"]  # Take only the input_ids, which are the tokenized texts.

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {"input_ids": self.examples[idx]}

train_dataset = CustomTextDataset(train_data_path, tokenizer)

# Step 5: Data Collator for Padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Not using Masked Language Modeling
)

# Step 6: Define Progress Bar and Logging Callbacks for Tracking
class ProgressBarCallback(Trainer.Callback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        # Initialize the progress bar at the beginning of each epoch
        self.epoch_progress = tqdm(total=state.max_steps // args.num_train_epochs, 
                                     desc=f"Epoch {state.epoch + 1}",
                                     position=0, leave=True)

    def on_step_end(self, args, state, control, **kwargs):
        # Update progress bar at each step
        if state.global_step % args.logging_steps == 0:
            self.epoch_progress.set_postfix(loss=state.log_history[-1]["loss"], refresh=True)
            self.epoch_progress.update(1)

    def on_epoch_end(self, args, state, control, **kwargs):
        # End progress bar and log metrics to wandb at the end of each epoch
        self.epoch_progress.close()
        print(f"\nEpoch {state.epoch + 1} completed. Saving model...")
        wandb.log({"epoch": state.epoch + 1, "training_loss": state.log_history[-1]["loss"]})

# Step 7: Training Arguments Configuration
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Adjust batch size for memory constraints
    save_steps=10_000,
    save_total_limit=2,  # Keep only the 2 most recent checkpoints
    logging_dir="./logs",  # Directory for TensorBoard logs
    logging_steps=500,
    evaluation_strategy="steps",
    eval_steps=2_000,
    do_train=True,
    do_eval=False,  # Set to True if you have a validation set
    fp16=True,  # Mixed precision training for faster training (if supported)
    report_to="wandb",  # Log metrics to Weights & Biases
    learning_rate=5e-4,
    warmup_steps=500,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Automatically use GPU if available
)

# Step 8: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    callbacks=[ProgressBarCallback()]  # Add progress bar callback for training feedback
)

# Step 9: Start Training
trainer.train()

# Step 10: Save the trained model
trainer.save_model(output_dir)

# Step 11: Finish Weights & Biases tracking
wandb.finish()

# Final Output
print(f"Model training complete. Checkpoints saved at {output_dir}")
