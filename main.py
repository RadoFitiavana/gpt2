from transformers import (GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, Trainer, 
                          TrainingArguments, TextDataset, DataCollatorForLanguageModeling, TrainerCallback)
import wandb
from tqdm import tqdm  # Import tqdm for progress bars

# Initialize Weights & Biases
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

# Step 4: Prepare Dataset
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size  # Maximum sequence length
    )
    return dataset

train_dataset = load_dataset(train_data_path, tokenizer)

# Step 5: Data Collator for Padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Not using Masked Language Modeling
)

# Custom Callback for Console Logs and Progress Bar
class ProgressBarCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"\n\nEpoch {state.epoch + 1}/{args.num_train_epochs} starting...")
        self.epoch_progress = tqdm(total=state.max_steps // args.num_train_epochs, 
                                     desc=f"Epoch {state.epoch + 1}",
                                     position=0, 
                                     leave=True)
    
    def on_step_end(self, args, state, control, **kwargs):
        # Update the progress bar every step
        if state.global_step % args.logging_steps == 0:
            self.epoch_progress.set_postfix(loss=state.log_history[-1]["loss"], refresh=True)
            self.epoch_progress.update(1)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\nEpoch {state.epoch + 1} completed. Saving model...")
        self.epoch_progress.close()  # Close the progress bar at the end of the epoch
        wandb.log({"epoch": state.epoch + 1, "training_loss": state.log_history[-1]["loss"]})

# Step 6: Training Arguments
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
    fp16=True,  # Mixed precision training
    report_to="wandb",  # Use Weights & Biases for logging
    learning_rate=5e-4,
    warmup_steps=500
)

# Step 7: Trainer Initialization with Custom Progress Bar Callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    callbacks=[ProgressBarCallback()]  # Adding the progress bar callback
)

# Step 8: Train the Model
trainer.train()

# Save the model
trainer.save_model(output_dir)
wandb.finish()

print(f"Model training complete. Checkpoints saved at {output_dir}")
