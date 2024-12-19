import os
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast


# Utility 1: Train and Save a Tokenizer
def train_and_save_tokenizer(file_path, tokenizer_dir="tokenizer_output", vocab_size=50257):
    """
    Trains a Byte Pair Encoding (BPE) tokenizer on the given dataset and saves it.

    Args:
        file_path (str): Path to the text file containing the training data.
        tokenizer_dir (str): Directory to save the trained tokenizer.
        vocab_size (int): Desired vocabulary size for the tokenizer.

    Returns:
        GPT2TokenizerFast: A tokenizer compatible with GPT-2.
        int: The actual vocabulary size.
    """
    # Initialize BPE Tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Define Trainer with special tokens
    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], vocab_size=vocab_size)

    # Train the tokenizer
    print("Training tokenizer...")
    tokenizer.train([file_path], trainer)

    # Save the tokenizer
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")
    tokenizer.save(tokenizer_file)

    # Wrap the tokenizer for GPT-2 compatibility
    print("Wrapping tokenizer for GPT-2 compatibility...")
    gpt2_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_tokenizer.save_pretrained(tokenizer_dir)

    vocab_size = gpt2_tokenizer.vocab_size
    print(f"Tokenizer trained and saved to {tokenizer_dir}. Vocab size: {vocab_size}")
    return gpt2_tokenizer, vocab_size


# Utility 2: Custom Dataset Class
class TextDataset(Dataset):
    """
    A PyTorch Dataset for loading and tokenizing text data.

    Args:
        file_path (str): Path to the text file containing the training data.
        tokenizer (GPT2TokenizerFast): The tokenizer to encode the data.
        block_size (int): The maximum sequence length.
    """
    def __init__(self, file_path, tokenizer, block_size=128):
        self.examples = []

        print("Loading and tokenizing dataset...")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize and split the text into chunks
        tokenized_text = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            chunk = tokenized_text[i:i + block_size]
            self.examples.append(chunk)

        print(f"Dataset loaded. Total examples: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# Utility 3: Model Setup
def setup_model(vocab_size, n_layer=4, n_head=12, n_embd=512):
    """
    Initializes a GPT-2 model with the specified vocabulary size and architecture.

    Args:
        vocab_size (int): Vocabulary size for the tokenizer.
        n_layer (int): Number of transformer layers.
        n_head (int): Number of attention heads.
        n_embd (int): Dimension of the model's embedding layer.

    Returns:
        GPT2LMHeadModel: A GPT-2 language model ready for training.
    """
    print("Setting up GPT-2 model...")
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=1024,
        n_ctx=1024,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head
    )
    model = GPT2LMHeadModel(config)
    print("GPT-2 model initialized.")
    return model
