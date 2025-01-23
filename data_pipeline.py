import torch
import ijson
import json
import transformers
import numpy as np
import torch.nn as nn
from functools import partial
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def load_clean_data(file_path: str) -> list[dict]:
    """
    Load data from a JSON lines file.

    Args:
        file_path (str): The path to the JSON lines file.

    Returns:
        list: A list of dictionaries, each representing a JSON object.
    """
    try:
        data = []
        with open(file_path, "r") as f:
            for item in ijson.items(f, 'item'):
                data.append(item)
        print("Finished loading data")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"File is not valid JSON: {file_path}")
        return []

def filter_and_clean_data(data: list[dict]) -> list[dict]:
    """
    Filter the data to only include posts where the author's mind was changed.
    Format these posts into a list of dictionaries with the input and response.

    Args:
        data (list): A list of dictionaries containing post data.

    Returns:
        list: A list of dictionaries containing post data where the author's mind was changed with an input and response.
    """
    fine_tune_data = []
    # Loop through all posts
    for post in data:
        try:
            # Check if person's mind was changed
            if post["delta_label"] == True:
                # Save original post as the input
                input_text = post["original_post"]
                # Compile responses from poster that changed the original poster's mind
                author = post['successful_author'].strip()
                responses = [item.get('body') for item in post['comments'] if item.get('author') == post['successful_author']]
                # If the filter returns posts, add them to the fine-tune data
                if len(responses) > 0:
                    fine_tune_data.append({"input": input_text, "response": " ".join(responses)})
            else:
                pass
        except KeyError:
            pass
    
    return fine_tune_data

def format_input(post: dict) -> str:
    """
    Format the input text for the model.

    Args:
        post (dict): A dictionary containing the post data.

    Returns:
        str: The formatted input text.
    """
    instruction_text = (
        "Below is a post from the Change My Views subreddit. "
        "Write a response that you think will change the author's view."
    )
    input_text = (f"\n\n### Input:\n{post['input']}" if post['input'] else "")
    return instruction_text + input_text

def format_output(post):
    """
    Format the output text for the model.

    Args:
        post (dict): A dictionary containing the post data.

    Returns:
        str: The formatted output text.
    """
    return f"\n\n### Response:\n{post['response']}"

def split_data(data: list[dict], split_ratios: dict) -> dict:
    """
    Split the data into training, validation, and test sets.

    Args:
        data (list): A list of dictionaries containing post data.
        split_ratios (dict): A dictionary containing the ratios to split the data.

    Returns:
        dict: A dictionary containing the split data.
    """
    # Calculate the number of posts for each split
    total_posts = len(data)
    train_count = int(total_posts * split_ratios["train"])
    test_count = int(total_posts * split_ratios["test"])
    val_count = total_posts - train_count - test_count

    # Randomly shuffle the data
    np.random.shuffle(data)

    # Split the data into training, validation, and test sets
    train_data = data[:train_count]
    test_data = data[train_count:train_count + test_count]
    val_data = data[train_count + test_count:]

    return train_data, val_data, test_data

class CVMDataset(Dataset):
    """
    Custom Dataset for Change My Views (CVM) data.

    This dataset class is designed to handle the data for the Change My Views subreddit.
    It tokenizes the input text and prepares the input IDs, attention masks, and labels
    for training a language model.

    Attributes:
        data (list): A list of dictionaries containing the text and labels.
        tokenizer (PreTrainedTokenizer): A tokenizer instance from the transformers library.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the tokenized input IDs, attention masks, and labels for a given index.
    """
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the tokenized input IDs, attention masks, and labels for a given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the tokenized input IDs, attention masks, and labels.
        """
        post = self.data[idx]
        instruction_and_input = format_input(post)
        response = format_output(post)
        all_text = instruction_and_input + response

        # Tokenize the full text and the instruction and input separately
        encoded = self.tokenizer(
            all_text,
            return_tensors="pt"
        )
        encoded_inputs = self.tokenizer(
            instruction_and_input,
            return_tensors="pt"
        )

        # Get the index where the instruction and input ends
        instruction_idx = len(encoded_inputs["input_ids"][0])

        # Extract the input IDs and labels
        inputs = encoded["input_ids"].squeeze(0)
        labels = encoded["input_ids"].clone().squeeze(0)

        return {
            "prompt_tokens": inputs[:instruction_idx],
            "target_tokens":labels[instruction_idx:]
        }

def collate_func(batch: list, tokenizer: transformers.PreTrainedTokenizer) -> dict:
    """
    Collate function to process the data for the DataLoader.

    This function pads the input IDs and labels to the maximum sequence length in the batch.

    Args:
        batch (list): A list of dictionaries containing the tokenized input IDs and labels.
        tokenizer (PreTrainedTokenizer): A tokenizer instance from the transformers library.

    Returns:
        dict: A dictionary containing the padded input IDs, attention masks, and labels.
    """
    # Extract prompt and target tokens from each item in the batch
    prompt_tokens = [item['prompt_tokens'] for item in batch]
    target_tokens = [item['target_tokens'] for item in batch]

    # Combine prompt and target tokens into a single sequence for each item
    full_inputs = []
    for a, b in zip(prompt_tokens, target_tokens):
        full_inputs.append(torch.cat((a, b)))

    # Determine max sequence length
    max_length = max(len(ids) for ids in full_inputs)
    
    # Pad sequences to the length of the longest sequence in the batch
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        full_inputs, batch_first=True, padding_value=tokenizer.encode(tokenizer.pad_token)[-1]
    )

    # Create attention masks, setting 1 for non-padding tokens and 0 for padding tokens
    attention_masks = torch.ones_like(padded_input_ids)
    attention_masks[padded_input_ids == tokenizer.pad_token_id] = 0

    # Pad labels similarly to input IDs, but use -100 as the padding value
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        full_inputs, batch_first=True, padding_value=-100
    )

    # Mask the prompt tokens in the labels to ensure the model doesn't compute loss on them
    for i, p in enumerate(prompt_tokens):
        padded_labels[i, :len(p)] = -100
    
    return {
        'input_ids': padded_input_ids,
        'attention_mask': attention_masks,
        'labels': padded_labels
    }

if __name__ == "__main__":
    # Load and clean the dataset
    data = load_clean_data("cmv_posts_cleaned.json")
    data = filter_and_clean_data(data)
    with open ("fine_tune_data.json", "w") as f:
        json.dump(data, f)
    
    # Split the data into training, validation, and test sets
    train_data, val_data, test_data = split_data(data, {"train": 0.85, "test": 0.1})

    # Instantiate the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer.pad_token = tokenizer.eos_token
    padding_value = tokenizer.encode(tokenizer.pad_token)[-1]

    # Create the datasets
    train_dataset = CVMDataset(train_data, tokenizer)
    val_dataset = CVMDataset(val_data, tokenizer)
    test_dataset = CVMDataset(test_data, tokenizer)

    # Save the datasets and tokenizer
    torch.save(train_dataset, "train_dataset.pth")
    torch.save(val_dataset, "val_dataset.pth")
    torch.save(test_dataset, "test_dataset.pth")

    model_save_path = "./fine_tuned_llama_model"
    tokenizer.save_pretrained(model_save_path)