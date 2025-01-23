import torch
import re
import os
import json
import ijson
import transformers
import pandas as pd
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, LlamaForCausalLM, get_scheduler
from data_pipeline import load_clean_data, CVMDataset, collate_func, create_dataloaders


def load_data_loaders(file_path: str) -> torch.utils.data.DataLoader:
    """
    Load a PyTorch DataLoader from a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        torch.utils.data.DataLoader: The DataLoader object.
    """
    data_loader = torch.load(file_path)
    return data_loader

def batch_loss_calc(
        input_batch: torch.Tensor,
        target_batch:torch.Tensor,
        model:torch.nn.Module,
        device: torch.device
    ) -> torch.Tensor:
    """
    Calculate the loss for a batch of input and target sequences.

    Args:
        input_batch (torch.Tensor): The input sequences.
        target_batch (torch.Tensor): The target sequences.
        model (torch.nn.Module): The model to evaluate.
        device (torch.device): The device to run the model on.

    Returns:
        torch.Tensor: The loss value.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    outputs = model(input_batch)
    logits = outputs.logits
    logits = logits.view(-1, logits.size(-1))
    target_batch = target_batch.view(-1)
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def all_batch_loss_calc(
        data_loader: torch.utils.data.DataLoader, 
        model: torch.nn.Module, 
        device: torch.device, 
        num_workers: int = None 
    ) -> float:
    """
    Calculate the average loss for all batches in a data loader.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader to evaluate.
        model (torch.nn.Module): The model to evaluate.
        device (torch.device): The device to run the model on.
        num_workers (int): The number of workers to use for the data loader.

    Returns:
        float: The average loss value.
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_workers is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, batch in enumerate(data_loader):
        if i < num_batches:
            input_batch = batch['input_ids']
            target_batch = batch['labels']
            loss = batch_loss_calc(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def generate_response(
        model: torch.nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        prompt: str,
        device: torch.device,
        max_length: int = 1000
    ) -> str:
    """
    Generate a response from the model given a prompt.

    Args:
        model (torch.nn.Module): The model to generate the response.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        prompt (str): The prompt to generate the response.
        device (torch.device): The device to run the model on.
        max_length (int): The maximum length of the generated response.

    Returns:
        str: The generated response.
    """
    instructions = "Below is a post from the Change My Views subreddit. Write a response that you think will change the author's view."
    prompt = instructions + "\n\n" + prompt
    
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)
    
    # Ensure pad_token_id is distinct and attention_mask is included
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],  # Pass attention mask explicitly
            max_length=max_length,
            num_beams=1, 
            pad_token_id=pad_token_id  # Explicitly set pad_token_id
        )
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Model Response")
    return response

def loss_eval(
        model: torch.nn.Module, 
        train_loader: torch.utils.data.DataLoader, 
        val_loader: torch.utils.data.DataLoader, 
        device: torch.device, 
    ) -> tuple[int]:
    """
    Evaluate the loss of the model on the training and validation data.

    Args:
        model (torch.nn.Module): The model to evaluate.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        device (torch.device): The device to run the model on.

    Returns:
        tuple: A tuple containing the training and validation loss values.
    """
    model.eval()
    with torch.no_grad():
        train_loss = all_batch_loss_calc(train_loader, model, device)
        val_loss = all_batch_loss_calc(val_loader, model, device)
    model.train()
    return train_loss, val_loss

def train_model(
        model: torch.nn.Module, 
        train_loader: torch.utils.data.DataLoader, 
        val_loader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler._LRScheduler, 
        device: torch.device, 
        num_epochs: int, 
        eval_freq: int,
    ) -> tuple[list]:
    """
    Train the model on the training data and evaluate it on the validation data.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        device (torch.device): The device to run the model on.
        num_epochs (int): The number of epochs to train for.
        eval_freq (int): The evaluation frequency.
        eval_iter (int): The number of batches to evaluate.

    Returns:
        tuple: A tuple containing the training losses, validation losses, and tokens seen.
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            print("test")
            input_batch, target_batch = batch['input_ids'], batch['labels']
            optimizer.zero_grad()
            loss = batch_loss_calc(input_batch, target_batch, model, device)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            
            tokens_seen += input_batch.numel()
            global_step += 1

            print(f"[Epoch {epoch} Batch {batch_idx}] Loss: {loss.item():.4f}")

            if global_step % eval_freq == 0:
                train_loss, val_loss = loss_eval(model, train_loader, val_loader, device)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch: {epoch}, Global step: {global_step:06d}, Tokens seen: {tokens_seen:06d}")
                print(f"Training loss: {train_loss:.3f}, Validation loss: {val_loss:.3f}")

    return train_losses, val_losses, track_tokens_seen

if __name__ == "__main__":
    # Load data loaders and cleaned data
    train_loader = load_data_loaders("train_loader.pth")
    val_loader = load_data_loaders("val_loader.pth")
    test_loader = load_data_loaders("test_loader.pth")
    fine_tune_data = load_clean_data("fine_tune_data.json")

    # Load the model - CHANGE TO 3B BEFORE COMMITTING
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")

    # Set device and move model to device
    device = torch.device("mps")
    model.to(device)

    # Evaluate the model without training - disable gradient tracking for efficiency
    with torch.no_grad():
        train_loss = all_batch_loss_calc(train_loader, model, device)
        val_loss = all_batch_loss_calc(val_loader, model, device)
    # Print the loss values and generate a response
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)
    prompt = fine_tune_data[0]['input']
    generate_response(model, tokenizer, prompt, device)

    # Fine-tune the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=10, num_training_steps=50)
    num_epochs = 1

    train_losses, val_losses, tokens_seen = train_model(
        model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, eval_freq=5
    )

    # Print results after training
    print("Final training loss:", train_losses[-1])
    print("Final validation loss:", val_losses[-1])
    prompt = fine_tune_data[0]['input']
    generate_response(model, tokenizer, prompt, device)

    # Save the model
    model_save_path = "./fine_tuned_llamam_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)