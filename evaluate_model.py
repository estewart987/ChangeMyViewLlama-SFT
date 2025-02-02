import torch
import os
import transformers
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer, LlamaForCausalLM

from fine_tune_model import generate_response, load_clean_data


def load_model_and_tokenizer(model_save_path: str) -> tuple[LlamaForCausalLM, AutoTokenizer]:
    """
    Load the fine-tuned model and tokenizer from the specified directory.

    Args:
        model_save_path (str): The directory where the model and tokenizer are saved.

    Returns:
        model (LlamaForCausalLM): The loaded model.
        tokenizer (AutoTokenizer): The loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_save_path)
    model = LlamaForCausalLM.from_pretrained(model_save_path)
    return model, tokenizer

# Evaluate responses using GPT 4o-mini
def gpt4o_eval(
        input: str,
        generated_response: str,
        target_response: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 100,
        temperature: float = 0.5
    ) -> dict:
    """
    Evaluate a generated response using the GPT-4o-mini model.

    Args:
        input (str): The input given to the model.
        generated_response (str): The response generated by the model.
        target_response (str): The target response.
        model (str): The model to use for evaluation.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature for sampling.

    Returns:
        dict: The evaluation response from the GPT-4o-mini model.
    """

    eval_prompt = f"""
        Evaluate the following response generated by a model:
        This was the input given to the model: "Below is a post from the Change my Views subreddit. Write a response that you think will change the author's view.\n {input}"
        Generated Response: "{generated_response}"
        Target Response: "{target_response}"

        Score the model's response using a Likert scale from 1 to 7, where 1 is the worst and 7 is the best. Consider the relevance and fluency of the response when assigning a score.
        """
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key = api_key)
    chat_completion = client.chat.completions.create(
        messages = [
            {"role":"user",
             "content":eval_prompt}
        ],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )

    return chat_completion.choices[0].message.content

def evaluate_test_set(
        test_data: list[dict],
        trained_model: torch.nn.Module,
        tokenizer: transformers.AutoTokenizer,
        eval_model: str,
        device: torch.device
    ) -> float:
    """
    Evaluate the fine-tuned model on the test set using the GPT-4o-mini model.

    Args:
        test_data (list): A list of dictionaries containing the test data.
        trained_model (LlamaForCausalLM): The fine-tuned model.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        eval_model (str): The model to use for evaluation.
        device (torch.device): The device to run the model on.

    Returns:
        float: The average evaluation score for the test set.
    """
    # Store scores for each criterion
    scores = []

    for example in tqdm(test_data, desc="Evaluating test set"):
        input_text = example["input"]
        target_response = example["response"]

        # Generate response from Llama
        generated_response = generate_response(
            model=trained_model, 
            tokenizer=tokenizer, 
            prompt=input_text, 
            device=device
        )
        # Send input, generated response, and target response to GPT-4o-mini
        evaluation = gpt4o_eval(
            input=input_text,
            generated_response=generated_response,
            target_response=target_response,
            model=eval_model
        )

        # Get score
        try:
            score = int(evaluation)
        except (IndexError, ValueError):
            print("Failed to get GPT-4o-mini response:", evaluation)
            continue

        # Append scores to the respective lists
        scores.append(score)

    # Calculate and return average scores
    avg_score = sum(scores) / len(scores) if scores else 0

    print(f"Average Evaluation Score: {avg_score:.2f}")
    return avg_score


if __name__ == "__main__":
    test_data = load_clean_data("test_data.json")

    # Load in fine-tuned model
    model_save_path = "./fine_tuned_llama_model"
    model, tokenizer = load_model_and_tokenizer(model_save_path)

    device = torch.device("mps")

    avg_scores = evaluate_test_set(test_data, model, tokenizer, eval_model="gpt-4o-mini", device=device)