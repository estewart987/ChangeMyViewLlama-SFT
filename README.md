# Emulating Effective Persuasive Communication with Instruction Tuning

## Overview
This project involves fine-tuning the Llama 3.2 language model using data from the Change My View (CMV) subreddit [1]. The primary objective is to train the model to generate responses that emulate the persuasive and empathetic communication strategies demonstrated by users who successfully influenced others’ opinions. This project aims to provide a foundation for investigating the capacity of large language models (LLMs) to adapt their communication style to effectively persuade diverse audiences, depending on the context of the discussion and the issues being addressed.

## Features
The repository contains four Python scripts, each serving a specific purpose:

1. **`cvm_clean.py`**  
   - The CMV dataset contains around 25,000 samples representing tree-structured Reddit threads. I reconstruct individual conversations to get coherent data for emulation. The cleaned dataset includes the original posts along with the corresponding comment threads. The comments are structured in chronological order and are linked to the specific comments/posts they directly respond to, ensuring a clear hierarchical representation of the discussion.
2. **`data_pipeline.py`**  
   - Prepares instruction tuning data for Llama 3.2 in an instruction-response structure.
3. **`fine_tune_model.py`**  
   - Single file implementation of instruction tuning via SFT on the CMV data.
4. **`evaluate_model.py`**  
   - Uses LLM-as-a-judge evaluation [2] to evaluate the final tuned model's responses on a 7-point Likert scale using ChatGPT 4o-mini

### Capabilities
- Fine-tunes the Llama 3.2 model to emulate effective persuasive communication.
- Evaluates the model's responses using an automated scoring system powered by ChatGPT 4o-mini.

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/estewart987/ChangeMyViewLlama-SFT.git
   cd ChangeMyViewLlama-SFT
   ```

2. Install the required dependencies using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the scripts in the following order to execute the project:

1. Clean the raw dataset:
   ```bash
   python cvm_clean.py
   ```

2. Format the data for fine-tuning:
   ```bash
   python data_pipeline.py
   ```

3. Fine-tune the Llama model:
   ```bash
   python fine_tune_model.py
   ```

4. Evaluate the model using ChatGPT 4o-mini:
   ```bash
   python evaluate_model.py
   ```

### Prerequisites
- Access to the Llama 3.2 model. You can request access for the [3B parameter](https://huggingface.co/meta-llama/Llama-3.2-3B) and/or [1B parameter](https://huggingface.co/meta-llama/Llama-3.2-1B) model through a Hugging Face account.
- [OpenAI API](https://platform.openai.com/docs/overview) key set as an environment variable:
   ```bash
   export OPENAI_API_KEY=<your-api-key>
   ```

## Data Format
Below is an example of the data format used for instruction fine-tuning. We also provide the following instructions for the model: "Below is a post from the Change My Views subreddit. Write a response that you think will change the author's view."
```json
[
    {
        "input": "CMV: The current state of political discussion makes it pointless to seriously discuss politics on the Internet. I have to say that I am very disappointed with the current state of political discourse in today's society. Both in mass media and the Internet, political discussion seems to be ruled by angry extremists who think that the other side is evil and shout down, insult and in some case censor anyone who doesn't think so..." # truncated,
        "response": "I would offer this very sub as a counterpoint - throughout this election we have had very civil, well informed discussions about both candidates (albeit more democratic leaning just due to Reddit's natural demographics).  Vitriolic discussion has been shut down by the mod team and only level headed discussion allowed to remain..." # truncated 
    },"..."
]
```
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References
[1] Tan, C., Niculae, V., Danescu-Niculescu-Mizil, C., & Lee, L. (2016). *Winning Arguments: Interaction Dynamics and Persuasion Strategies in Good-faith Online Discussions*. Proceedings of WWW.

[2] Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). *Judging LLM-as-a-judge with MT-Bench and Chatbot Arena*. arXiv. [DOI:10.48550/arXiv.2306.05685](https://doi.org/10.48550/arXiv.2306.05685)