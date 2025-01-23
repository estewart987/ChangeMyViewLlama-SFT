# Fine-Tuning Llama 3.2 on Change My Views Dataset

## Overview
This project fine-tunes the Llama 3.2 model using the Change My Views (CMV) subreddit dataset. The goal is to train the model to generate responses that align with the persuasive and empathetic communication strategies used by posters who successfully changed someone’s opinion. The project serves as a foundation for exploring the potential of large language models (LLMs) to adapt their persuasion techniques depending on the user and issue being discussed.

## Features
The repository contains four Python scripts, each serving a specific purpose:

1. **`cvm_clean.py`**  
   - Cleans and preprocesses the raw CMV dataset to prepare it for formatting and fine-tuning.  
2. **`data_pipeline.py`**  
   - Formats the cleaned data into the required structure for fine-tuning Llama 3.2.  
3. **`fine_tune_model.py`**  
   - Sets up the training pipeline for fine-tuning the Llama model on the CMV dataset.  
4. **`evaluate_model.py`**  
   - Evaluates the fine-tuned model's responses using ChatGPT 4o-mini as an evaluator.  

### Capabilities
- Fine-tunes the Llama 3.2 model to emulate effective persuasive communication.
- Evaluates the model's responses using an automated scoring system powered by ChatGPT 4o-mini.

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required dependencies using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the project using the provided `setup.py`:
   ```bash
   python setup.py install
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

4. Evaluate the model:
   ```bash
   python evaluate_model.py
   ```

### Prerequisites
- Access to the Llama 3.2 model.
- OpenAI API key set as an environment variable:
   ```bash
   export OPENAI_API_KEY=<your-api-key>
   ```

## Folder Structure
The repository follows a simple structure:
```
<repository-root>/
│
├── cvm_clean.py        # Cleans the raw CMV dataset
├── data_pipeline.py    # Formats data for fine-tuning
├── fine_tune_model.py  # Fine-tunes the Llama model
├── evaluate_model.py   # Evaluates the fine-tuned model
├── requirements.txt    # Lists all necessary dependencies
└── setup.py            # Facilitates installation of the project
```

## Configuration
Ensure that the following configurations are in place:
- OpenAI API key: Set as an environment variable as shown above.
- Paths to the raw dataset and any other relevant files should be updated in the scripts if necessary.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
The project utilizes the Change My Views dataset:
```
@inproceedings{tan+etal:16a,
      author = {Chenhao Tan and Vlad Niculae and Cristian Danescu-Niculescu-Mizil and Lillian Lee},
      title = {Winning Arguments: Interaction Dynamics and Persuasion Strategies in Good-faith Online Discussions},
      year = {2016},
      booktitle = {Proceedings of WWW}
}
```

## Notes
This project demonstrates the potential for training LLMs to adopt effective persuasive communication styles. It serves as a foundational framework for future research and development in the intersection of LLMs and human-centric communication strategies.