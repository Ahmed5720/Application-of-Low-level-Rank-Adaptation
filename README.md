# Application-of-Low-level-Rank-Adaptation


## Overview
In this project, I attempt to utilize Low-Rank Adaptation based on the [LORA paper](https://arxiv.org/abs/2106.09685) to effieciently fine-tune a transformer model for the task of sentiement classification and study the tradeoffs in computational efficiency, training time, and performance compared to standard finetuning. I am using the implementation of LORA from the [PEFT](https://github.com/huggingface/peft) library available at [HuggingFace](https://huggingface.co/). 

## Project Structure
```
├── data/                   # Dataset used for training and evaluation
├── models/                 # Saved models and checkpoints
├── scripts/                # Training and evaluation scripts
├── results/                # Performance metrics and analysis
├── README.md               # Project documentation
└── requirements.txt        # Required dependencies
```

## Methodology
We conducted experiments by fine-tuning a transformer-based model on a classification task using:
1. **Standard Fine-Tuning:** Updates all model parameters.
2. **LoRA Fine-Tuning:** Uses Low-Rank Adaptation to update a smaller subset of parameters efficiently.

## Experiment Setup
- **Model:** [Specify the transformer model used, e.g., BERT, GPT, T5]
- **Dataset:** [Specify dataset used, e.g., IMDb, AG News, etc.]
- **Hardware:** [GPU details if applicable]
- **Framework:** PyTorch + Hugging Face Transformers

## Results
| Approach        | Trainable Params | Training Time | Memory Usage | Accuracy | F1 Score |
|---------------|-----------------|---------------|-------------|---------|---------|
| Standard Fine-Tuning | 124.6M | 54.8 min | ~8GB | 74.3% | 0.67 |
| LoRA Fine-Tuning | 0.89M | 41.8 min | ~6GB | 68.3% | 0.69 |

### Key Observations
- **LoRA significantly reduces trainable parameters (~0.71% of full model) while maintaining reasonable accuracy.**
- **LoRA reduces memory usage and training time, making it suitable for resource-constrained environments.**
- **Standard fine-tuning achieves higher accuracy but has higher computational cost.**
- **LoRA improves F1-score, indicating better generalization for certain cases.**

## Installation
To set up the environment, run:
```sh
pip install -r requirements.txt
```

## Running Experiments
To train using standard fine-tuning:
```sh
python scripts/train.py --method standard
```
To train using LoRA:
```sh
python scripts/train.py --method lora
```
To evaluate models:
```sh
python scripts/evaluate.py --model_path models/your_model.pth
```

## Future Work
- Experiment with different LoRA rank (`r`) values.
- Investigate hybrid approaches (LoRA + partial fine-tuning).
- Test on different model architectures and datasets.

## References
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

---
© 2025 Your Name | License: MIT

