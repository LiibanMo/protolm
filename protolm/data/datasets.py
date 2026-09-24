from datasets import load_dataset

gsm8k_train = load_dataset("openai/gsm8k", "main", split="train")
gms8k_test = load_dataset("openai/gsm8k", "main", split="test")
