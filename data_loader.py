# data_loader.py
import logging
from datasets import load_dataset, DatasetDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Keywords that indicate forbidden content (diagnosis, prescription)
FORBIDDEN_KEYWORDS = [
    "diagnose", "diagnosis", "diagnosing",
    "prescribe", "prescription", "dosage",
    "recommend treatment", "treatment plan",
    "mg", "milligram", "clinic rule"
]

def format_prompt(example: dict) -> dict:
    """Formats a single data example into the Alpaca instruction-following format.
    The final output is a single string with the full instruction and response.
    """
    if example.get("input") and example['input'].strip() and example['input'].lower() != '<noinput>':
        # Case with instruction, input, and output
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        # Case with instruction and output only
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
    return {"text": prompt}

def contains_forbidden_keywords(example: dict) -> bool:
    """Checks if the 'output' field of an example contains any forbidden keywords.
    Returns True if a forbidden keyword is found, False otherwise.
    """
    output_text = example.get("output", "").lower()
    return any(keyword in output_text for keyword in FORBIDDEN_KEYWORDS)

def load_and_prepare_dataset(
    dataset_name: str = "lavita/AlpaCare-MedInstruct-52k",
    split_ratios: tuple = (0.9, 0.05, 0.05),
    seed: int = 42
) -> DatasetDict:
    """Loads, filters, formats, and splits the AlpaCare-MedInstruct-52k dataset."""
    logging.info(f"Loading dataset '{dataset_name}' from the Hub...")
    # The dataset only has a 'train' split by default
    dataset = load_dataset(dataset_name, split="train")
    
    original_size = len(dataset)
    logging.info(f"Original dataset size: {original_size} samples.")

    # --- Proactive Safety Filtering ---
    logging.info("Applying proactive safety filter to remove diagnostic/prescriptive content...")
    filtered_dataset = dataset.filter(lambda example: not contains_forbidden_keywords(example))
    
    filtered_size = len(filtered_dataset)
    removed_count = original_size - filtered_size
    logging.info(f"Removed {removed_count} samples containing forbidden keywords.")
    logging.info(f"Dataset size after filtering: {filtered_size} samples.")

    # --- Formatting ---
    logging.info("Formatting dataset into Alpaca instruction format...")
    formatted_dataset = filtered_dataset.map(format_prompt)

    # --- Splitting the Dataset ---
    logging.info(f"Splitting dataset into train/validation/test with ratios {split_ratios}...")
    if sum(split_ratios) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    # First split: train and a temporary set for validation+test
    train_val_test_split = formatted_dataset.train_test_split(
        test_size=(split_ratios[1] + split_ratios[2]),
        seed=seed
    )
    
    train_split = train_val_test_split['train']
    
    # Second split: validation and test from the temporary set
    val_test_split = train_val_test_split['test'].train_test_split(
        test_size=(split_ratios[2] / (split_ratios[1] + split_ratios[2])),
        seed=seed
    )
    
    validation_split = val_test_split['train']
    test_split = val_test_split['test']

    final_splits = DatasetDict({
        'train': train_split,
        'validation': validation_split,
        'test': test_split
    })
    
    logging.info("Dataset preparation complete.")
    logging.info(f"Final split sizes: Train={len(final_splits['train'])}, Validation={len(final_splits['validation'])}, Test={len(final_splits['test'])}")
    
    return final_splits

if __name__ == '__main__':
    # Example of how to run the function
    prepared_datasets = load_and_prepare_dataset()
    print("\nSample from the training set:")
    print(prepared_datasets['train'][0]['text'])