from datasets import load_dataset
import json
import random
from tqdm import tqdm

def convert_to_conversations(ds, user_key, assistant_key):
    """
    Convert a dataset to user/assistant conversations with a progress bar.
    
    Args:
        ds: Dataset (list of dicts)
        user_key: column name for user input
        assistant_key: column name for assistant output
    
    Returns:
        List of conversations, each conversation is a list of dicts with 'role' and 'content'
    """
    conversations = []
    for row in tqdm(ds, desc="Converting dataset"):
        user_content = row.get(user_key)
        assistant_content = row.get(assistant_key)
        if user_content and assistant_content:
            conversations.append([
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ])
    return conversations

def convert_allenai_to_conversations(ds):
    """
    Convert AllenAI-style dataset to user/assistant conversations.
    
    Each entry in ds should have a 'messages' list with dicts containing 'role' and 'content'.
    
    Returns a list of conversations suitable for JSONL saving.
    """
    conversations = []
    for row in tqdm(ds, desc="Converting AllenAI dataset"):
        messages = row.get("messages", [])
        conv = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role and content:
                conv.append({"role": role, "content": content})
        if conv:
            conversations.append(conv)
    return conversations

def save_conversations_jsonl(conversations, filename):
    """Save conversations to a line-delimited JSON file with progress."""
    with open(filename, "w", encoding="utf-8") as f:
        for conv in tqdm(conversations, desc=f"Saving to {filename}"):
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

def sample_from_jsonl(filename, n_samples):
    """
    Randomly sample rows from a .jsonl file without loading the entire file.
    
    Args:
        filename: path to the .jsonl file
        n_samples: number of rows to sample
        
    Returns:
        List of conversations
    """
    print(f"Sampling Data from {filename}")
    print("finding length")
    with open(filename, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    print('choosing sample rows')
    chosen_lines = set(random.sample(range(total_lines), min(n_samples, total_lines)))

    print('sampling')
    samples = []
    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in chosen_lines:
                samples.append(json.loads(line))
    return samples

if __name__ == '__main__':
    print("Running Data File")
    #18 April, let's increase the split size of the NVidia and TULU data to 50k, math we cant as it is 8K only
    #loading math dataset
    '''
    print("Loading GSM8K Data")
    ds_gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    convs_gsm8k = convert_to_conversations(ds_gsm8k, user_key="question", assistant_key="answer")
    save_conversations_jsonl(convs_gsm8k, "evaluation/data2/openai.jsonl")

    #loading tulu
    print("Downloading TULU Data")
    ds_tulu = load_dataset("allenai/tulu-3-sft-mixture", split="train[:50000]")
    convs_tulu = convert_allenai_to_conversations(ds_tulu)
    save_conversations_jsonl(convs_tulu, "evaluation/data2/allenai50k.jsonl")

    #loading nvidia
    print("Downloading NVIDIA Data..")
    ds_nvidia = load_dataset("nvidia/OpenCodeInstruct", split="train[:50000]")
    convs_nvidia = convert_to_conversations(ds_nvidia, user_key="input", assistant_key="output")
    save_conversations_jsonl(convs_nvidia, "evaluation/data2/nvidia50k.jsonl")
    
    '''
    # Convert and save metaMath 
    print("Downloading MetaMath Data")
    ds_meta_math = load_dataset("meta-math/MetaMathQA", split = 'train[:50000]')
    metaMath_convs = convert_to_conversations(ds_meta_math, user_key="query", assistant_key="response")
    save_conversations_jsonl(metaMath_convs, "evaluation/data2/metamath50k.jsonl")
    
    # Convert and save metaMath 
    print("Downloading Wizard Data")
    ds_wizard = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split = 'train[:50000]')
    wizard_convs = convert_to_conversations(ds_meta_math, user_key="query", assistant_key="response")
    save_conversations_jsonl(wizard_convs, "evaluation/data2/wizard50k.jsonl")