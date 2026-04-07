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

    dataset_address = "meta-math/MetaMathQA"
    file_name = "data/" + "meta_math.jsonl"

    #loading metaMath
    print("Loading data")
    ds = load_dataset(dataset_address, split = 'train')
    
    # Convert and save metaMath but can use any other datasource 
    print("Converting Data")
    convs = convert_to_conversations(ds, user_key="query", assistant_key="response")
    print("Saving Data")
    save_conversations_jsonl(convs, file_name)

    #sampling example
    print("Sampling from data")
    # Later: sample 5 rows from metaMath
    sampled = sample_from_jsonl(file_name, 5)
    for conv in sampled:
        print("NEW SAMPLE")
        print("")
        print(100*'-')
        print(conv)
        print()