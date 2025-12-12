# dataset_loader.py
from datasets import load_dataset

def load_cnn_dailymail(split="train", num_samples=1000):
    """
    Loads the CNN/DailyMail dataset.
    Args:
        split: "train", "validation", or "test".
        num_samples: Number of articles to load.
    Returns:
        List of (article, summary) pairs.
    """
    dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)
    dataset = dataset.select(range(num_samples))  
    articles = dataset["article"]
    summaries = dataset["highlights"]
    return list(zip(articles, summaries))

if __name__ == "__main__":
    data = load_cnn_dailymail(num_samples=5)
    for i, (article, summary) in enumerate(data):
        print(f"Article {i+1}:\n{article}\nHeadline: {summary}\n{'='*50}")
