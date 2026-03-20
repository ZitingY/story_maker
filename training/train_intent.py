"""Train intent classifier: fine-tune DistilBERT on text adventure actions."""
import sys
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings


class IntentDataset(Dataset):
    """Dataset for intent classification training."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    """Compute accuracy for evaluation."""
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def load_data_from_jsonl(filepath: str) -> tuple:
    """Load training data from a JSONL file.

    Each line: {"text": "...", "intent": "..."}
    """
    texts = []
    labels = []
    label2id = {label: i for i, label in enumerate(settings.INTENT_LABELS)}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            text = sample.get("text", "")
            intent = sample.get("intent", "other")
            if text and intent in label2id:
                texts.append(text)
                labels.append(label2id[intent])

    print(f"Loaded {len(texts)} samples from {filepath}")
    # Print distribution
    from collections import Counter
    dist = Counter(labels)
    for label_idx, count in sorted(dist.items()):
        label_name = settings.INTENT_LABELS[label_idx]
        print(f"  {label_name}: {count}")
    return texts, labels


def create_synthetic_training_data():
    """
    Create synthetic training data for intent classification.
    
    In production, this should be replaced with annotated data from LIGHT dataset.
    """
    data = {
        "action": [
            "attack the goblin with my sword",
            "fight the dragon",
            "defend against the incoming arrows",
            "strike the bandit leader",
            "shoot an arrow at the wolf",
            "engage the enemy soldiers",
            "push the boulder aside",
            "kick down the door",
        ],
        "dialogue": [
            "talk to the village elder",
            "speak with the merchant",
            "ask the guard about the prisoner",
            "greet the stranger",
            "tell the innkeeper about our quest",
            "chat with the blacksmith",
            "have a conversation with the wizard",
            "say hello to the fairy",
        ],
        "explore": [
            "explore the dark cave",
            "search the room for hidden items",
            "look around the village",
            "investigate the strange noise",
            "wander through the forest",
            "survey the battlefield",
            "scout ahead for danger",
            "explore the abandoned temple",
        ],
        "use_item": [
            "use the healing potion",
            "drink the magic elixir",
            "apply the antidote",
            "activate the ancient scroll",
            "equip the enchanted armor",
            "throw the smoke bomb",
            "eat the bread to restore health",
            "use the key on the locked door",
        ],
        "ask_info": [
            "what is this place",
            "where can I find the blacksmith",
            "who is the king of this land",
            "how do I cross the river",
            "why is the village abandoned",
            "describe the creature",
            "examine the ancient inscription",
            "read the letter",
        ],
        "rest": [
            "rest by the campfire",
            "sleep at the inn",
            "wait until morning",
            "set up camp for the night",
            "meditate to focus my mind",
            "take a short nap",
            "heal my wounds and rest",
            "sit down and catch my breath",
        ],
        "trade": [
            "trade with the merchant",
            "buy a sword from the blacksmith",
            "sell my old armor",
            "barter for supplies",
            "negotiate a better price",
            "offer gold for the map",
            "shop for potions",
            "exchange gems for food",
        ],
        "other": [
            "sing a song to pass the time",
            "write a letter home",
            "pray at the shrine",
            "craft a torch from branches",
            "build a shelter",
            "start a campfire",
            "dance around the bonfire",
            "do nothing",
        ],
    }
    
    texts = []
    labels = []
    label2id = {label: i for i, label in enumerate(settings.INTENT_LABELS)}
    
    for intent, examples in data.items():
        for text in examples:
            texts.append(text)
            labels.append(label2id[intent])
    
    return texts, labels


def train(args):
    """Main training function."""
    model_name = args.model_name or settings.INTENT_MODEL_NAME
    print(f"Training intent classifier with {model_name}")
    print(f"CPU-friendly defaults: batch_size={args.batch_size}, max_length={args.max_length}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare data — load from JSONL or use synthetic
    if args.data_path and Path(args.data_path).exists():
        texts, labels = load_data_from_jsonl(args.data_path)
    else:
        texts, labels = create_synthetic_training_data()
        print(f"Using synthetic data: {len(texts)} samples")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    train_dataset = IntentDataset(train_texts, train_labels, tokenizer, max_length=args.max_length)
    val_dataset = IntentDataset(val_texts, val_labels, tokenizer, max_length=args.max_length)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(settings.INTENT_LABELS),
        id2label={i: l for i, l in enumerate(settings.INTENT_LABELS)},
        label2id={l: i for i, l in enumerate(settings.INTENT_LABELS)},
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    results = trainer.evaluate()
    print(f"\nValidation Results: {results}")
    
    # Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train intent classifier")
    parser.add_argument("--output_dir", default="models/intent_classifier", help="Output directory")
    parser.add_argument("--model_name", default=settings.INTENT_MODEL_NAME, help="Base encoder model")
    parser.add_argument("--data_path", default="", help="Path to JSONL training data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=settings.INTENT_CPU_BATCH_SIZE, help="Batch size (CPU: 8 recommended)")
    parser.add_argument("--max_length", type=int, default=settings.INTENT_MAX_LENGTH, help="Tokenizer max length")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    
    args = parser.parse_args()
    train(args)
