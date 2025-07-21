import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CircuitDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_circuit_files(directory):
    """Load all .cir files from a directory"""
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.cir'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    texts.append(content)
            except Exception as e:
                logger.warning("Error reading %s: %s", filepath, str(e))
    return texts

def create_dataset():
    """Create dataset from organized and errored circuit files"""
    # Load organized files (label 1)
    organized_dir = 'organized_cir_files'
    organized_texts = load_circuit_files(organized_dir)
    organized_labels = [1] * len(organized_texts)
    
    # Load errored files (label 0)
    errored_dir = 'errored_cir_files'
    errored_texts = []
    for filename in os.listdir(errored_dir):
        if filename.endswith('.cir'):  # Only load .cir files, not .txt log files
            filepath = os.path.join(errored_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    errored_texts.append(content)
            except Exception as e:
                logger.warning("Error reading %s: %s", filepath, str(e))
    
    errored_labels = [0] * len(errored_texts)
    
    # Combine datasets
    all_texts = organized_texts + errored_texts
    all_labels = organized_labels + errored_labels
    
    logger.info("Loaded %d organized files and %d errored files", len(organized_texts), len(errored_texts))
    logger.info("Total dataset size: %d", len(all_texts))
    
    return all_texts, all_labels

def train_model():
    """Train the reward model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)
    
    # Load data
    texts, labels = create_dataset()
    
    # Split into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    model.to(device)
    
    # Create datasets
    train_dataset = CircuitDataset(train_texts, train_labels, tokenizer)
    val_dataset = CircuitDataset(val_texts, val_labels, tokenizer)
    
    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Set up optimizer and scheduler
    epochs = 3
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        
        # Training
        model.train()
        total_train_loss = 0
        train_predictions = []
        train_true_labels = []
        
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Get predictions
            predictions = torch.argmax(outputs.logits, dim=-1)
            train_predictions.extend(predictions.cpu().numpy())
            train_true_labels.extend(labels.cpu().numpy())
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = accuracy_score(train_true_labels, train_predictions)
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_val_loss += loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                val_predictions.extend(predictions.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        
        logger.info("Train Loss: %.4f, Train Acc: %.4f", avg_train_loss, train_accuracy)
        logger.info("Val Loss: %.4f, Val Acc: %.4f", avg_val_loss, val_accuracy)
        
        # Print classification report for validation set
        logger.info("\nValidation Classification Report:")
        logger.info(classification_report(val_true_labels, val_predictions, 
                                        target_names=['Errored', 'Organized']))
    
    # Save the model
    model_save_path = 'reward_model'
    os.makedirs(model_save_path, exist_ok=True)
    
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    logger.info("Model saved to %s", model_save_path)
    
    return model, tokenizer

def evaluate_model(model, tokenizer, sample_text):
    """Evaluate the model on a sample text"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Tokenize input
    encoding = tokenizer(
        sample_text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1)
    
    return predicted_class.item(), predictions.cpu().numpy()[0]

if __name__ == "__main__":
    # Train the model
    model, tokenizer = train_model()
    
    # Test with a sample
    sample_organized = """.title KiCad schematic
V1 vin 0 pulse(0 3.3 0 0 0 100m 200m)
V2 VDD 0 3.3
M1 vout vin VDD VDD MPMOS
M2 vout vin 0 0 MNMOS
.tran 1m 400m
.model mnmos nmos level=8 version=3.3.0
.model mpmos pmos level=8 version=3.3.0
.control
run
plot v(vin)+5 v(vout)
.endc
.end"""
    
    prediction, probabilities = evaluate_model(model, tokenizer, sample_organized)
    logger.info("Sample prediction: %d (0=Errored, 1=Organized)", prediction)
    logger.info("Probabilities: Errored=%.4f, Organized=%.4f", probabilities[0], probabilities[1])
