import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

class RewardModel:
    def __init__(self, model_path='reward_model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, circuit_text):
        """
        Predict if circuit text is organized (1) or errored (0)
        Returns: (predicted_class, confidence_scores)
        """
        # Tokenize input
        encoding = self.tokenizer(
            circuit_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
        
        return predicted_class.item(), probabilities.cpu().numpy()[0]
    
    def get_reward_score(self, circuit_text):
        """
        Get a reward score between 0 and 1 (higher = better quality)
        """
        prediction, probabilities = self.predict(circuit_text)
        # Return probability of being organized (class 1)
        return probabilities[1]

def test_reward_model():
    """Test the trained reward model"""
    reward_model = RewardModel()
    
    # Test with sample texts
    organized_sample = """.title KiCad schematic
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
    
    errored_sample = """.title KiCad schematic
V1 vin 0 pulse(0 3.3 0 0 0 100m 200m)
V2 VDD 0 3.3
M1 vout vin VDD VDD MPMOS
M2 vout vin 0 0 MNMOS
.tran 1m 400m
.model m#mos nmos level=8 version=3.3.0
.model mpmos pmos level=8 version=3.3.0
.control
run
plot v(vin)+5 v(vout)
.endc
.end"""
    
    print("Testing organized sample:")
    prediction, probs = reward_model.predict(organized_sample)
    print(f"Prediction: {prediction} (0=Errored, 1=Organized)")
    print(f"Probabilities: Errored={probs[0]:.4f}, Organized={probs[1]:.4f}")
    print(f"Reward Score: {reward_model.get_reward_score(organized_sample):.4f}")
    
    print("\nTesting errored sample:")
    prediction, probs = reward_model.predict(errored_sample)
    print(f"Prediction: {prediction} (0=Errored, 1=Organized)")
    print(f"Probabilities: Errored={probs[0]:.4f}, Organized={probs[1]:.4f}")
    print(f"Reward Score: {reward_model.get_reward_score(errored_sample):.4f}")

if __name__ == "__main__":
    test_reward_model()
