import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class IntrusionDetectionSystem:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic network traffic data"""
        np.random.seed(42)
        
        # Normal traffic patterns
        normal_data = {
            'packet_size': np.random.normal(500, 100, int(n_samples * 0.9)),
            'duration': np.random.exponential(2, int(n_samples * 0.9)),
            'src_bytes': np.random.normal(1000, 200, int(n_samples * 0.9)),
            'dst_bytes': np.random.normal(800, 150, int(n_samples * 0.9)),
            'protocol_type': np.random.choice([0, 1, 2], int(n_samples * 0.9))
        }
        
        # Anomalous traffic patterns
        anomaly_data = {
            'packet_size': np.random.normal(2000, 500, int(n_samples * 0.1)),
            'duration': np.random.exponential(10, int(n_samples * 0.1)),
            'src_bytes': np.random.normal(5000, 1000, int(n_samples * 0.1)),
            'dst_bytes': np.random.normal(100, 50, int(n_samples * 0.1)),
            'protocol_type': np.random.choice([3, 4], int(n_samples * 0.1))
        }
        
        # Combine data
        data = {}
        labels = []
        
        for key in normal_data.keys():
            data[key] = np.concatenate([normal_data[key], anomaly_data[key]])
        
        labels = [0] * int(n_samples * 0.9) + [1] * int(n_samples * 0.1)
        
        df = pd.DataFrame(data)
        df['label'] = labels
        
        return df
    
    def train(self, data):
        """Train the IDS model"""
        # Use only normal traffic for training (unsupervised)
        normal_data = data[data['label'] == 0].drop('label', axis=1)
        
        # Scale features
        scaled_data = self.scaler.fit_transform(normal_data)
        
        # Train model
        self.model.fit(scaled_data)
        self.is_trained = True
        
        print("Model trained successfully!")
    
    def predict(self, data):
        """Detect intrusions in new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Remove label if present
        features = data.drop('label', axis=1) if 'label' in data.columns else data
        
        # Scale features
        scaled_data = self.scaler.transform(features)
        
        # Predict (-1 for anomaly, 1 for normal)
        predictions = self.model.predict(scaled_data)
        
        # Convert to binary (1 for intrusion, 0 for normal)
        return (predictions == -1).astype(int)
    
    def evaluate(self, test_data):
        """Evaluate model performance"""
        predictions = self.predict(test_data)
        true_labels = test_data['label'].values
        
        print("Classification Report:")
        print(classification_report(true_labels, predictions))
        print("\nConfusion Matrix:")
        print(confusion_matrix(true_labels, predictions))
    
    def save_model(self, filepath):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        loaded = joblib.load(filepath)
        self.model = loaded['model']
        self.scaler = loaded['scaler']
        self.is_trained = True
        print(f"Model loaded from {filepath}")

def main():
    # Initialize IDS
    ids = IntrusionDetectionSystem()
    
    # Generate sample data
    print("Generating sample network traffic data...")
    data = ids.generate_sample_data(1000)
    
    # Split data
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
    
    # Train model
    print("Training intrusion detection model...")
    ids.train(train_data)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    ids.evaluate(test_data)
    
    # Save model
    ids.save_model('ids_model.pkl')
    
    # Example of real-time detection
    print("\nExample real-time detection:")
    sample_traffic = pd.DataFrame({
        'packet_size': [600, 3000],
        'duration': [1.5, 15.0],
        'src_bytes': [1200, 8000],
        'dst_bytes': [900, 50],
        'protocol_type': [1, 4]
    })
    
    predictions = ids.predict(sample_traffic)
    for i, pred in enumerate(predictions):
        status = "INTRUSION DETECTED" if pred == 1 else "Normal Traffic"
        print(f"Traffic {i+1}: {status}")

if __name__ == "__main__":
    main()