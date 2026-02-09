"""
Example: Using TruthGPT API via REST API
=========================================

This example demonstrates how to use the TruthGPT API via HTTP requests.
Make sure the API server is running before executing this script.

Start the server:
    python start_server.py

Then run this script:
    python example_api_usage.py
"""

import requests
import json
import numpy as np
import time

# API base URL
BASE_URL = "http://localhost:8000"

def check_server():
    """Check if the server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Server is running!")
            print(f"   Health: {response.json()}")
            return True
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure it's running:")
        print("   python start_server.py")
        return False

def create_model():
    """Create a new model."""
    print("\n📦 Creating model...")
    
    response = requests.post(f"{BASE_URL}/models/create", json={
        "layers": [
            {"type": "dense", "params": {"units": 128, "activation": "relu"}},
            {"type": "dropout", "params": {"rate": 0.2}},
            {"type": "dense", "params": {"units": 64, "activation": "relu"}},
            {"type": "dropout", "params": {"rate": 0.2}},
            {"type": "dense", "params": {"units": 3, "activation": "softmax"}}
        ],
        "name": "ExampleModel"
    })
    
    if response.status_code == 200:
        model_data = response.json()
        print(f"✅ Model created: {model_data['model_id']}")
        return model_data['model_id']
    else:
        print(f"❌ Error creating model: {response.text}")
        return None

def compile_model(model_id):
    """Compile the model."""
    print(f"\n⚙️ Compiling model {model_id}...")
    
    response = requests.post(f"{BASE_URL}/models/{model_id}/compile", json={
        "optimizer": "adam",
        "optimizer_params": {"learning_rate": 0.001},
        "loss": "sparsecategoricalcrossentropy",
        "metrics": ["accuracy"]
    })
    
    if response.status_code == 200:
        print("✅ Model compiled successfully!")
        return True
    else:
        print(f"❌ Error compiling model: {response.text}")
        return False

def train_model(model_id):
    """Train the model."""
    print(f"\n🚀 Training model {model_id}...")
    
    # Generate dummy training data
    x_train = np.random.randn(1000, 10).astype(np.float32).tolist()
    y_train = np.random.randint(0, 3, 1000).astype(np.int64).tolist()
    
    # Generate dummy validation data
    x_val = np.random.randn(200, 10).astype(np.float32).tolist()
    y_val = np.random.randint(0, 3, 200).astype(np.int64).tolist()
    
    response = requests.post(f"{BASE_URL}/models/{model_id}/train", json={
        "x_train": x_train,
        "y_train": y_train,
        "epochs": 5,
        "batch_size": 32,
        "validation_data": {
            "x": x_val,
            "y": y_val
        },
        "verbose": 1
    })
    
    if response.status_code == 200:
        training_data = response.json()
        print("✅ Model trained successfully!")
        print(f"   Training history keys: {list(training_data['history'].keys())}")
        if 'loss' in training_data['history']:
            print(f"   Final loss: {training_data['history']['loss'][-1]:.4f}")
        if 'accuracy' in training_data['history']:
            print(f"   Final accuracy: {training_data['history']['accuracy'][-1]:.4f}")
        return True
    else:
        print(f"❌ Error training model: {response.text}")
        return False

def evaluate_model(model_id):
    """Evaluate the model."""
    print(f"\n📊 Evaluating model {model_id}...")
    
    # Generate dummy test data
    x_test = np.random.randn(200, 10).astype(np.float32).tolist()
    y_test = np.random.randint(0, 3, 200).astype(np.int64).tolist()
    
    response = requests.post(f"{BASE_URL}/models/{model_id}/evaluate", json={
        "x_test": x_test,
        "y_test": y_test,
        "verbose": 0
    })
    
    if response.status_code == 200:
        eval_data = response.json()
        print("✅ Model evaluated successfully!")
        print(f"   Results: {eval_data['results']}")
        return True
    else:
        print(f"❌ Error evaluating model: {response.text}")
        return False

def predict(model_id):
    """Make predictions with the model."""
    print(f"\n🔮 Making predictions with model {model_id}...")
    
    # Generate dummy input data
    x = np.random.randn(10, 10).astype(np.float32).tolist()
    
    response = requests.post(f"{BASE_URL}/models/{model_id}/predict", json={
        "x": x,
        "verbose": 0
    })
    
    if response.status_code == 200:
        pred_data = response.json()
        predictions = np.array(pred_data['predictions'])
        print("✅ Predictions generated successfully!")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   First 3 predictions:")
        for i in range(min(3, len(predictions))):
            pred_class = np.argmax(predictions[i])
            confidence = np.max(predictions[i])
            print(f"     Sample {i}: Class {pred_class} (confidence: {confidence:.3f})")
        return True
    else:
        print(f"❌ Error making predictions: {response.text}")
        return False

def list_models():
    """List all models."""
    print("\n📋 Listing all models...")
    
    response = requests.get(f"{BASE_URL}/models")
    
    if response.status_code == 200:
        models_data = response.json()
        print(f"✅ Found {models_data['count']} model(s)")
        for model in models_data['models']:
            print(f"   - {model['model_id']}: {model['name']} (compiled: {model['compiled']})")
        return True
    else:
        print(f"❌ Error listing models: {response.text}")
        return False

def main():
    """Main function."""
    print("=" * 60)
    print("TruthGPT API Usage Example")
    print("=" * 60)
    
    # Check if server is running
    if not check_server():
        return
    
    # Create model
    model_id = create_model()
    if not model_id:
        return
    
    # Compile model
    if not compile_model(model_id):
        return
    
    # Train model
    if not train_model(model_id):
        return
    
    # Evaluate model
    if not evaluate_model(model_id):
        return
    
    # Make predictions
    if not predict(model_id):
        return
    
    # List all models
    list_models()
    
    print("\n" + "=" * 60)
    print("✅ Example completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()











