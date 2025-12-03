# Simple Intrusion Detection System

A minimal machine learning-based intrusion detection system using Isolation Forest algorithm.

## Features

- Anomaly detection using unsupervised learning
- Synthetic network traffic data generation
- Model training and evaluation
- Real-time intrusion detection
- Model persistence (save/load)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python intrusion_detection.py
```

## How it Works

1. **Data Generation**: Creates synthetic network traffic with normal and anomalous patterns
2. **Training**: Uses Isolation Forest to learn normal traffic patterns
3. **Detection**: Identifies anomalies as potential intrusions
4. **Evaluation**: Provides performance metrics

## Key Components

- `IntrusionDetectionSystem`: Main class handling training and detection
- Isolation Forest: Unsupervised anomaly detection algorithm
- Feature scaling for improved performance
- Model persistence for deployment