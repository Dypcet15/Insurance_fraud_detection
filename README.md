# Insurance Fraud Detection

## Overview
This project aims to develop a comprehensive machine learning model for detecting fraudulent insurance claims using various classification algorithms. The goal is to minimize financial losses by identifying potentially fraudulent activities early in the claims process.

## Features
- Machine learning model implementation for fraud detection
- Extensive data preprocessing and feature engineering
- Visualization of data distributions and relationships
- User-friendly command-line interface for model evaluation
- Open-source and extensible codebase for further development

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Dypcet15/Insurance_fraud_detection.git
```
2. Navigate to the project directory:
```bash
cd Insurance_fraud_detection
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
To run the model, use the following command in the terminal:
```bash
python main.py --input <input_file.csv> --output <output_file.csv>
```
Replace `<input_file.csv>` with the path to your input dataset and `<output_file.csv>` with the desired output location for results.

## Project Structure
```
Insurance_fraud_detection/
├── data/                 # Contains datasets used for training and testing
├── models/               # Directory for machine learning models
├── utils/                # Utility scripts for data processing and model evaluation
├── requirements.txt      # List of Python package dependencies
└── main.py              # Entry point for executing the model
```