# Configuration settings for the Fraud Detection Model

# Model Parameters
MODEL_NAME = 'FraudDetector'
MODEL_TYPE = 'classification'

# File Paths
TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'
OUTPUT_MODEL_PATH = 'models/fraud_detector.pkl'

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50
DROPOUT_RATE = 0.5

# Other settings
RANDOM_SEED = 42