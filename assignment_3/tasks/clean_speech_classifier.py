import sys
import logging
import sys
import torch.nn as nn
from datetime import datetime

from ..src.models import NonLinearModel, ConvModel, LSTMModel
from ..src.dataset import create_clean_dataloaders
from ..src.utils import train_model, plot_history, plot_confusion, plot_roc_auc, evaluate
from ..src.data import get_train_test_files


def setup_logger(log_name="training_log", log_dir = "logs"):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    
    # Format: Timestamp | Level | Message
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # StreamHandler for the Console (Colab output)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # FileHandler to save logs to a file
    fh = logging.FileHandler(f"{log_dir}/{log_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

logger = setup_logger("Clean Speech Experiment")

class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.strip(): # Avoid logging empty newlines
            self.level(message.strip())

    def flush(self):
        pass

# Redirect stdout (print) and stderr (errors) to your logger
sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.error)

# --- Start Experiment ---

device = "cuda"
train_files, test_files = get_train_test_files()
logger.info(f"Loaded {len(train_files)} training files and {len(test_files)} test files.")

experiments = [
    {"name": "NonLinearModel", "model": NonLinearModel(), "feature": "stft"},
    {"name": "ConvModel", "model": ConvModel(), "feature": "mel"},
    {"name": "LSTMModel", "model": LSTMModel(), "feature": "mel"}
]

for exp in experiments:
    logger.info(f"{'='*20} Starting Experiment: {exp['name']} ({exp['feature']}) {'='*20}")
    
    try:
        # Data Loading
        train_loader, test_loader = create_clean_dataloaders(train_files, test_files, exp['feature'])
        logger.info(f"Dataloaders created for {exp['feature']}")

        # Training
        logger.info(f"Beginning training for {exp['name']} on {device}...")
        model, history = train_model(exp['model'], train_loader, test_loader, device=device)
        logger.info(f"Training finished. Final Loss: {history['train_loss'][-1]:.4f}")

        # Evaluation & Plotting
        logger.info(f"Evaluating {exp['name']}...")
        plot_history(history, exp['name'])
        _, _, preds, labels = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
        
        plot_confusion(labels, preds)
        plot_roc_auc(model, test_loader, device, f"{exp['name']} ROC")
        
        logger.info(f"Finished {exp['name']} successfully.\n")

    except Exception as e:
        logger.error(f"Error during {exp['name']}: {str(e)}")

logger.info("All experiments completed.")