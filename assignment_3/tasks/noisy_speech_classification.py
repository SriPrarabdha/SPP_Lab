import logging
import sys
import os
from datetime import datetime
from torch.utils.data import DataLoader
import torch.nn as nn


from ..src.noisy_utils import spectral_subtraction, wiener_filter, mmse_denoise, facebook_denoise, voicefixer_denoise
from ..src.models import LSTMModel, ConvModel, NonLinearModel
from ..src.load_data import load_noises, load_rirs
from ..src.dataset_util import NoisyTorgoDataset
from ..src.clean_utils import train_model, plot_confusion, evaluate
from ..src.dataset import pad_collate

output_dir = "output/noise_exp"

class LoggerWriter:
    def __init__(self, level):
        self.level = level
    def write(self, message):
        if message.strip():
            self.level(message.strip())
    def flush(self):
        pass

def setup_experiment_logger(log_name="Full_Experiment" , log_dir = "logs"):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')

    # Console Output
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Output
    log_file = f"{log_dir}/{log_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Redirect all 'print' statements to this logger
    sys.stdout = LoggerWriter(logger.info)
    sys.stderr = LoggerWriter(logger.error)
    
    return logger, log_file

logger, log_filename = setup_experiment_logger("Noisy_Speech_Experiment")

# --- 2. The Experiment Function ---

def run_full_experiment(train_files, test_files, device="cuda"):
    noises = load_noises()
    
    snrs = [-10, -5, 0]
    denoisers = {
        "specsub": spectral_subtraction,
        "wiener": wiener_filter,
        "mmse": mmse_denoise,
        "facebook": facebook_denoise,
        "voicefixer": voicefixer_denoise,
    }
    models = {
        "linear": (NonLinearModel, "stft", 257),
        "conv": (ConvModel, "mel", 64),
        "lstm": (LSTMModel, "mel", 64),
    }

    results = []
    total_runs = len(noises) * len(snrs) * len(denoisers) * len(models)
    current_run = 0

    logger.info(f"Starting Noisy Experiment: {total_runs} total configurations.")

    for noise_name, noise_wav in noises.items():
        for snr in snrs:
            for den_name, den_fn in denoisers.items():
                for model_name, (Model, feat, nf) in models.items():
                    current_run += 1
                    conf_title = f"{noise_name}_{snr}db_{den_name}_{model_name}"
                    
                    logger.info(f"--- [Run {current_run}/{total_runs}] Configuration: {conf_title} ---")
                    
                    try:
                        # Dataset setup
                        train_ds = NoisyTorgoDataset(train_files, feat, noise_wav, snr, den_fn)
                        test_ds = NoisyTorgoDataset(test_files, feat, noise_wav, snr, den_fn)

                        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=pad_collate)
                        test_loader = DataLoader(test_ds, batch_size=256, collate_fn=pad_collate)

                        # Training
                        model, _ = train_model(Model(nf), train_loader, test_loader, device=device, epochs=8)

                        # Evaluation
                        _, acc, preds, labels = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
                        plot_confusion(labels, preds, title=conf_title, save_fig=output_dir)
                        
                        results.append({
                            "noise": noise_name,
                            "snr": snr,
                            "denoise": den_name,
                            "model": model_name,
                            "accuracy": acc,
                        })
                        logger.info(f"Success! Accuracy: {acc:.4f}")

                    except Exception as e:
                        logger.error(f"FAILED Run {current_run}: {conf_title}. Error: {str(e)}")
                        continue # Skip to next configuration

    logger.info(f"Full Experiment Complete. Log saved to {log_filename}")
    return results