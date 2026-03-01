import logging
import sys
import os
from datetime import datetime
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from ..src.models import LSTMModel, ConvModel, NonLinearModel
from ..src.load_data import load_rirs
from ..src.reverb_util import TorgoAugmentedDataset, voicefixer_dereverb
from ..src.clean_utils import train_model, plot_confusion, evaluate
from ..src.dataset import pad_collate
from ..src.load_data import get_train_test_files

output_dir = "output/reverb_exp"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class LoggerWriter:
    def __init__(self, level):
        self.level = level
    def write(self, message):
        if message.strip():
            self.level(message.strip())
    def flush(self):
        pass

def setup_experiment_logger(log_name="Full_Experiment", log_dir="logs"):
    os.makedirs(f"{log_dir}/{log_name}", exist_ok=True)
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    log_file = f"{log_dir}/{log_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sys.stdout = LoggerWriter(logger.info)
    sys.stderr = LoggerWriter(logger.error)

    return logger, log_file


logger, log_filename = setup_experiment_logger("Reverb_Speech_Experiment")


def run_3(device="cuda"):
    """
    Experiment 3.4 – Reverberant Speech Classification
    Experiment 3.5 – Dereverberation-Based Classification (VoiceFixer)

    Loops over:
        - Each RIR (3 conditions)
        - Each model architecture (linear / conv / lstm)
        - Two modes: reverb-only vs reverb + dereverb
    """
    train_files, test_files = get_train_test_files()
    os.makedirs(output_dir, exist_ok=True)

    rirs = load_rirs()   # dict: {rir_name: tensor}

    models = {
        "linear": (NonLinearModel, "stft", 257),
        "conv":   (ConvModel,      "mel",  64),
        "lstm":   (LSTMModel,      "mel",  64),
    }

    # Two modes: plain reverberation (3.4) and reverberation + dereverberation (3.5)
    modes = {
        "reverb_only":  {"apply_reverb": True,  "apply_dereverb": False, "dereverb_fn": None},
        "reverb_dereverb": {"apply_reverb": True, "apply_dereverb": True,  "dereverb_fn": voicefixer_dereverb},
    }

    results = []
    total_runs = len(rirs) * len(models) * len(modes)
    current_run = 0

    logger.info(f"Starting Reverb Experiment: {total_runs} total configurations.")
    logger.info(f"RIRs: {list(rirs.keys())}")
    logger.info(f"Models: {list(models.keys())}")
    logger.info(f"Modes: {list(modes.keys())}")

    for rir_name, rir_wav in rirs.items():
        for mode_name, mode_kwargs in modes.items():
            for model_name, (Model, feat, nf) in models.items():
                current_run += 1
                conf_title = f"{rir_name}_{mode_name}_{model_name}"

                logger.info(f"--- [Run {current_run}/{total_runs}] Configuration: {conf_title} ---")

                # Pass only the single current RIR as a dict
                rir_dict = {rir_name: rir_wav}

                train_ds = TorgoAugmentedDataset(
                    train_files,
                    feat,
                    rirs=rir_dict,
                    use_cache=True,
                    cache_size=None,
                    **mode_kwargs,
                )
                test_ds = TorgoAugmentedDataset(
                    test_files,
                    feat,
                    rirs=rir_dict,
                    use_cache=True,
                    cache_size=None,
                    **mode_kwargs,
                )

                train_loader = DataLoader(
                    train_ds,
                    batch_size=32,
                    shuffle=True,
                    collate_fn=pad_collate,
                    num_workers=4,          # start with 4
                    pin_memory=True,
                    persistent_workers=True
                )

                test_loader = DataLoader(
                    test_ds,
                    batch_size=32,
                    shuffle=False,
                    collate_fn=pad_collate,
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True
                )

                logger.info(f"  Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

                model, _ = train_model(
                    Model(nf), train_loader, test_loader,
                    device=device, epochs=8
                )

                _, acc, preds, labels = evaluate(
                    model, test_loader, nn.CrossEntropyLoss(), device
                )

                plot_confusion(labels, preds, title=conf_title, save_fig=output_dir)

                results.append({
                    "rir":      rir_name,
                    "mode":     mode_name,
                    "model":    model_name,
                    "accuracy": acc,
                })

                logger.info(f"  Success! Accuracy: {acc:.4f}")

    logger.info("=" * 60)
    logger.info("Full Reverb Experiment Complete.")
    logger.info(f"Log saved to: {log_filename}")

    # Summary table
    logger.info("--- Results Summary ---")
    for r in results:
        logger.info(
            f"  RIR={r['rir']:<20} mode={r['mode']:<20} model={r['model']:<8} acc={r['accuracy']:.4f}"
        )

    return results