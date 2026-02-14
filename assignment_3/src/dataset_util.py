import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as T


SAMPLE_RATE = 16000

# ---- STFT magnitude (257 freq bins) ----
stft_transform = T.Spectrogram(
    n_fft=512,          # -> 257 freq bins
    hop_length=160,
    power=2.0
)

# ---- Log-Mel spectrogram ----
mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=512,
    hop_length=160,
    n_mels=64
)

db_transform = T.AmplitudeToDB()


LABEL_MAP = {
    "F_Con": 0,
    "M_Con": 0,
    "F_Dys": 1,
    "M_Dys": 1,
}


class TorgoDataset(Dataset):
    def __init__(self, file_list, feature_type="stft"):
        """
        feature_type:
            "stft" -> for NonLinearModel
            "mel"  -> for ConvModel / LSTMModel
        """
        self.file_list = file_list
        self.feature_type = feature_type

    def __len__(self):
        return len(self.file_list)

    def _load_audio(self, path):
        wav, sr = torchaudio.load(path)

        # mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # resample
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

        return wav

    def _extract_features(self, wav):
        if self.feature_type == "stft":
            spec = stft_transform(wav)        # (1, 257, T)
            return spec.squeeze(0)            # (257, T)

        elif self.feature_type == "mel":
            mel = mel_transform(wav)
            mel_db = db_transform(mel)
            return mel_db.squeeze(0)          # (n_mels, T)

        else:
            raise ValueError("Unknown feature_type")

    def __getitem__(self, idx):
        path, label = self.file_list[idx]

        wav = self._load_audio(path)
        feats = self._extract_features(wav)

        return feats, LABEL_MAP[label]

