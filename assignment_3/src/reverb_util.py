import torch.nn.functional as F
import torch
import torchaudio
import tempfile
import soundfile as sf
import os
from voicefixer import VoiceFixer

from .dataset_util import TorgoDataset

def add_reverb(wav, rir):
    """
    Convolve waveform with Room Impulse Response.
    """
    wav = wav.unsqueeze(0).unsqueeze(0)        # (1,1,T)
    rir = rir.unsqueeze(0).unsqueeze(0)        # (1,1,R)

    out = F.conv1d(wav, rir.flip(-1), padding=rir.shape[-1] - 1)
    out = out.squeeze()

    # normalize energy
    out = out / (out.abs().max() + 1e-8)
    return out


voicefixer = VoiceFixer()


def voicefixer_dereverb(wav, sr=16000):
    """
    NOTE: VoiceFixer is NOT true dereverberation.
    This performs speech restoration/denoising only.
    """

    try:
        # ---- ensure 1-D float32 ----
        if wav.dim() > 1:
            wav = wav.squeeze()

        wav = wav.float().cpu()

        use_cuda = torch.cuda.is_available()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as inp:
            inp_path = inp.name
            out_path = inp_path.replace(".wav", "_out.wav")

        # ---- write temp input ----
        sf.write(inp_path, wav.numpy(), sr)

        # ---- run VoiceFixer ----
        voicefixer.restore(inp_path, out_path, cuda=use_cuda)

        # ---- load output ----
        wav_out, _ = torchaudio.load(out_path)

        # ---- cleanup ----
        os.remove(inp_path)
        os.remove(out_path)

        return wav_out.squeeze(0)

    except Exception as e:
        print("VoiceFixer dereverb error:", e)
        return wav
    
class TorgoAugmentedDataset(TorgoDataset):
    def __init__(
        self,
        file_list,
        feature_type,
        noises=None,
        rirs=None,
        snr=None,
        apply_reverb=False,
        apply_dereverb=False,
        dereverb_fn=None,
    ):
        super().__init__(file_list, feature_type)

        self.noises = noises
        self.rirs = rirs
        self.snr = snr
        self.apply_reverb = apply_reverb
        self.apply_dereverb = apply_dereverb
        self.dereverb_fn = dereverb_fn

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        wav = self._load_audio(path).squeeze(0)

        # ---- Reverberation ----
        if self.apply_reverb and self.rirs:
            rir = random.choice(list(self.rirs.values()))
            wav = add_reverb(wav, rir)


        # ---- Dereverberation ----
        if self.apply_dereverb and self.dereverb_fn is not None:
            wav = self.dereverb_fn(wav)

        feats = self._extract_features(wav.unsqueeze(0))
        return feats, LABEL_MAP[label]
