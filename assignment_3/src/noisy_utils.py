import torch
import torchaudio

def add_noise(clean, noise, snr_db):
    """
    clean: (T,)
    noise: (T,) or shorter
    """

    if noise.numel() < clean.numel():
        repeat = clean.numel() // noise.numel() + 1
        noise = noise.repeat(repeat)

    noise = noise[: clean.numel()]

    clean_power = clean.pow(2).mean()
    noise_power = noise.pow(2).mean()

    snr_linear = 10 ** (snr_db / 10)

    scale = torch.sqrt(clean_power / (snr_linear * noise_power))
    noisy = clean + scale * noise

    return noisy

def spectral_subtraction(noisy, sr=16000):
    spec = torch.stft(noisy, n_fft=512, hop_length=160, return_complex=True)
    mag, phase = spec.abs(), spec.angle()

    noise_est = mag[:, :10].mean(dim=1, keepdim=True)  # noise from first frames
    clean_mag = torch.clamp(mag - noise_est, min=0.0)

    clean_spec = clean_mag * torch.exp(1j * phase)
    clean = torch.istft(clean_spec, n_fft=512, hop_length=160)

    return clean

def wiener_filter(noisy):
    spec = torch.stft(noisy, n_fft=512, hop_length=160, return_complex=True)
    mag, phase = spec.abs(), spec.angle()

    noise_est = mag[:, :10].mean(dim=1, keepdim=True)
    gain = mag.pow(2) / (mag.pow(2) + noise_est.pow(2))

    clean_mag = gain * mag
    clean_spec = clean_mag * torch.exp(1j * phase)

    clean = torch.istft(clean_spec, n_fft=512, hop_length=160)
    return clean

def mmse_denoise(noisy):
    spec = torch.stft(noisy, n_fft=512, hop_length=160, return_complex=True)
    mag, phase = spec.abs(), spec.angle()

    noise_est = mag[:, :10].mean(dim=1, keepdim=True)
    snr = mag.pow(2) / (noise_est.pow(2) + 1e-8)

    gain = snr / (1 + snr)
    clean_mag = gain * mag

    clean_spec = clean_mag * torch.exp(1j * phase)
    clean = torch.istft(clean_spec, n_fft=512, hop_length=160)

    return clean

_fb_model = None

def facebook_denoise(wav):
    global _fb_model

    try:
        if _fb_model is None:
            from denoiser import pretrained
            _fb_model = pretrained.dns64().cuda().eval()

        with torch.no_grad():
            return _fb_model(wav.unsqueeze(0).cuda()).cpu().squeeze()

    except Exception as e:
        print("facebook denoise error:", e)
        return wav

vf = None

def voicefixer_denoise(wav, sr=16000):
    global vf
    try:
        import tempfile
        from pathlib import Path

        if vf is None:
            from voicefixer import VoiceFixer
            vf = VoiceFixer()

        use_cuda = torch.cuda.is_available()

        wav = wav.squeeze().float()

        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "in.wav"
            out = Path(d) / "out.wav"

            torchaudio.save(inp.as_posix(), wav.unsqueeze(0), sr)

            vf.restore(
                input=inp.as_posix(),
                output=out.as_posix(),
                cuda=use_cuda,   # ← automatic GPU usage
            )

            clean, _ = torchaudio.load(out)

        return clean.squeeze(0)

    except Exception as e:
        print("VoiceFixer error:", e)
        return wav

