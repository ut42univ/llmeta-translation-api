import wave
from typing import TYPE_CHECKING, Union

import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    import numpy as np


def save_wav(
    filepath: str, waveform: Union[torch.Tensor, "np.ndarray"], sample_rate: int
) -> None:
    """Persist a waveform to a WAV file using the standard library to avoid backend issues."""
    try:
        import numpy as np  # Lazy import to keep numpy optional until needed
    except ImportError as exc:  # pragma: no cover - guard for clearer error messages
        raise RuntimeError(
            "Saving WAV files requires numpy. Please install it with `pip install numpy`."
        ) from exc

    if isinstance(waveform, torch.Tensor):
        tensor = waveform.detach().cpu()
    else:
        tensor = torch.tensor(waveform)

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            f"waveform must have shape [channels, samples], got {tensor.shape}"
        )

    if not torch.isfinite(tensor).all():
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)

    max_val = tensor.abs().max()
    if max_val > 1:
        tensor = tensor / max_val

    pcm16 = (tensor.clamp(-1.0, 1.0) * (2**15 - 1)).round().to(torch.int16)

    with wave.open(filepath, "wb") as wav_file:
        wav_file.setnchannels(pcm16.size(0))
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.transpose(0, 1).contiguous().numpy().tobytes())

    print(f"Saved {filepath}")


def main():
    print("Hello from translation-api!")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

    # from text
    text_inputs = processor(
        text="Hello, my dog is cute", src_lang="eng", return_tensors="pt"
    )
    audio_array_from_text = (
        model.generate(**text_inputs, tgt_lang="jpn")[0].cpu().numpy().squeeze()
    )

    print("Generated audio array from text:", audio_array_from_text)
    resampled_audio = torchaudio.functional.resample(
        torch.tensor(audio_array_from_text).unsqueeze(0), 16000, 24000
    )
    save_wav("output/output_from_text.wav", resampled_audio, 24000)


if __name__ == "__main__":
    main()
