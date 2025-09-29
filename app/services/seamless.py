from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model


@dataclass
class TranslationResult:
    text: Optional[str]
    audio: Optional[np.ndarray]
    sample_rate: Optional[int]


class SeamlessTranslator:
    """Utility wrapper around the SeamlessM4T v2 model for multimodal translation."""

    def __init__(
        self,
        model_id: str = "facebook/seamless-m4t-v2-large",
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = SeamlessM4Tv2Model.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        self.sampling_rate: int = getattr(self.model.config, "sampling_rate", 16000)
        self._lock = threading.Lock()

    def translate_text(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        *,
        max_new_tokens: Optional[int] = None,
        return_audio: bool = False,
    ) -> TranslationResult:
        """Translate a text string to the target language, optionally returning speech."""

        inputs = self.processor(
            text=text,
            src_lang=src_lang,
            return_tensors="pt",
        )
        return self._generate(
            inputs,
            tgt_lang=tgt_lang,
            max_new_tokens=max_new_tokens,
            return_audio=return_audio,
        )

    def translate_audio(
        self,
        audio: np.ndarray,
        sampling_rate: int,
        src_lang: str,
        tgt_lang: str,
        *,
        max_new_tokens: Optional[int] = None,
        return_audio: bool = False,
    ) -> TranslationResult:
        """Translate an audio waveform to the target language."""

        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)
        if audio.ndim != 1:
            raise ValueError("Audio input must be a 1-D array after channel mixing.")

        audio = audio.astype(np.float32, copy=False)

        inputs = self.processor(
            audios=audio,
            sampling_rate=sampling_rate,
            src_lang=src_lang,
            return_tensors="pt",
        )
        return self._generate(
            inputs,
            tgt_lang=tgt_lang,
            max_new_tokens=max_new_tokens,
            return_audio=return_audio,
        )

    def _generate(
        self,
        inputs,
        *,
        tgt_lang: str,
        max_new_tokens: Optional[int],
        return_audio: bool,
    ) -> TranslationResult:
        with self._lock:
            if hasattr(inputs, "to"):
                model_inputs = inputs.to(self.device)
            else:
                model_inputs = {k: v.to(self.device) for k, v in inputs.items()}

            generation_kwargs = {
                "tgt_lang": tgt_lang,
                "generate_speech": return_audio,
                "return_dict_in_generate": True,
            }

            if not return_audio:
                forced_decoder_ids = None
                if hasattr(self.processor, "get_decoder_prompt_ids"):
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language=tgt_lang,
                        task="translate",
                    )
                elif hasattr(self.processor, "get_decoder_prompt_tokens"):
                    forced_decoder_ids = self.processor.get_decoder_prompt_tokens(
                        tgt_lang=tgt_lang,
                        task="translate",
                    )

                if forced_decoder_ids is not None:
                    generation_kwargs["forced_decoder_ids"] = forced_decoder_ids

            if max_new_tokens is not None:
                generation_kwargs["max_new_tokens"] = max_new_tokens

            with torch.inference_mode():
                outputs = self.model.generate(**model_inputs, **generation_kwargs)

        text = self._extract_text(outputs)
        audio = self._extract_audio(outputs)

        if not return_audio:
            audio = None

        if text is not None:
            print(f"[SeamlessTranslator] Generated text ({tgt_lang}): {text}")
        else:
            print("[SeamlessTranslator] Generated text: <empty>")

        return TranslationResult(
            text=text,
            audio=audio,
            sample_rate=self.sampling_rate if audio is not None else None,
        )

    def _extract_text(self, outputs) -> Optional[str]:
        sequences = getattr(outputs, "sequences", None)
        if sequences is None:
            return None

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "batch_decode"):
            decoded = tokenizer.batch_decode(
                sequences,
                skip_special_tokens=True,
            )
        elif hasattr(self.processor, "batch_decode"):
            decoded = self.processor.batch_decode(  # type: ignore[operator]
                sequences,
                skip_special_tokens=True,
            )
        else:
            return None

        if not decoded:
            return None
        return decoded[0].strip()

    def _extract_audio(self, outputs) -> Optional[np.ndarray]:
        audio_values = getattr(outputs, "audio_values", None)
        if audio_values is None:
            if isinstance(outputs, torch.Tensor):
                audio_tensor = outputs
            else:
                return None
        else:
            if isinstance(audio_values, (list, tuple)):
                if not audio_values:
                    return None
                audio_tensor = audio_values[0]
            else:
                audio_tensor = audio_values

        audio = audio_tensor.detach().cpu().numpy()
        return np.squeeze(audio).astype(np.float32)
