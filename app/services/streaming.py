from __future__ import annotations

import asyncio
import base64
import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchaudio.transforms as T
from seamless_communication.streaming.agents.seamless_s2st import SeamlessS2STAgent
from seamless_communication.streaming.agents.seamless_streaming_s2st import (
    SeamlessStreamingS2STAgent,
)
from seamless_communication.streaming.agents.seamless_streaming_s2t import (
    SeamlessStreamingS2TDetokAgent,
)
from simuleval.data.segments import EmptySegment, Segment, SpeechSegment

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000
PCM_SCALE = 32768.0


class StreamingServiceError(RuntimeError):
    """Raised when the streaming service configuration is invalid."""


@dataclass(slots=True)
class StreamingConfig:
    src_lang: str
    tgt_lang: str
    sample_rate: int
    return_text: bool = True
    return_streaming_audio: bool = True
    return_expressive_audio: bool = False

    def normalized(self) -> "StreamingConfig":
        if not self.src_lang or not self.tgt_lang:
            raise StreamingServiceError(
                "Both 'src_lang' and 'tgt_lang' must be provided."
            )
        src = self.src_lang.strip().lower()
        tgt = self.tgt_lang.strip().lower()
        if len(src) != 3 or len(tgt) != 3:
            raise StreamingServiceError(
                "Language codes must be ISO 639-3 (three letters)."
            )
        if self.sample_rate <= 0:
            raise StreamingServiceError("Sample rate must be a positive integer.")
        if not (
            self.return_text
            or self.return_streaming_audio
            or self.return_expressive_audio
        ):
            raise StreamingServiceError(
                "At least one of text, streaming audio, or expressive audio must be enabled."
            )
        return StreamingConfig(
            src_lang=src,
            tgt_lang=tgt,
            sample_rate=self.sample_rate,
            return_text=self.return_text,
            return_streaming_audio=self.return_streaming_audio,
            return_expressive_audio=self.return_expressive_audio,
        )


@dataclass(frozen=True)
class PipelineDescriptor:
    key: str
    agent_cls: type
    task: str
    default_kwargs: Dict[str, object]
    requires_expressive_assets: bool = False


class PipelineResource:
    def __init__(self, name: str, agent, sample_rate: int) -> None:
        self.name = name
        self.agent = agent
        self.sample_rate = sample_rate
        self.lock = asyncio.Lock()

    def spawn_session(self) -> "PipelineSession":
        return PipelineSession(self)


class PipelineSession:
    def __init__(self, resource: PipelineResource) -> None:
        self.resource = resource
        self.states = resource.agent.build_states()

    async def process_segments(
        self,
        samples: np.ndarray,
        *,
        finished: bool,
        tgt_lang: str,
        config: Dict[str, object],
    ) -> List[Segment]:
        segment = SpeechSegment(
            content=samples.tolist(),
            sample_rate=self.resource.sample_rate,
            finished=finished,
            tgt_lang=tgt_lang,
        )
        if config:
            segment.config = dict(config)

        async with self.resource.lock:
            self.resource.agent.push(segment, self.states)
            outputs: List[Segment] = []
            while True:
                result = self.resource.agent.pop(self.states)
                if isinstance(result, EmptySegment) or getattr(
                    result, "is_empty", False
                ):
                    break
                outputs.append(result)
        return outputs

    async def reset(self) -> None:
        async with self.resource.lock:
            self.states = self.resource.agent.build_states()


PIPELINES: Dict[str, PipelineDescriptor] = {
    "text": PipelineDescriptor(
        key="text",
        agent_cls=SeamlessStreamingS2TDetokAgent,
        task="s2tt",
        default_kwargs={},
    ),
    "streaming": PipelineDescriptor(
        key="streaming",
        agent_cls=SeamlessStreamingS2STAgent,
        task="s2st",
        default_kwargs={"vocoder_name": "vocoder_v2"},
    ),
    "expressive": PipelineDescriptor(
        key="expressive",
        agent_cls=SeamlessS2STAgent,
        task="s2st",
        default_kwargs={"vocoder_name": "vocoder_pretssel"},
        requires_expressive_assets=True,
    ),
}


def _pcm16le_to_float32(data: bytes) -> np.ndarray:
    if not data:
        return np.empty(0, dtype=np.float32)
    aligned = data[: len(data) - (len(data) % 2)]
    if not aligned:
        return np.empty(0, dtype=np.float32)
    return np.frombuffer(aligned, dtype="<i2").astype(np.float32) / PCM_SCALE


def _float32_to_pcm16_bytes(samples: Iterable[float]) -> bytes:
    array = np.asarray(list(samples), dtype=np.float32)
    if array.size == 0:
        return b""
    clipped = np.clip(array, -1.0, 1.0)
    ints = (clipped * (PCM_SCALE - 1)).astype(np.int16)
    return ints.tobytes()


class StreamingSession:
    def __init__(
        self,
        config: StreamingConfig,
        pipeline_sessions: Dict[str, PipelineSession],
        *,
        target_sample_rate: int,
    ) -> None:
        self.config = config
        self.pipeline_sessions = pipeline_sessions
        self.target_sample_rate = target_sample_rate
        self._resampler: Optional[T.Resample] = None
        if config.sample_rate != target_sample_rate:
            self._resampler = T.Resample(
                orig_freq=config.sample_rate,
                new_freq=target_sample_rate,
            )
        self._closed = False

    async def process_chunk(self, audio_bytes: bytes) -> List[Dict[str, object]]:
        if self._closed:
            raise StreamingServiceError("Streaming session already closed.")

        samples = _pcm16le_to_float32(audio_bytes)
        if samples.size == 0:
            return []

        if self._resampler is not None:
            tensor = torch.from_numpy(samples).unsqueeze(0)
            with torch.inference_mode():
                resampled = self._resampler(tensor)
            samples = resampled.squeeze(0).cpu().numpy()

        return await self._dispatch(samples, finished=False)

    async def finalize(self) -> List[Dict[str, object]]:
        if self._closed:
            return []
        events = await self._dispatch(np.empty(0, dtype=np.float32), finished=True)
        self._closed = True
        return events

    async def close(self) -> None:
        for session in self.pipeline_sessions.values():
            await session.reset()
        self._closed = True

    async def _dispatch(
        self, samples: np.ndarray, *, finished: bool
    ) -> List[Dict[str, object]]:
        events: List[Dict[str, object]] = []
        dynamic_config = {
            "sourceLanguage": self.config.src_lang,
            "targetLanguage": self.config.tgt_lang,
        }
        if self.config.return_expressive_audio:
            dynamic_config["expressive"] = True

        for name, session in self.pipeline_sessions.items():
            segments = await session.process_segments(
                samples,
                finished=finished,
                tgt_lang=self.config.tgt_lang,
                config=dynamic_config,
            )
            events.extend(self._segments_to_events(name, segments))
        return events

    def _segments_to_events(
        self, pipeline_name: str, segments: Iterable[Segment]
    ) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for segment in segments:
            data_type = getattr(segment, "data_type", None)
            if data_type == "text":
                text = str(getattr(segment, "content", "")).strip()
                if not text:
                    continue
                results.append(
                    {
                        "type": "partial_text",
                        "stream": pipeline_name,
                        "text": text,
                        "final": bool(segment.finished),
                        "target_lang": segment.tgt_lang or self.config.tgt_lang,
                    }
                )
            elif data_type == "speech":
                payload = getattr(segment, "content", [])
                if not payload:
                    continue
                audio_bytes = _float32_to_pcm16_bytes(payload)
                if not audio_bytes:
                    continue
                results.append(
                    {
                        "type": (
                            "expressive_audio"
                            if pipeline_name == "expressive"
                            else "streaming_audio"
                        ),
                        "stream": pipeline_name,
                        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
                        "sample_rate": getattr(
                            segment, "sample_rate", self.target_sample_rate
                        ),
                        "final": bool(segment.finished),
                        "target_lang": segment.tgt_lang or self.config.tgt_lang,
                    }
                )
        return results


class StreamingService:
    def __init__(
        self,
        *,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        expressive_asset_dir: Optional[Path] = None,
    ) -> None:
        self.target_sample_rate = target_sample_rate
        self.device = self._detect_device()
        self.dtype = torch.float16 if self.device != "cpu" else torch.float32
        self._cache: Dict[Tuple[str, str], PipelineResource] = {}
        self._expressive_dir = expressive_asset_dir
        self._expressive_available = expressive_asset_dir is not None
        logger.info(
            "Initialized streaming service (device=%s, dtype=%s, expressive=%s)",
            self.device,
            self.dtype,
            self._expressive_available,
        )

    @property
    def expressive_available(self) -> bool:
        return self._expressive_available

    def create_session(self, config: StreamingConfig) -> StreamingSession:
        cfg = config.normalized()
        pipeline_sessions: Dict[str, PipelineSession] = {}

        if cfg.return_text:
            pipeline_sessions["text"] = self._get_resource(
                "text", cfg.tgt_lang
            ).spawn_session()

        if cfg.return_streaming_audio:
            pipeline_sessions["streaming"] = self._get_resource(
                "streaming", cfg.tgt_lang
            ).spawn_session()

        if cfg.return_expressive_audio:
            if not self._expressive_available:
                raise StreamingServiceError(
                    "SeamlessExpressive assets are not available. Set 'expressive_asset_dir'."
                )
            pipeline_sessions["expressive"] = self._get_resource(
                "expressive", cfg.tgt_lang
            ).spawn_session()

        return StreamingSession(
            cfg,
            pipeline_sessions,
            target_sample_rate=self.target_sample_rate,
        )

    def _get_resource(self, pipeline_key: str, target_lang: str) -> PipelineResource:
        cache_key = (pipeline_key, target_lang)
        if cache_key not in self._cache:
            descriptor = PIPELINES[pipeline_key]
            agent = self._build_agent(descriptor, target_lang)
            self._cache[cache_key] = PipelineResource(
                pipeline_key, agent, self.target_sample_rate
            )
        return self._cache[cache_key]

    def _build_agent(self, descriptor: PipelineDescriptor, target_lang: str):
        if descriptor.requires_expressive_assets and not self._expressive_dir:
            raise StreamingServiceError(
                "Expressive pipeline requires gated assets. Provide 'expressive_asset_dir'."
            )

        parser = ArgumentParser(add_help=False)
        descriptor.agent_cls.add_args(parser)
        args = parser.parse_args([])
        args.task = descriptor.task
        args.device = self.device
        args.fp16 = bool(self.dtype == torch.float16)
        args.dtype = "fp16" if args.fp16 else "fp32"
        args.sample_rate = self.target_sample_rate
        args.tgt_lang = target_lang
        args.output_index = getattr(args, "output_index", None)
        args.output_type = getattr(args, "output_type", None)
        args.debug = getattr(args, "debug", False)
        for key, value in descriptor.default_kwargs.items():
            setattr(args, key, value)
        if descriptor.requires_expressive_assets:
            setattr(args, "gated_model_dir", self._expressive_dir)

        logger.info(
            "Loading %s pipeline for target '%s' on %s",
            descriptor.key,
            target_lang,
            self.device,
        )
        agent = descriptor.agent_cls.from_args(args)
        return agent

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


__all__ = [
    "StreamingConfig",
    "StreamingService",
    "StreamingServiceError",
    "StreamingSession",
]
