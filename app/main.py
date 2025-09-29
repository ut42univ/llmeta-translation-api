from __future__ import annotations

import base64
import io
from typing import Optional

import scipy.io.wavfile as wavfile
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from services.seamless import SeamlessTranslator, TranslationResult


class TextTranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to translate.")
    src_lang: str = Field(
        "eng", description="Source language in ISO 639-3 format (e.g. 'eng')."
    )
    tgt_lang: str = Field(
        "jpn", description="Target language in ISO 639-3 format (e.g. 'jpn')."
    )
    return_audio: bool = Field(
        True, description="Whether to synthesize translated speech audio."
    )
    max_new_tokens: Optional[int] = Field(
        None,
        ge=1,
        le=512,
        description="Optional limit for generated tokens to constrain latency.",
    )


class TranslationResponse(BaseModel):
    translated_text: Optional[str]
    audio_base64: Optional[str]
    audio_sample_rate: Optional[int]
    target_lang: str


app = FastAPI(
    title="Seamless Simultaneous Translation API",
    description="FastAPI backend exposing the SeamlessM4Tv2 model for text and speech translation.",
    version="0.1.0",
)


def get_translator() -> SeamlessTranslator:
    translator = getattr(app.state, "translator", None)
    if translator is None:
        raise HTTPException(
            status_code=503, detail="Translation model is still loading. Please retry."
        )
    return translator


def build_response(result: TranslationResult, target_lang: str) -> TranslationResponse:
    audio_base64 = None
    audio_sample_rate = None

    if result.audio is not None and result.sample_rate is not None:
        audio_buffer = io.BytesIO()
        wavfile.write(audio_buffer, result.sample_rate, result.audio)
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")
        audio_sample_rate = result.sample_rate

    return TranslationResponse(
        translated_text=result.text,
        audio_base64=audio_base64,
        audio_sample_rate=audio_sample_rate,
        target_lang=target_lang,
    )


@app.on_event("startup")
async def load_model() -> None:
    app.state.translator = SeamlessTranslator()


@app.get("/health")
async def health() -> dict[str, str]:
    translator = getattr(app.state, "translator", None)
    status = "ready" if translator is not None else "loading"
    device = translator.device if translator is not None else "initializing"
    return {"status": status, "device": device}


@app.post("/translate/text", response_model=TranslationResponse)
async def translate_text(payload: TextTranslationRequest) -> TranslationResponse:
    translator = get_translator()

    try:
        result = translator.translate_text(
            text=payload.text,
            src_lang=payload.src_lang,
            tgt_lang=payload.tgt_lang,
            max_new_tokens=payload.max_new_tokens,
            return_audio=payload.return_audio,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return build_response(result, target_lang=payload.tgt_lang)


@app.post("/translate/audio", response_model=TranslationResponse)
async def translate_audio(
    file: UploadFile = File(
        ..., description="Audio file containing speech to translate."
    ),
    src_lang: str = Form("eng"),
    tgt_lang: str = Form("jpn"),
    return_audio: bool = Form(True),
    max_new_tokens: Optional[int] = Form(None),
) -> TranslationResponse:
    translator = get_translator()

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    try:
        waveform, sampling_rate = torchaudio.load(io.BytesIO(raw_bytes))
    except Exception as exc:  # pragma: no cover - torchaudio errors vary
        raise HTTPException(
            status_code=400, detail=f"Unable to read audio stream: {exc}"
        ) from exc

    audio_np = waveform.cpu().numpy()

    try:
        result = translator.translate_audio(
            audio=audio_np,
            sampling_rate=sampling_rate,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_new_tokens=max_new_tokens,
            return_audio=return_audio,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return build_response(result, target_lang=tgt_lang)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
