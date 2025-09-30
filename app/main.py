from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Optional

import scipy.io.wavfile as wavfile
import torchaudio
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from services.seamless import SeamlessTranslator, TranslationResult
from services.streaming import StreamingConfig, StreamingService, StreamingServiceError


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


logger = logging.getLogger(__name__)


app = FastAPI(
    title="Seamless Simultaneous Translation API",
    description="FastAPI backend exposing the SeamlessM4Tv2 model for text and speech translation.",
    version="0.1.0",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/demo", StaticFiles(directory=STATIC_DIR, html=True), name="demo")


@app.get("/", include_in_schema=False)
async def root():
    if STATIC_DIR.exists():
        return RedirectResponse(url="/demo")
    return {"message": "Seamless translation API is running. Visit /docs for usage."}


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
    loop = asyncio.get_running_loop()

    translator_future = loop.run_in_executor(None, SeamlessTranslator)

    expressive_dir_value = os.getenv("SEAMLESS_EXPRESSIVE_DIR")
    expressive_path: Optional[Path] = None
    if expressive_dir_value:
        candidate = Path(expressive_dir_value).expanduser()
        if candidate.exists():
            expressive_path = candidate
        else:
            logger.warning(
                "SEAMLESS_EXPRESSIVE_DIR=%s does not exist; expressive streaming disabled.",
                candidate,
            )

    def build_streaming() -> StreamingService:
        return StreamingService(expressive_asset_dir=expressive_path)

    streaming_future = loop.run_in_executor(None, build_streaming)

    translator_result, streaming_result = await asyncio.gather(
        translator_future, streaming_future, return_exceptions=True
    )

    if isinstance(translator_result, Exception):
        raise translator_result

    app.state.translator = translator_result

    if isinstance(streaming_result, Exception):
        logger.warning("Streaming service initialization failed: %s", streaming_result)
        app.state.streaming = None
    else:
        app.state.streaming = streaming_result


@app.get("/health")
async def health() -> dict[str, object]:
    translator = getattr(app.state, "translator", None)
    status = "ready" if translator is not None else "loading"
    device = translator.device if translator is not None else "initializing"
    streaming_service: Optional[StreamingService] = getattr(
        app.state, "streaming", None
    )
    streaming_status = "ready" if streaming_service is not None else "loading"
    streaming_device = (
        streaming_service.device if streaming_service is not None else "initializing"
    )
    expressive_available = (
        streaming_service.expressive_available
        if streaming_service is not None
        else False
    )
    return {
        "status": status,
        "device": device,
        "streaming": streaming_status,
        "streaming_device": streaming_device,
        "expressive_available": expressive_available,
        "target_sample_rate": (
            streaming_service.target_sample_rate
            if streaming_service is not None
            else None
        ),
    }


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


@app.websocket("/ws/translate")
async def websocket_translate(websocket: WebSocket) -> None:
    await websocket.accept()
    streaming_service: Optional[StreamingService] = getattr(
        app.state, "streaming", None
    )
    if streaming_service is None:
        await websocket.send_json(
            {
                "type": "error",
                "message": "Streaming service is not ready. Please retry shortly.",
            }
        )
        await websocket.close(code=1013)
        return

    session = None

    try:
        while True:
            message = await websocket.receive()
            message_type = message.get("type")
            if message_type == "websocket.disconnect":
                break

            text_payload = message.get("text")
            binary_payload = message.get("bytes")

            if text_payload is not None:
                try:
                    data = json.loads(text_payload)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"type": "error", "message": "Invalid JSON payload."}
                    )
                    continue

                msg_type = data.get("type")

                if msg_type == "config":
                    sample_rate_raw = data.get(
                        "sample_rate", streaming_service.target_sample_rate
                    )
                    try:
                        sample_rate = int(sample_rate_raw)
                    except (TypeError, ValueError):
                        sample_rate = streaming_service.target_sample_rate

                    config = StreamingConfig(
                        src_lang=str(data.get("src_lang", "eng")),
                        tgt_lang=str(data.get("tgt_lang", "jpn")),
                        sample_rate=sample_rate,
                        return_text=bool(data.get("return_text", True)),
                        return_streaming_audio=bool(data.get("streaming_audio", True)),
                        return_expressive_audio=bool(
                            data.get("expressive_audio", False)
                        ),
                    )

                    try:
                        if session is not None:
                            await session.close()
                        session = streaming_service.create_session(config)
                    except StreamingServiceError as exc:
                        await websocket.send_json(
                            {"type": "error", "message": str(exc)}
                        )
                        session = None
                        continue

                    await websocket.send_json(
                        {
                            "type": "ready",
                            "target_sample_rate": streaming_service.target_sample_rate,
                            "expressive_available": streaming_service.expressive_available,
                        }
                    )

                elif msg_type == "end":
                    if session is not None:
                        try:
                            events = await session.finalize()
                            for event in events:
                                await websocket.send_json(event)
                        finally:
                            await session.close()
                            session = None
                    await websocket.send_json({"type": "done"})
                    break

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Unknown message type: {msg_type}",
                        }
                    )

            elif binary_payload is not None:
                if session is None:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Send a config message before streaming audio.",
                        }
                    )
                    continue

                try:
                    events = await session.process_chunk(binary_payload)
                except StreamingServiceError as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    continue

                for event in events:
                    await websocket.send_json(event)

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Unhandled websocket error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except RuntimeError:
            pass
    finally:
        if session is not None:
            await session.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
