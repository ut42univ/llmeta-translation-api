# llmeta-translation-api

SeamlessM4T v2 を利用した FastAPI バックエンドです。テキストと音声の同時翻訳を提供し、翻訳結果の音声合成も行えます。Python のパッケージ管理は [uv](https://github.com/astral-sh/uv) を使用します。

## セットアップ

```bash
uv sync
```

モデルの初回ロード時には Hugging Face から数 GB のモデルがダウンロードされます。十分なディスク容量とネットワーク帯域を確保してください。

## サーバー起動

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

起動後、`http://localhost:8000/docs` で OpenAPI ドキュメントを確認できます。

## エンドポイント概要

| メソッド | パス               | 説明                                 |
| -------- | ------------------ | ------------------------------------ |
| GET      | `/health`          | モデルロード状況とデバイス情報を返す |
| POST     | `/translate/text`  | テキストの翻訳と音声合成             |
| POST     | `/translate/audio` | 音声ファイルの翻訳と音声合成         |

### `/translate/text`

リクエスト例:

```http
POST /translate/text HTTP/1.1
Content-Type: application/json

{
	"text": "Hello, my dog is cute",
	"src_lang": "eng",
	"tgt_lang": "jpn",
	"return_audio": true
}
```

レスポンス例:

```json
{
  "translated_text": "こんにちは、私の犬はかわいいです",
  "audio_base64": "...",
  "audio_sample_rate": 16000,
  "target_lang": "jpn"
}
```

`audio_base64` は WAV バイト列を Base64 でエンコードしたものです。デコードして保存すると翻訳音声を再生できます。

### `/translate/audio`

サポートするファイル形式は `torchaudio.load` が扱えるもの（WAV, FLAC, MP3 など）です。curl を使ったサンプル:

```bash
curl -X POST \
	-F "file=@input.wav" \
	-F "src_lang=eng" \
	-F "tgt_lang=jpn" \
	-F "return_audio=true" \
	http://localhost:8000/translate/audio
```

## 言語コードについて

SeamlessM4T v2 は ISO 639-3 ベースの言語コードを使用します。主要なコード例は以下の通りです。

- `eng`: 英語
- `jpn`: 日本語
- `cmn`: 中国語 (Mandarin)
- `kor`: 韓国語

その他のコードは [Hugging Face モデルカード](https://huggingface.co/facebook/seamless-m4t-v2-large) を参照してください。

## テスト

構文エラーのチェックには次のコマンドを利用できます。

```bash
uv run python -m compileall app
```
