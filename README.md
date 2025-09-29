# llmeta-translation-api

SeamlessM4T v2 を利用した FastAPI バックエンドです。テキストと音声の同時翻訳を提供し、翻訳結果の音声合成も行えます。Python のパッケージ管理は [uv](https://github.com/astral-sh/uv) を使用します。

## セットアップ

```bash
uv sync
```

Mac で `fairseq2` 系のライブラリを利用するために `libsndfile` が必要です。未インストールの場合は Homebrew などで追加してください。

```bash
brew install libsndfile
```

モデルの初回ロード時には Hugging Face から数 GB のモデルがダウンロードされます。十分なディスク容量とネットワーク帯域を確保してください。

## サーバー起動

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

起動後、`http://localhost:8000/docs` で OpenAPI ドキュメントを確認できます。

フロントエンドのデモページを確認する場合は `http://localhost:8000/demo` にアクセスしてください。

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

## リアルタイム音声ストリーミング

WebSocket を用いたリアルタイム通訳エンドポイントを追加しています。クライアントは 16-bit PCM (リトルエンディアン) の音声チャンクをバイナリで送信し、翻訳結果のテキストと音声を逐次受信できます。

- エンドポイント: `ws://<host>:<port>/ws/translate`
- 音声は任意のサンプルレートで送って構いません（内部で 16 kHz にリサンプルされます）。
- 最初に JSON メッセージで構成情報を送信し、その後にバイナリ音声チャンクを連続送信します。

### 初期メッセージ例

```jsonc
{
  "type": "config",
  "src_lang": "eng", // ISO 639-3 言語コード
  "tgt_lang": "jpn",
  "sample_rate": 48000, // クライアント側オーディオのサンプルレート
  "return_text": true, // 文字起こしを受信するか
  "streaming_audio": true, // SeamlessStreaming で逐次音声を受信するか
  "expressive_audio": false // SeamlessExpressive を利用するか（gated モデルが必要）
}
```

以降は `Uint8Array` もしくは `ArrayBuffer` で PCM16 の音声データを送信してください。発話が終わったタイミングで JSON メッセージ `{"type": "end"}` を送ると、残りのバッファがフラッシュされてセッションが終了します。

### サーバーからのメッセージ種別

| type               | 説明                                                                             |
| ------------------ | -------------------------------------------------------------------------------- |
| `ready`            | 準備完了。`target_sample_rate` と `expressive_available` が含まれます。          |
| `partial_text`     | 部分的な翻訳テキスト。`final=true` の場合は発話が確定したことを表します。        |
| `streaming_audio`  | SeamlessStreaming によるリアルタイム音声。`audio_base64` は PCM16 を Base64 化。 |
| `expressive_audio` | SeamlessExpressive による音声（利用可能な場合のみ）。                            |
| `error`            | エラー内容。                                                                     |
| `done`             | セッション終了。                                                                 |

## SeamlessExpressive を利用する場合

SeamlessExpressive のモデルと vocoder は gated asset のため、事前に Meta のライセンスと Hugging Face 上のアクセス申請が必要です。取得したモデルファイルを任意のディレクトリに配置し、環境変数 `SEAMLESS_EXPRESSIVE_DIR` でパスを指定してください。

```bash
export SEAMLESS_EXPRESSIVE_DIR=/path/to/seamless_expressive_assets
```

上記が設定されていない場合は Expressive 音声の要求が拒否され、テキストおよび通常のストリーミング音声のみが利用できます。

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
