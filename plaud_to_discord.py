#!/usr/bin/env python3
"""
Plaud → Gemini 文字起こし → Gemini アクション抽出 → Discord 送信 → Google Drive 保存
GitHub Actions での定期自動実行対応
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime

# dotenv はローカル実行時のみ（GitHub Actions では不要）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# === 設定（GitHub Secrets / 環境変数から取得） ===
PLAUD_API_DOMAIN = os.getenv("PLAUD_API_DOMAIN", "https://api-apne1.plaud.ai")
PLAUD_BEARER_TOKEN = os.getenv("PLAUD_BEARER_TOKEN", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
GOOGLE_OAUTH_CLIENT_ID = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")
GOOGLE_OAUTH_CLIENT_SECRET = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "")
GOOGLE_OAUTH_REFRESH_TOKEN = os.getenv("GOOGLE_OAUTH_REFRESH_TOKEN", "")

BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "audio_files"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
PROCESSED_FILE = BASE_DIR / "processed_ids.json"
AUDIO_DIR.mkdir(exist_ok=True)
TRANSCRIPT_DIR.mkdir(exist_ok=True)

HEADERS = {
    "Authorization": PLAUD_BEARER_TOKEN,
    "Content-Type": "application/json",
}

MIN_DURATION_SEC = 300    # 5分未満の録音はスキップ


# === 処理済みID管理 ===

def load_processed_ids():
    """処理済みファイルIDリストを読み込み"""
    if PROCESSED_FILE.exists():
        return set(json.loads(PROCESSED_FILE.read_text(encoding="utf-8")))
    return set()


def save_processed_id(file_id):
    """処理済みIDを追加保存"""
    ids = load_processed_ids()
    ids.add(file_id)
    PROCESSED_FILE.write_text(json.dumps(sorted(ids), ensure_ascii=False), encoding="utf-8")


# === Plaud API ===

def list_files(limit=100, skip=0):
    """Plaud からファイル一覧を取得"""
    url = f"{PLAUD_API_DOMAIN}/file/simple/web"
    params = {
        "skip": skip,
        "limit": limit,
        "is_trash": 0,
        "sort_by": "edit_time",
        "is_desc": "true",
    }
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != 0:
        raise Exception(f"Plaud API error: {data}")
    return data.get("data_file_list", [])


def download_audio(file_id):
    """音声ファイルをダウンロード"""
    mp3_path = AUDIO_DIR / f"{file_id}.mp3"
    if mp3_path.exists():
        print(f"  [skip] ダウンロード済み")
        return mp3_path

    url = f"{PLAUD_API_DOMAIN}/file/temp-url/{file_id}?is_opus=0"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    temp_url = resp.json().get("temp_url")
    if not temp_url:
        raise Exception("ダウンロードURL取得失敗")

    print(f"  ダウンロード中...")
    resp = requests.get(temp_url, stream=True)
    resp.raise_for_status()
    with open(mp3_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = mp3_path.stat().st_size / 1024 / 1024
    print(f"  完了: {size_mb:.1f}MB")
    return mp3_path


def rename_plaud_file(file_id, new_name):
    """Plaud 上のファイル名を変更"""
    url = f"{PLAUD_API_DOMAIN}/file/{file_id}"
    resp = requests.patch(url, headers=HEADERS, json={"filename": new_name})
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != 0:
        print(f"  [warn] リネーム失敗: {data}")
        return False
    print(f"  Plaud ファイル名変更: {new_name}")
    return True


def generate_title_gemini(transcript_text, max_retries=5):
    """Gemini で文字起こしから簡潔なタイトルを生成"""
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""以下の会議の文字起こしの冒頭部分を読み、この会議の内容を表す簡潔なタイトルを1つだけ生成してください。

## ルール
- 「誰と何の話をしたか」が一目でわかるタイトルにする
- 最大30文字以内
- 参加者の名前がわかれば含める
- 余計な説明や装飾は不要。タイトルのみ出力すること
- 例: 「鈴木・田中: 新規事業の方向性」「チーム定例: Q1振り返り」「遼太郎: キャリア相談」

## 文字起こし（冒頭部分）
{transcript_text[:5000]}
"""

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                request_options={"timeout": 180},
            )
            title = (response.text or "").strip().strip('"').strip("「").strip("」")
            if not title:
                raise Exception("タイトルが空です")
            # 30文字制限
            if len(title) > 30:
                title = title[:30]
            return title
        except Exception as e:
            print(f"    [retry {attempt + 1}/{max_retries}] タイトル生成: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (2 ** attempt))
            else:
                raise


# === 文字起こし (Gemini) ===

def transcribe_with_gemini(mp3_path, max_retries=5):
    """Gemini File API + generateContent で音声を文字起こし"""
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

    # File API でアップロード
    print(f"  Gemini File API にアップロード中...")
    uploaded = genai.upload_file(str(mp3_path), mime_type="audio/mpeg")

    # アップロード完了を待つ（最大 180 秒まで）
    poll_deadline = time.time() + 180
    while uploaded.state.name == "PROCESSING":
        if time.time() > poll_deadline:
            raise Exception(f"Gemini ファイルアップロード処理がタイムアウトしました ({uploaded.name})")
        time.sleep(3)
        uploaded = genai.get_file(uploaded.name)

    if uploaded.state.name == "FAILED":
        raise Exception(f"Gemini ファイルアップロード失敗: {uploaded.state}")

    print(f"  アップロード完了: {uploaded.name}")

    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = """以下の音声ファイルを日本語で文字起こししてください。

## ルール
- タイムスタンプを付けてください（[HH:MM:SS] 形式）
- 話者が変わったタイミングで改行してください
- 話者が複数いる場合は「話者A:」「話者B:」のようにラベルを付けてください（名前が判別できればその名前を使用）
- 聞き取れない箇所は（聞き取り不明）と記載してください
- できるだけ正確に、発言内容をそのまま書き起こしてください
- 「えー」「あのー」などのフィラーは省略して構いません
- 固有名詞・数字・日付は特に正確に記載してください
"""

    try:
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    [prompt, uploaded],
                    request_options={"timeout": 600},
                )
                text = (response.text or "").strip()
                if len(text) < 50:
                    raise Exception(f"文字起こし結果が短すぎます ({len(text)} 文字)")
                return text
            except Exception as e:
                print(f"    [retry {attempt + 1}/{max_retries}] 文字起こし: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    # 指数バックオフ: 10, 20, 40, 80 秒
                    time.sleep(10 * (2 ** attempt))
                else:
                    raise
    finally:
        # アップロード済みファイルを必ず削除（リソースリーク防止）
        try:
            genai.delete_file(uploaded.name)
        except Exception:
            pass


def transcribe_audio(mp3_path, file_id):
    """Gemini API で文字起こし"""
    transcript_path = TRANSCRIPT_DIR / f"{file_id}.txt"
    if transcript_path.exists():
        cached = transcript_path.read_text(encoding="utf-8").strip()
        if len(cached) >= 100:
            print(f"  [skip] 文字起こし済み ({len(cached)} 文字)")
            return cached
        else:
            print(f"  [redo] キャッシュが短すぎるため再実行 ({len(cached)} 文字)")

    if not GEMINI_API_KEY or len(GEMINI_API_KEY) < 10:
        raise Exception(f"GEMINI_API_KEY が無効です (長さ: {len(GEMINI_API_KEY)})")

    size_mb = mp3_path.stat().st_size / 1024 / 1024
    print(f"  文字起こし中 (Gemini 2.5 Flash)... ({size_mb:.1f}MB)")

    text = transcribe_with_gemini(mp3_path)
    if len(text) < 100:
        raise Exception(f"文字起こし結果が短すぎます ({len(text)} 文字)")

    transcript_path.write_text(text, encoding="utf-8")
    print(f"  文字起こし完了: {len(text)} 文字")

    return text


# === アクション抽出 (Gemini) ===

ACTION_PROMPT_TEMPLATE = """あなたは超一流の会議アシスタントです。以下の会議の文字起こしを深く読み込み、実務で即使える議事録を日本語で出力してください。

## 絶対に守るルール
- 必ず日本語で出力すること
- Discord で表示するため Markdown 形式で整形すること
- **すべての指定セクションを必ず出力すること**（該当情報がなければ「- 特になし」と明記）
- 話者名が判別できる場合は必ず実名を使用。判別できない場合のみ「話者A」「話者B」と表記
- 抽象的にまとめず、文字起こしに含まれる具体的な内容（固有名詞・数値・日付・金額・ツール名など）を必ず保持する
- **「特に具体的なアクションがなかった」と安易に結論付けない**。潜在的な次アクション・検討すべき論点も抽出する

## 出力セクション（必ずこの順で、見出しも完全一致）

### 👥 参加者
- 参加者名と（分かる範囲で）役割を箇条書き

### 📅 日時・場所・コンテキスト
- 日時 / 場所 / 会議の目的・背景を 1〜3 行で

### 📋 会議要約
（会議の目的・主な内容・重要な論点・結論を 5〜8 行で、具体的に）

### ✅ アクションリスト
**[名前/話者]**
- [ ] アクション内容（期限があれば記載）

**[名前/話者]**
- [ ] アクション内容

*明示的なアクションが少ない場合でも、文脈から読み取れる「次に取るべき具体的行動」を必ず提案として記載すること。各参加者につき可能な限り 2 項目以上挙げる。*

### 📌 決定事項
- 会議で合意・決定した内容（箇条書き、固有名詞を保持）

### 💡 補足・注意点
- 議論中に出た重要な懸念・気づき・リスク・未解決の論点
- 後でフォローアップすべき話題

### 🌱 ネクストステップ候補
- 次回以降の会議・行動・調査のアイデア（2〜4 個）

## 文字起こし（会議名: {filename}）
{transcript}
"""


def _looks_valid_action_output(text: str) -> bool:
    """アクション抽出出力が最低限の構造を満たしているかチェック"""
    if not text or len(text) < 200:
        return False
    required_headers = ["### 📋 会議要約", "### ✅ アクションリスト", "### 📌 決定事項"]
    return all(h in text for h in required_headers)


def extract_actions_gemini(transcript_text, filename, max_retries=5):
    """Gemini API で会議のアクションリストを抽出（品質チェック付き）"""
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-pro")

    prompt = ACTION_PROMPT_TEMPLATE.format(
        filename=filename,
        transcript=transcript_text[:120000],
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                request_options={"timeout": 300},
            )
            text = (response.text or "").strip()
            if not _looks_valid_action_output(text):
                raise Exception(f"出力が構造要件を満たしていません (長さ: {len(text)})")
            return text
        except Exception as e:
            last_error = e
            print(f"    [retry {attempt + 1}/{max_retries}] アクション抽出: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                # 指数バックオフ: 10, 20, 40, 80 秒
                time.sleep(10 * (2 ** attempt))
            else:
                # 最後の手段: gemini-2.5-flash でフォールバック
                print(f"    [fallback] gemini-2.5-flash で再試行")
                try:
                    fallback_model = genai.GenerativeModel("gemini-2.5-flash")
                    response = fallback_model.generate_content(
                        prompt,
                        request_options={"timeout": 300},
                    )
                    text = (response.text or "").strip()
                    if text:
                        return text
                except Exception as fe:
                    print(f"    [fallback failed] {type(fe).__name__}: {fe}")
                raise last_error


# === Discord 送信 ===

def send_to_discord(text, title=""):
    """Discord Webhook に送信"""
    if not DISCORD_WEBHOOK_URL:
        print("  [warn] DISCORD_WEBHOOK_URL 未設定")
        print(text)
        return False

    header = f"## 🎙️ {title}\n" if title else ""
    content = header + text

    # 2000文字制限で分割
    messages = []
    while content:
        if len(content) <= 2000:
            messages.append(content)
            break
        pos = content[:2000].rfind("\n")
        if pos == -1:
            pos = 2000
        messages.append(content[:pos])
        content = content[pos:].lstrip("\n")

    for i, msg in enumerate(messages):
        resp = requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
        if resp.status_code not in (200, 204):
            print(f"  [error] Discord: {resp.status_code} {resp.text}")
            return False
        if i < len(messages) - 1:
            time.sleep(1)

    print(f"  Discord送信完了 ({len(messages)}メッセージ)")
    return True


# === Google Drive アップロード ===

def get_drive_service():
    """OAuth2 リフレッシュトークンで Google Drive API クライアントを取得"""
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = Credentials(
        token=None,
        refresh_token=GOOGLE_OAUTH_REFRESH_TOKEN,
        client_id=GOOGLE_OAUTH_CLIENT_ID,
        client_secret=GOOGLE_OAUTH_CLIENT_SECRET,
        token_uri="https://oauth2.googleapis.com/token",
    )
    return build("drive", "v3", credentials=creds)


def upload_to_google_drive(title, date_str, transcript_text, actions_text):
    """Google Drive に「タイトル_日付」フォルダを作成し、文字起こしとアクションリストを保存"""
    if not GOOGLE_OAUTH_REFRESH_TOKEN or not GOOGLE_DRIVE_FOLDER_ID:
        print("  [skip] Google Drive 未設定（GOOGLE_OAUTH_REFRESH_TOKEN / GOOGLE_DRIVE_FOLDER_ID）")
        return None

    from googleapiclient.http import MediaInMemoryUpload

    service = get_drive_service()
    folder_name = f"{title}_{date_str}"

    # 同名フォルダが既存か確認
    query = (
        f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' "
        f"and '{GOOGLE_DRIVE_FOLDER_ID}' in parents and trashed = false"
    )
    existing = service.files().list(q=query, fields="files(id)").execute().get("files", [])

    if existing:
        folder_id = existing[0]["id"]
        print(f"  Google Drive: 既存フォルダ使用 ({folder_name})")
    else:
        folder_meta = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [GOOGLE_DRIVE_FOLDER_ID],
        }
        folder = service.files().create(body=folder_meta, fields="id").execute()
        folder_id = folder["id"]
        print(f"  Google Drive: フォルダ作成 ({folder_name})")

    # transcript.txt アップロード
    txt_media = MediaInMemoryUpload(transcript_text.encode("utf-8"), mimetype="text/plain")
    service.files().create(
        body={"name": "transcript.txt", "parents": [folder_id]},
        media_body=txt_media,
    ).execute()

    # actions.md アップロード
    md_media = MediaInMemoryUpload(actions_text.encode("utf-8"), mimetype="text/markdown")
    service.files().create(
        body={"name": "actions.md", "parents": [folder_id]},
        media_body=md_media,
    ).execute()

    print(f"  Google Drive: アップロード完了 (transcript.txt, actions.md)")
    return folder_id


# === パイプライン ===

def process_file(file_info):
    """1ファイルの完全パイプライン: DL → 文字起こし → アクション抽出 → Discord"""
    file_id = file_info["id"]
    filename = file_info["filename"]
    duration_min = file_info["duration"] / 1000 / 60

    print(f"\n{'=' * 50}")
    print(f"📁 {filename} ({duration_min:.0f}分)")
    print(f"{'=' * 50}")

    # 1. ダウンロード
    mp3_path = download_audio(file_id)

    # 2. 文字起こし
    transcript = transcribe_audio(mp3_path, file_id)

    # 3. タイトル生成 & Plaud リネーム
    print("  タイトル生成中 (Gemini)...")
    title = generate_title_gemini(transcript)
    print(f"  生成タイトル: {title}")
    rename_plaud_file(file_id, title)
    display_name = f"{filename} - {title}"

    # 4. アクション抽出 (Gemini)
    print("  アクション抽出中 (Gemini)...")
    actions = extract_actions_gemini(transcript, display_name)
    print("  アクション抽出完了")

    # 結果をファイルに保存
    result_path = TRANSCRIPT_DIR / f"{file_id}_actions.md"
    result_path.write_text(f"# {display_name}\n\n{actions}", encoding="utf-8")

    # 5. Discord 送信
    send_to_discord(actions, title=display_name)

    # 6. Google Drive アップロード
    date_str = datetime.now().strftime("%Y-%m-%d")
    try:
        upload_to_google_drive(title, date_str, transcript, actions)
    except Exception as e:
        print(f"  [warn] Google Drive アップロード失敗: {type(e).__name__}: {e}")

    # 7. 音声ファイル削除（ディスク節約）
    if mp3_path.exists():
        mp3_path.unlink()
        print(f"  音声ファイル削除済み")

    # 8. 処理済みIDを記録
    save_processed_id(file_id)

    return actions


def run_auto():
    """自動実行: 未処理の新しいファイルだけ処理"""
    print("🔄 新規ファイルチェック中...")
    files = list_files(limit=50)
    processed = load_processed_ids()

    new_files = [
        f for f in files
        if f["id"] not in processed
        and f["duration"] / 1000 >= MIN_DURATION_SEC  # 5分未満はスキップ
    ]

    if not new_files:
        print("✅ 新しいファイルはありません")
        return

    # GitHub Actions の時間制限対策: 1回の実行で最大3件まで
    MAX_PER_RUN = 3
    if len(new_files) > MAX_PER_RUN:
        print(f"📡 {len(new_files)} 件の新規ファイルあり（今回は最新 {MAX_PER_RUN} 件を処理）\n")
        new_files = new_files[:MAX_PER_RUN]
    else:
        print(f"📡 {len(new_files)} 件の新規ファイルを処理します\n")
    for f in new_files:
        try:
            process_file(f)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"  [error] {f['filename']}: {type(e).__name__}: {e}")
            print(tb)
            # traceback の末尾5行までDiscordに含める（調査しやすく）
            tb_tail = "\n".join(tb.strip().split("\n")[-8:])
            send_to_discord(
                f"⚠️ 処理エラー: **{f['filename']}** (id=`{f['id']}`)\n"
                f"```\n{type(e).__name__}: {e}\n\n{tb_tail}\n```\n"
                f"※ processed_ids には未登録のため、次回実行時に再試行されます。",
                title="処理エラー通知",
            )

    print("\n✅ 全処理完了")


# === CLI ===

def main():
    if len(sys.argv) < 2:
        print("使い方:")
        print("  python3 plaud_to_discord.py auto               # 新規ファイル自動処理 (GitHub Actions用)")
        print("  python3 plaud_to_discord.py list               # ファイル一覧")
        print("  python3 plaud_to_discord.py process <id>       # 1件をフル処理 (DL→文字起こし→アクション→Discord)")
        print("  python3 plaud_to_discord.py transcribe <id>    # 1件を文字起こしのみ")
        print("  python3 plaud_to_discord.py discord <text>     # Discord にテスト送信")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "auto":
        if not PLAUD_BEARER_TOKEN:
            print("❌ PLAUD_BEARER_TOKEN が未設定"); sys.exit(1)
        if not GEMINI_API_KEY:
            print("❌ GEMINI_API_KEY が未設定"); sys.exit(1)
        run_auto()

    elif cmd == "list":
        if not PLAUD_BEARER_TOKEN:
            print("❌ PLAUD_BEARER_TOKEN が未設定"); sys.exit(1)
        files = list_files(limit=200)
        processed = load_processed_ids()
        print(f"{'ID':<34} {'日時':<22} {'時間':>6}  {'状態'}")
        print("-" * 80)
        for f in files:
            dur = f["duration"] / 1000 / 60
            tid = f["id"]
            status = "✅" if tid in processed else "❌"
            print(f"{tid}  {f['filename']:<22} {dur:>5.0f}分  {status}")
        print(f"\n合計: {len(files)} 件 (処理済み: {len(processed)}件)")

    elif cmd == "process" and len(sys.argv) > 2:
        if not PLAUD_BEARER_TOKEN or not GEMINI_API_KEY:
            print("❌ 必要な環境変数が未設定"); sys.exit(1)
        file_id = sys.argv[2]
        files = list_files(limit=200)
        target = next((f for f in files if f["id"] == file_id), None)
        if target:
            process_file(target)
        else:
            print(f"❌ ファイルが見つかりません: {file_id}")

    elif cmd == "transcribe" and len(sys.argv) > 2:
        if not PLAUD_BEARER_TOKEN or not GEMINI_API_KEY:
            print("❌ 必要な環境変数が未設定"); sys.exit(1)
        file_id = sys.argv[2]
        print(f"🎙️ 文字起こし開始: {file_id}")
        mp3_path = download_audio(file_id)
        transcribe_audio(mp3_path, file_id)

    elif cmd == "discord":
        text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "テスト送信"
        send_to_discord(text)

    else:
        print(f"不明なコマンド: {cmd}")


if __name__ == "__main__":
    main()
