#!/usr/bin/env python3
"""30分以上の未処理ファイルを全て1件ずつ処理するバッチスクリプト"""
import sys
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

from plaud_to_discord import (
    list_files, load_processed_ids, download_audio, transcribe_audio,
    generate_title_gemini, rename_plaud_file, extract_actions_gemini,
    send_to_discord, save_processed_id, upload_to_google_drive, TRANSCRIPT_DIR
)
from datetime import datetime
import traceback

MIN_DURATION_MIN = 30

def main():
    print("=" * 60)
    print("バッチ処理開始: 30分以上の未処理ファイル全件")
    print("=" * 60)

    files = list_files(limit=200)
    processed = load_processed_ids()

    targets = [
        f for f in files
        if f["id"] not in processed
        and f["duration"] / 1000 / 60 >= MIN_DURATION_MIN
    ]

    print(f"\n対象: {len(targets)} 件\n")

    success = 0
    errors = []

    for i, f in enumerate(targets):
        file_id = f["id"]
        filename = f["filename"]
        duration_min = f["duration"] / 1000 / 60

        print(f"\n{'=' * 60}")
        print(f"[{i+1}/{len(targets)}] {filename} ({duration_min:.0f}分)")
        print(f"{'=' * 60}")

        try:
            # 1. ダウンロード
            print("  1/6 ダウンロード中...")
            mp3_path = download_audio(file_id)

            # 2. 文字起こし
            print("  2/6 文字起こし中...")
            transcript = transcribe_audio(mp3_path, file_id)

            # 3. タイトル生成 & リネーム
            print("  3/6 タイトル生成中...")
            title = generate_title_gemini(transcript)
            print(f"  生成タイトル: {title}")
            rename_plaud_file(file_id, title)
            display_name = f"{filename} - {title}"

            # 4. アクション抽出
            print("  4/6 アクション抽出中...")
            actions = extract_actions_gemini(transcript, display_name)
            result_path = TRANSCRIPT_DIR / f"{file_id}_actions.md"
            result_path.write_text(f"# {display_name}\n\n{actions}", encoding="utf-8")

            # 5. Discord 送信
            print("  5/6 Discord送信中...")
            send_to_discord(actions, title=display_name)

            # 6. Google Drive アップロード
            print("  6/6 Google Drive アップロード中...")
            date_str = datetime.now().strftime("%Y-%m-%d")
            upload_to_google_drive(title, date_str, transcript, actions)

            # 音声ファイル削除
            if mp3_path.exists():
                mp3_path.unlink()

            # 処理済み記録
            save_processed_id(file_id)
            success += 1
            print(f"  ✅ 完了!")

        except Exception as e:
            tb = traceback.format_exc()
            print(f"  ❌ エラー: {type(e).__name__}: {e}")
            print(tb)
            errors.append((filename, str(e)))
            # エラーでも次のファイルへ続行

    print(f"\n{'=' * 60}")
    print(f"バッチ処理完了: 成功 {success}/{len(targets)} 件")
    if errors:
        print(f"エラー {len(errors)} 件:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
