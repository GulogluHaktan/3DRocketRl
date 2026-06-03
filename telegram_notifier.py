from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str, timeout: float = 5.0):
        self.bot_token = bot_token.strip()
        self.chat_id = str(chat_id).strip()
        self.timeout = float(timeout)
        self.enabled = bool(self.bot_token and self.chat_id)

    @classmethod
    def from_config(cls, config_path: str | Path):
        config_path = Path(config_path)
        data = {}
        if config_path.exists():
            try:
                with config_path.open(encoding="utf-8") as file:
                    data = json.load(file)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[Telegram] config okunamadi: {exc}", flush=True)
                return None

        bot_token = data.get("bot_token") or os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = data.get("chat_id") or os.getenv("TELEGRAM_CHAT_ID", "")
        timeout = data.get("timeout", 5.0)
        notifier = cls(bot_token=bot_token, chat_id=chat_id, timeout=timeout)
        return notifier if notifier.enabled else None

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = urllib.parse.urlencode({
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": "true",
        }).encode("utf-8")
        request = urllib.request.Request(url, data=payload, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return 200 <= response.status < 300
        except OSError as exc:
            print(f"[Telegram] mesaj gonderilemedi: {exc}", flush=True)
            return False
