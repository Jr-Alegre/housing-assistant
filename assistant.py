"""Housing-search personal assistant: texts you on Telegram until you find a place.

Run hourly (GitHub Actions does this). Each run it:
  1. reads your replies to the bot (reply "found" to stop it for good),
  2. sends a message if one of the SEND_HOURS slots is due,
  3. saves its memory to state.json.

Local testing:  python assistant.py --dry-run --force
"""

import html
import json
import os
import random
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------

ASSISTANT_NAME = "Max"
TIMEZONE = ZoneInfo("Europe/Amsterdam")
SEND_HOURS = [7, 10, 13, 16, 19, 22]
# If GitHub runs the job late, still send if we're within this many hours of the slot.
LATE_WINDOW = timedelta(hours=2)

SITES = {
    "Pararius": "https://www.pararius.com",
    "Kamernet": "https://kamernet.nl",
    "Holland2Stay": "https://holland2stay.com",
    "MyHousing": "https://www.myhousing.nl",
    "Househunting": "https://www.househunting.nl",
    "Lightcity Housing": "https://www.lightcityhousing.nl",
    "Brick Vastgoed": "https://www.brickvastgoed.nl",
    "Rotsvast": "https://www.rotsvast.nl",
}

GEMINI_MODELS = [m for m in (os.environ.get("GEMINI_MODEL"), "gemini-flash-latest", "gemini-2.5-flash") if m]
STOP_WORDS = {"found", "/found"}
STATE_FILE = Path(__file__).with_name("state.json")
ENV_FILE = Path(__file__).with_name(".env")

FALLBACK_MESSAGES = [
    "Hey, quick nudge: new listings drop all the time. Got 5 minutes for a scroll?",
    "Checking in! Have you looked at the sites yet today? Someone's dream room gets posted every hour.",
    "Reminder from your housing buddy: refresh those searches. The early bird gets the viewing 🏠",
    "You know what I'm going to say... time for a quick housing round!",
    "Small habit, big payoff: one quick check of the listings right now?",
    "Don't let a good place slip past you. Quick look at the sites?",
    "Housing check! Even a two-minute scroll counts.",
    "Hey you, go find your future home. I'll wait 😄",
    "New hour, new listings. Take a peek?",
    "Consistency wins this game. Quick check of the sites?",
]
FALLBACK_CONGRATS = "YOU FOUND A PLACE!! 🎉 I'm so happy for you. That's my last reminder. Enjoy your new home!"

# ----------------------------------------------------------------------
# STATE
# ----------------------------------------------------------------------


def load_state() -> dict:
    state = {
        "start_date": None,
        "last_slot": "",
        "update_offset": 0,
        "found": False,
        "messages_sent": 0,
        "recent_messages": [],
        "pending_replies": [],
    }
    if STATE_FILE.exists():
        state.update(json.loads(STATE_FILE.read_text(encoding="utf-8")))
    return state


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def due_slot(now: datetime, last_slot: str) -> str | None:
    """Return the key of the slot that should be sent now, or None."""
    for hour in reversed(SEND_HOURS):
        slot_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if slot_time <= now:
            key = slot_time.strftime("%Y-%m-%dT%H")
            if now - slot_time <= LATE_WINDOW and key > last_slot:
                return key
            return None
    return None


# ----------------------------------------------------------------------
# HTTP HELPERS
# ----------------------------------------------------------------------


def post_json(url: str, payload: dict, headers: dict | None = None) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def telegram(method: str, payload: dict) -> dict:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    result = post_json(f"https://api.telegram.org/bot{token}/{method}", payload)
    if not result.get("ok"):
        raise RuntimeError(f"Telegram {method} failed: {result}")
    return result


# ----------------------------------------------------------------------
# TELEGRAM
# ----------------------------------------------------------------------


def read_replies(state: dict) -> list[str]:
    """Fetch new messages you sent to the bot. Messages from anyone else are ignored."""
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    updates = telegram(
        "getUpdates",
        {"offset": state["update_offset"], "timeout": 0, "allowed_updates": ["message"]},
    )["result"]
    texts = []
    for update in updates:
        state["update_offset"] = update["update_id"] + 1
        msg = update.get("message") or {}
        if str(msg.get("chat", {}).get("id")) == chat_id and msg.get("text"):
            texts.append(msg["text"])
    return texts


def send(text_html: str, dry_run: bool) -> None:
    if dry_run:
        print("----- WOULD SEND -----\n" + text_html + "\n----------------------")
        return
    telegram(
        "sendMessage",
        {
            "chat_id": os.environ["TELEGRAM_CHAT_ID"],
            "text": text_html,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        },
    )


# ----------------------------------------------------------------------
# AI WRITING (Gemini free tier, with built-in fallback)
# ----------------------------------------------------------------------

PERSONA = (
    f"You are {ASSISTANT_NAME}, a warm, upbeat friend who texts the user on Telegram to keep them "
    "motivated while they search for a place to rent in the Netherlands. Write exactly like a real "
    "person texting a friend: short (1-3 sentences, under 60 words), casual, varied, at most one "
    "emoji, no hashtags, no formal greetings, no sign-off. Never mention being an AI, a bot or an "
    "assistant. Do not include links or list all the websites (links are added automatically), "
    "but you may mention one or two site names naturally. Never reuse the wording of your recent messages."
)


def gemini(prompt: str) -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    payload = {
        "system_instruction": {"parts": [{"text": PERSONA}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 1.1, "maxOutputTokens": 2048},
    }
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            data = post_json(url, payload, {"x-goog-api-key": key})
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
            if text:
                return text
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                print(f"Gemini {model} rate-limited; using fallback.", file=sys.stderr)
                return None
            print(f"Gemini model {model} failed: HTTP {exc.code}", file=sys.stderr)
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError) as exc:
            print(f"Gemini model {model} failed: {exc}", file=sys.stderr)
    return None


def part_of_day(hour: int) -> str:
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    if hour < 21:
        return "evening"
    return "late evening"


def write_reminder(state: dict, now: datetime) -> str:
    day_number = (now.date() - datetime.fromisoformat(state["start_date"]).date()).days + 1
    recent = "\n".join(f"- {m}" for m in state["recent_messages"]) or "(none yet)"
    replies = "\n".join(f"- {r}" for r in state["pending_replies"])
    prompt = (
        f"It's {now:%A} {now:%H:%M} ({part_of_day(now.hour)}). "
        f"This is day {day_number} of their housing search. "
        f"Sites they check: {', '.join(SITES)}.\n\n"
        f"Your recent messages (don't repeat these):\n{recent}\n\n"
    )
    if replies:
        prompt += f"They replied to you since your last message (react naturally):\n{replies}\n\n"
    if state["messages_sent"] == 0:
        prompt += "This is your very first message to them: introduce yourself in a casual way.\n\n"
    prompt += "Write your next message nudging them to check the housing sites now."

    text = gemini(prompt)
    if text:
        return text
    options = [m for m in FALLBACK_MESSAGES if m not in state["recent_messages"]] or FALLBACK_MESSAGES
    return random.choice(options)


def write_congrats(state: dict) -> str:
    days = state["messages_sent"] // len(SEND_HOURS) + 1
    text = gemini(
        f"The user just told you they FOUND a place after about {days} days of searching! "
        "Write a short, genuinely excited congratulations and mention this is your last reminder."
    )
    return text or FALLBACK_CONGRATS


def format_message(text: str, first: bool) -> str:
    links = " · ".join(f'<a href="{url}">{html.escape(name)}</a>' for name, url in SITES.items())
    message = f"{html.escape(text)}\n\n{links}"
    if first:
        message += '\n\n<i>(Reply "found" when you get a place and I\'ll stop.)</i>'
    return message


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------


def set_output(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")


def load_env_file() -> None:
    """Local testing only: read KEY=VALUE lines from .env (never committed)."""
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"'))


def main(argv: list[str]) -> None:
    load_env_file()
    dry_run = "--dry-run" in argv
    force = "--force" in argv
    state = load_state()
    now = datetime.now(TIMEZONE)

    if state["found"]:
        print("Already found a place; nothing to do.")
        set_output("found", "true")
        return

    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        replies = read_replies(state)
        state["pending_replies"] = (state["pending_replies"] + replies)[-10:]
        if any(r.strip().lower().strip("!. ") in STOP_WORDS for r in replies):
            send(html.escape(write_congrats(state)), dry_run)
            if dry_run:
                return
            state["found"] = True
            save_state(state)
            set_output("found", "true")
            print("Found a place! Stopping.")
            return

    slot = due_slot(now, state["last_slot"])
    if not slot and not force:
        if not dry_run:
            save_state(state)
        print(f"{now:%H:%M}: no slot due.")
        return

    state["start_date"] = state["start_date"] or now.date().isoformat()
    text = write_reminder(state, now)
    send(format_message(text, first=state["messages_sent"] == 0), dry_run)

    if not dry_run:
        state["last_slot"] = slot or state["last_slot"]
        state["messages_sent"] += 1
        state["recent_messages"] = (state["recent_messages"] + [text])[-6:]
        state["pending_replies"] = []
        save_state(state)
    print(f"Sent slot {slot or 'forced'}.")


if __name__ == "__main__":
    main(sys.argv[1:])
