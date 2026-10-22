import sys
import logging
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

from stock_ai.cloud.discord_bot import DiscordPollingBot

if __name__ == "__main__":
    print("=================================================================")
    print("🤖 [Stock-AI] Interactive Discord Analyst Polling Bot Starting...")
    print("=================================================================")
    bot = DiscordPollingBot(poll_interval=3.0)
    bot.run()
