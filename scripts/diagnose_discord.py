import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))

from stock_ai.cloud.discord_provisioner import DiscordProvisioner

p = DiscordProvisioner()
g_id = p.get_target_guild_id()
channels = p.get_existing_channels(g_id)

print(f"=== GUILD ID: {g_id} ===")
for ch in channels:
    print(f"Channel: {ch.get('name')} | ID: {ch.get('id')} | Type: {ch.get('type')}")

print("\n=== RECENT MESSAGES ===")
for ch in channels:
    if ch.get("type") == 0:  # text channel
        ch_id = ch.get("id")
        ch_name = ch.get("name")
        print(f"\n--- Channel: {ch_name} ({ch_id}) ---")
        msgs = p._api_request(f"/channels/{ch_id}/messages?limit=5")
        if isinstance(msgs, list):
            for m in msgs:
                author = m.get("author", {})
                print(
                    f"  [ID: {m.get('id')}] User: {author.get('username')} (bot={author.get('bot')}) | Content: {repr(m.get('content'))}"
                )
        else:
            print("  (Failed to fetch messages or empty)")
