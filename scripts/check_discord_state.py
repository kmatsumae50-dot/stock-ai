import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from stock_ai.cloud.discord_provisioner import DiscordProvisioner

prov = DiscordProvisioner()
guild_id = prov.get_target_guild_id()
channels = prov.get_existing_channels(guild_id)
print(f"Target Guild: {guild_id}")
cats = {c["id"]: c["name"] for c in channels if c.get("type") == 4}
print("Categories:", cats)
for c in channels:
    if c.get("type") == 0:
        p_name = cats.get(c.get("parent_id"), "None")
        print(f"Channel: {c['name']} (id={c['id']}, parent_id={c.get('parent_id')}, parent={p_name})")
