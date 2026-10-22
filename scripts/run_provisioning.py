import sys
import logging
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from stock_ai.cloud.discord_provisioner import DiscordProvisioner

def main():
    p = DiscordProvisioner()
    res = p.provision_all()
    print("Channels provisioned:")
    for k, v in res["channels"].items():
        print(f"  {k}: {v}")
    print("Webhooks:")
    for k, v in res["webhooks"].items():
        print(f"  {k}: {v}")

    p.save_webhooks_to_env_file(res["webhooks"])
    print("Saved to .env file.")

    # Post welcome message specifically to the newly added portfolio room
    p.send_welcome_messages(res["webhooks"])
    print("Sent welcome cards successfully!")

if __name__ == "__main__":
    main()
