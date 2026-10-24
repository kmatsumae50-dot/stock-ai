"""Dispatch Carver-scale 4-masterpiece integrated Daytrade report to Discord."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure src is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_ai.cloud.discord_bot import build_premarket_daytrade_forecast_payloads
from stock_ai.cloud.notifications import _post_single_webhook_payload, get_configured_channel_webhooks


def main() -> None:
    webhooks = get_configured_channel_webhooks()
    url = webhooks.get("daytrade")
    if not url:
        print("ERROR: daytrade webhook not found in get_configured_channel_webhooks().")
        return

    print(f"Target Daytrade Webhook: {url[:45]}...")
    payloads = build_premarket_daytrade_forecast_payloads()
    print(f"Generated {len(payloads)} daytrade forecast payloads.")

    for idx, p in enumerate(payloads, 1):
        code = _post_single_webhook_payload(url, p)
        print(f"[{idx}/{len(payloads)}] Posted to Discord (HTTP status: {code})")
        time.sleep(1.0)

    print("Successfully delivered Daytrade Sniper report!")


if __name__ == "__main__":
    main()
