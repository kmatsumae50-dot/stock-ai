import sys
import io
import time
import json
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from stock_ai.cloud.discord_bot import build_premarket_daytrade_forecast_payloads
from stock_ai.cloud.notifications import get_configured_channel_webhooks, _post_single_webhook_payload

webhooks = get_configured_channel_webhooks()
daytrade_url = webhooks.get("daytrade")
print("Target Daytrade Webhook URL:", daytrade_url)

if not daytrade_url:
    print("ERROR: daytrade webhook URL not found!")
    sys.exit(1)

payloads = build_premarket_daytrade_forecast_payloads()
print(f"Generated {len(payloads)} payloads for daytrade channel.")

for idx, p in enumerate(payloads):
    print(f"Delivering payload {idx+1}/{len(payloads)}: {p.get('username')}...")
    code = _post_single_webhook_payload(daytrade_url, p)
    print(f"Status Code: {code}")
    time.sleep(0.8)

print("SUCCESS: Delivered revamped waveform forecast to ⚡｜当日デイトレ特化予想!")
