import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from stock_ai.models.intraday import predict_daytrade_sniper

from datetime import datetime

tickers = [("7012.T", "川崎重工業"), ("5803.T", "フジクラ"), ("7203.T", "トヨタ自動車")]
for code, name in tickers:
    res = predict_daytrade_sniper(code, name=name, now_dt=datetime(2026, 10, 23, 7, 45))
    if not res:
        print(f"Failed to get {code}")
        continue
    print("=" * 60)
    print(f"【{res['judgment']}】{res['ticker']} {res['name']} ({res['phase_title']})")
    print(f"波形: {res['waveform_title']}")
    print(f"解説: {res['waveform_description']}")
    print("時間帯別軌道:")
    for k, v in res["timeline_items"]:
        print(f"  {k}: {v}")
    print(f"🛒 仕込みゾーン: {res['best_entry_zone']}")
    print(f"🎯 第1利確: {res['target_1_price']:,.1f}円 ({res['target_1_pct']:+.1f}%)")
    print(f"🚀 最終ターゲット: {res['target_final_price']:,.1f}円 ({res['target_final_pct']:+.1f}%)")
    print(f"🛡️ 損切り: {res['stop_loss_price']:,.1f}円 ({res['stop_loss_pct']:+.1f}%)")
