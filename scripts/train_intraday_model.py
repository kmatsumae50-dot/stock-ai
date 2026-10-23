import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path("src").resolve()))

import logging
logging.basicConfig(level=logging.INFO)

from stock_ai.models.intraday import train_pooled_cross_sectional_intraday_model

print("Starting Pooled Cross-Sectional Intraday Model Training...")
bundle, feats, metrics = train_pooled_cross_sectional_intraday_model()
print("Trained successfully!")
print("Features count:", len(feats))
print("Validation AUC (CS Outperform):", f"{metrics['validation_auc']:.4f}")
print("Validation AUC (Clean Win):", f"{metrics['validation_clean_auc']:.4f}")
print("Top 10 features:")
for f in metrics["top_features"][:10]:
    print(f"  - {f['feature']}: {f['importance']}")
