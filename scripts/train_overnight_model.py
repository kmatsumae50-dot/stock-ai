import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))

import logging
logging.basicConfig(level=logging.INFO)

from stock_ai.models.overnight import train_pooled_cross_sectional_overnight_model

print("Starting Pooled Cross-Sectional Overnight Model Training...")
model, feats, metrics = train_pooled_cross_sectional_overnight_model()
print("Trained successfully!")
print("Features count:", len(feats))
print("Validation AUC:", f"{metrics['validation_auc']:.4f}")
print("Validation Accuracy:", f"{metrics['validation_accuracy']:.4f}")
