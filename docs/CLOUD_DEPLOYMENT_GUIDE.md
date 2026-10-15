# ☁️ Stock-AI 完全クラウド自動運用 & スマホ閲覧ガイド

本ガイドでは、**PCの電源を完全にシャットダウン**していても、クラウド上で毎日自動的に：
1. 東証プライム20銘柄＋外部市場データの最新取得
2. 20銘柄の Overnight (C2O: 翌朝始値ギャップ) 予測 & 動的Quantile判定
3. 仮想100万円ファンド（Overnight Paper Trader）の日次損益更新
4. 💎 Value & Turnaround（将来回復しそうな激安株）スキャナーの更新
5. Discord / Slack への毎朝サマリー自動通知
を実行し、**スマホやタブレットからいつでも最新ダッシュボードを閲覧・操作**できる環境の構築手順を解説します。

---

## 🎯 構成の全体像

```
[GitHub リポジトリ (Private/Public)]
      │
      ├── ① 定時自動バッチ (JST 16:00 / 07:30)
      │     └── GitHub Actions (daily_cloud_predict.yml)
      │           ├── データ取得 & 推論 & 100万ファンド更新 & 激安株スキャン
      │           ├── スナップショット JSON 自動コミット
      │           └── Discord/Slack へ自動通知 (Webhook)
      │
      └── ② スマホWebダッシュボード (常時アクセス可能)
            ├── 選択肢A: Streamlit Community Cloud (完全無料・推奨)
            └── 選択肢B: Google Cloud Run (独自ドメイン・企業向け)
```

---

## 🚀 3分で完了！セットアップ手順

### ステップ 1: GitHub へのプッシュと Actions の有効化

1. **GitHub にリポジトリを作成してプッシュ**:
   ```bash
   git remote add origin https://github.com/<あなたのユーザー名>/stock-ai.git
   git branch -M master
   git push -u origin master
   ```
2. **GitHub Actions の実行権限を許可**:
   - GitHub リポジトリの **[Settings]** → **[Actions]** → **[General]** を開きます。
   - **Workflow permissions** で **「Read and write permissions」** にチェックを入れて **[Save]** します。
   - これにより、GitHub Actions が日次実行した予測スナップショットをリポジトリへ自動コミット・保存できるようになります。

---

### ステップ 2: 【無料・簡単】Streamlit Community Cloud でスマホ閲覧

PCのブラウザだけでなく、**スマホのホーム画面にアイコン追加**してアプリ感覚で毎朝チェックできます。

1. **[share.streamlit.io](https://share.streamlit.io/)** にアクセスし、GitHub アカウントでログイン。
2. **[New app]** をクリックし、以下を設定：
   - **Repository**: `<あなたのユーザー名>/stock-ai`
   - **Branch**: `master`
   - **Main file path**: `streamlit_app.py`
3. **[Deploy!]** をクリック（約2分でWebアプリが起動します）。
4. **スマホからの利用**:
   - 発行されたURL（`https://<app-name>.streamlit.app`）をスマホのブラウザで開きます。
   - Safari（iOS）なら「共有」→「ホーム画面に追加」、Chrome（Android）ならメニュー→「ホーム画面に追加」を行うと、ネイティブアプリ感覚でワンタップ起動できます。
   - **画面上部の「☁️ クラウド同期実行」ボタン**を押せば、いつでも最新データへの強制更新が可能です。

---

### ステップ 3: 毎朝 Discord / Slack へ自動通知を飛ばす（任意設定）

毎朝の寄り付き前に、スマホへ自動的に「本日の注目銘柄」が通知されます。

1. **Discord で Webhook URL を取得**:
   - Discord の通知用チャンネルの「設定（歯車マーク）」→「連携」→「ウェブフックを作成」。
   - 「ウェブフックURLをコピー」をクリック。
2. **GitHub の Secrets に登録**:
   - GitHub リポジトリの **[Settings]** → **[Secrets and variables]** → **[Actions]** を開きます。
   - **[New repository secret]** をクリック：
     - **Name**: `STOCK_AI_WEBHOOK_URL`
     - **Secret**: コピーした Discord の Webhook URL
   - **[Add secret]** で保存。
3. **通知内容の例**:
   ```
   📊 Stock AI Daily Cloud Pipeline (2026-10-14)
   
   🏆 Overnight Q5 (明朝ギャップアップ期待 Top 3):
     1. 7974.T 任天堂 (ゲーム・エンタメ): 61.0% (Q5 - 強気ギャップアップ帯)
     2. 9983.T ファーストリテイ (小売・内需): 59.2% (Q5 - 強気ギャップアップ帯)
     3. 7267.T 本田技研工業 (自動車): 59.0% (Q5 - 強気ギャップアップ帯)

   💰 仮想100万円ファンド (Overnight Paper Trader):
     ・総資産: 1,328,500 円 (+32.9%)
     ・勝率: 61.2% | Sharpe: 2.14 | MaxDD: -6.8%
     ・本日エントリー予定: 7974.T, 9983.T

   💎 Value & Turnaround (将来回復しそうな激安株 Top 3):
     1. 7201.T 日産自動車: スコア 72.0点 (52週高値比: -39.1%, PBR: 0.20倍) — 💎 超割安・底打ち反転圏 ⭐ 始動
     2. 7974.T 任天堂: スコア 60.2点 (52週高値比: -41.1%, PBR: 3.27倍) — 👀 調整監視圏 ⭐ 始動
     3. 7011.T 三菱重工業: スコア 59.1点 (52週高値比: -28.3%, PBR: 3.95倍) — 👀 調整監視圏
   ```

---

### ステップ 4: 【上級者向け】Google Cloud Run へのコンテナデプロイ

企業環境や独自ドメイン、SLAを求める場合は、リポジトリに含まれる `Dockerfile` を使って Google Cloud Run へ1コマンドでデプロイできます。

1. **Google Cloud SDK の初期化**:
   ```bash
   gcloud init
   gcloud auth configure-docker
   ```
2. **Cloud Run へデプロイ**:
   ```bash
   gcloud run deploy stock-ai \
     --source . \
     --region asia-northeast1 \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 2
   ```
3. デプロイ完了時に発行される `https://stock-ai-xxx.a.run.app` からダッシュボードへアクセスできます。

---

## ⚙️ クラウド自動実行パイプラインの動作仕様

- **CLIコマンド**: `python -m stock_ai.cli run-cloud-daily-pipeline`
- **実行スケジュール**:
  - **東証大引け後**: 平日 16:00 JST (UTC 07:00: `0 7 * * 1-5`) — 当日の大引け確定値と外部市場の速報値をもとに夜間ギャップを予測。
  - **東証寄り付き前**: 平日 07:30 JST (UTC 22:30: `30 22 * * 0-4`) — 米国市場終了後の最新動向を織り込んで最終判定を確定。
- **Self-Healing（自己修復機能）**:
  万が一、クラウドインスタンスの初回起動等でモデルファイルが欠落している場合でも、パイプラインが自動的に軽量学習・キャリブレーションを行ってから推論を実行します（エラーで停止しません）。
