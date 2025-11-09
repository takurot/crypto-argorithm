# Crypto Algorithm - 仮想通貨価格変動予測システム

BTC-USDの価格変動を予測する機械学習ベースの分析・予測システムです。マクロ経済指標、オンチェーンメトリクス、センチメントデータ、テクニカル指標を統合し、堅牢なウォークフォワード検証とレジーム適応メカニズムを備えています。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## 🎯 主な特徴

- **マルチソースデータ統合**: 10社以上の取引所価格、FRED/Yahoo Finance、Fear & Greed Index、ニュースセンチメント
- **クロスアセット相関**: BTC/ETH/SOL/BNBの相互依存性を捉える集合的予測アプローチ
- **レジーム適応**: HMM/Drawdown/ボラティリティの合成レジーム検出と動的重み調整
- **厳格な検証**: Purged/Embargoed Walk-Forward評価で時間的漏洩を防止
- **運用自動化**: MLflow実験管理、Airflowジョブスケジューリング、Streamlit監視ダッシュボード

## 📊 精度目標

| 指標 | 目標値 | 備考 |
|------|--------|------|
| Directional Accuracy (1D) | > 0.57 | 90日ローリング平均 |
| Directional Accuracy (3D) | > 0.58 | 3日移動平均リターン |
| RMSE | < 0.025 | 過去最良モデル比+10%以内 |
| 年率Sharpe比 | > 0.7 | 手数料0.1%込み |
| 最大ドローダウン | < 15% | バックテスト期間 |

## 🏗️ プロジェクト構造

```
crypt-argorithm/
├── data/                    # データ取得・保存層
│   ├── raw/                 # 生データキャッシュ
│   ├── processed/           # 前処理済み（VWAP集約）
│   └── features/            # 特徴量エンジニアリング済み
├── models/                  # 学習済みモデル保存
│   ├── baseline/
│   ├── lightgbm/
│   └── ensemble/
├── scripts/                 # 実行スクリプト
│   ├── crypto_kpi_analysis.py  # 相関分析
│   ├── data_collection.py      # データ収集
│   ├── feature_engineering.py  # 特徴量生成
│   ├── train.py                # モデル訓練
│   ├── backtest.py             # WF評価
│   └── predict.py              # 推論
├── src/                     # 再利用可能モジュール
│   ├── data/                # コレクター・集約・バリデーション
│   ├── features/            # ラグ・テクニカル・レジーム・PCA
│   ├── models/              # ベースライン・LightGBM・アンサンブル
│   └── evaluation/          # メトリクス・WF・トレードシミュ
├── dashboard/               # Streamlit監視ダッシュボード
├── prompt/                  # 仕様書・計画書
│   ├── spec.md              # 相関分析仕様
│   ├── spec2.md             # 価格予測仕様
│   └── plan.md              # 実装計画
├── output/                  # 分析結果（CSV, PNG, MD）
├── ref/                     # 参考実装
│   └── crypt-arbitrage/     # 取引所価格取得
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```

## 🚀 クイックスタート

### 1. 環境構築

```bash
# リポジトリクローン
git clone https://github.com/takurot/crypto-argorithm.git
cd crypto-argorithm

# Python仮想環境作成（Python 3.11+推奨）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージインストール
pip install -r requirements.txt
```

### 2. APIキー設定

`.env.example` をコピーして `.env` を作成し、各APIキーを設定：

```bash
cp .env.example .env
```

`.env` の内容:
```env
# マクロ経済データ（FRED）
FRED_API_KEY=your_fred_api_key

# オンチェーン指標（Glassnode）- 任意
GLASSNODE_API_KEY=your_glassnode_api_key

# デリバティブ（Coinglass）- 任意
COINGLASS_API_KEY=your_coinglass_api_key

# ソーシャルメトリクス（Santiment）- 任意
SANTIMENT_API_KEY=your_santiment_api_key
```

#### APIキー入手先

- **FRED** (必須): [https://fred.stlouisfed.org/](https://fred.stlouisfed.org/) で無料アカウント作成
- **Glassnode** (任意): [https://glassnode.com/](https://glassnode.com/) - 無料枠でも一部指標利用可
- **Coinglass** (任意): [https://coinglass.com/](https://coinglass.com/) - 先物データ
- **Santiment** (任意): [https://app.santiment.net/](https://app.santiment.net/) - SNSメトリクス

### 3. 相関分析の実行

```bash
# KPI相関分析（過去2年間のデータ）
python scripts/crypto_kpi_analysis.py \
  --start-date 2023-11-10 \
  --end-date 2025-11-09 \
  --output-dir output

# 結果確認
ls output/
# correlation_matrix.csv
# btc_kpi_heatmap.png
# btc_kpi_network.png
# granger_results.csv
# summary_report.md
```

### 4. データ収集（実装予定）

```bash
# 取引所価格データ収集
python scripts/data_collection.py --date 2025-11-09

# 特徴量生成
python scripts/feature_engineering.py --date 2025-11-09
```

### 5. モデル訓練とバックテスト（実装予定）

```bash
# LightGBMモデル訓練
python scripts/train.py --model lightgbm --cv-folds 12

# Purged/Embargoed Walk-Forward評価
python scripts/backtest.py --model lightgbm --embargo-days 3
```

### 6. 推論実行（実装予定）

```bash
# 翌日の価格方向を予測
python scripts/predict.py --horizon 1d

# 結果確認
cat output/forecast_results.csv
```

### 7. 監視ダッシュボード起動（実装予定）

```bash
streamlit run dashboard/app.py
# http://localhost:8501 でアクセス
```

## 📈 分析結果（2023-11-10〜2025-11-09）

### 主要な相関知見

- **価格系列**: Open/High/Low/Closeは互いに0.99前後で強い多重共線性
- **マクロ指標**: NASDAQ (0.89), Gold (0.88), M2供給量 (0.93) がBTCと高相関
- **逆相関**: ドル指数（DXY）は -0.39 と負の相関
- **センチメント**: Fear & Greed Indexは単純相関0.03だが、RandomForest重要度4位（非線形寄与）

### グレンジャー因果性（p値昇順）

| 特徴量 | 最適ラグ | p値 | 解釈 |
|--------|----------|-----|------|
| BTC-USD_low | 2 | 0.010 | 価格自身の自己回帰性が強い |
| BTC-USD_open | 2 | 0.044 | 2日前の始値が有意 |
| btc_close | 1 | 0.049 | 1日ラグが最も効果的 |
| nasdaq | 1 | 0.118 | 短期ラグで遅行影響 |
| BTC-USD_volume | 6 | 0.126 | 週次ラグが有意 |

詳細は `output/summary_report.md` を参照。

## 🛠️ 技術スタック

### データ取得
- **取引所API**: Binance, Coinbase, Kraken, OKX, KuCoin, Gate.io, Bitstamp, Gemini, Crypto.com
- **マクロ**: FRED API, Yahoo Finance
- **センチメント**: Alternative.me, CryptoPanic, LunarCrush

### 機械学習
- **ベースライン**: ARIMA, VAR, Ridge/Lasso, TVP-Logistic Regression
- **主力モデル**: LightGBM（回帰・分類）
- **アンサンブル**: スタッキング（メタ学習器: Logit/Elastic Net）
- **拡張**: Temporal Fusion Transformer（条件付き導入）

### 特徴量
- ラグ特徴（価格1-7, 出来高1-7, クロスアセット1-3, マクロ1-3）
- テクニカル指標（SMA, EMA, Bollinger, RSI, MACD）
- ボラティリティ（Realized Vol, Parkinson, Garman-Klass）
- レジーム検出（HMM, Drawdown, VIX/DXY合成）
- PCA（多重共線性対策）

### 運用基盤
- **実験管理**: MLflow
- **ジョブスケジューリング**: Apache Airflow
- **監視**: Streamlit Dashboard
- **バージョニング**: DVC, Docker
- **検証**: Purged/Embargoed Walk-Forward (Lopez de Prado)

## 📋 実装ロードマップ

現在の進捗: **Phase 0完了、Phase 1着手準備中**

| Phase | 内容 | 期間 | ステータス |
|-------|------|------|------------|
| Phase 0 | 環境構築・設定管理 | 1週 | ✅ 完了 |
| Phase 1 | データ取得層実装 | 2週 | 🔄 準備中 |
| Phase 2 | 特徴量エンジニアリング | 2週 | ⏳ 未着手 |
| Phase 3 | ベースラインモデル | 1週 | ⏳ 未着手 |
| Phase 4 | LightGBM実装・最適化 | 2週 | ⏳ 未着手 |
| Phase 5 | Purged/Embargoed WF評価 | 2週 | ⏳ 未着手 |
| Phase 6 | レジーム適応アンサンブル | 1週 | ⏳ 未着手 |
| Phase 7 | 運用パイプライン構築 | 2週 | ⏳ 未着手 |
| Phase 8 | テスト・ドキュメント・デプロイ | 1週 | ⏳ 未着手 |

詳細は `prompt/plan.md` を参照。

## 📚 ドキュメント

- **[spec.md](prompt/spec.md)**: 相関分析の仕様書
- **[spec2.md](prompt/spec2.md)**: 価格変動予測の仕様書（詳細）
- **[plan.md](prompt/plan.md)**: 実装計画書（Phase 0-9の詳細）

## 🤝 コントリビューション

プルリクエスト歓迎です！大きな変更を提案する場合は、まずIssueを開いて議論してください。

### 開発プラクティス

- Pull Request必須（2名以上の承認で main へマージ）
- CI（lint, test）通過が条件
- すべての実験をMLflowに記録（再現性: seed固定、Docker化）
- コードカバレッジ80%以上を目標

## 📄 ライセンス

MIT License - 詳細は [LICENSE](LICENSE) を参照

## 🔗 関連リンク

- **GitHub Repository**: [https://github.com/takurot/crypto-argorithm](https://github.com/takurot/crypto-argorithm)
- **研究参考**:
  - [C2P2: Collective Cryptocurrency Price Prediction](https://arxiv.org/abs/1906.00564) - 集合的予測
  - [CryptoPulse: Short-term Crypto Prediction](https://arxiv.org/abs/2502.19349) - レジーム適応
  - [Coinvisor: RL-based Trading Agent](https://arxiv.org/abs/2510.17235) - 強化学習
  - [CoinCLIP: Multimodal Memecoin Evaluation](https://arxiv.org/abs/2412.07591) - マルチモーダル

## 📧 コンタクト

Issue・Pull Requestでお気軽にご連絡ください。

---

**注意事項**: このプロジェクトは教育・研究目的です。実際の投資判断に使用する場合は、自己責任で十分なバックテストと検証を行ってください。暗号資産取引にはリスクが伴います。

