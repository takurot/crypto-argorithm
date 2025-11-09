# 仮想通貨価格変動予測 実装計画書（plan.md）

本計画書は `spec2.md` の仕様を実装するための段階的なロードマップを定義します。

---

## 実装の全体構成

```
crypt-argorithm/
├── data/                    # データ取得・保存層
│   ├── raw/                 # 生データキャッシュ（取引所・API別）
│   ├── processed/           # 前処理済み（VWAP集約・正規化）
│   └── features/            # 特徴量エンジニアリング済み
├── models/                  # 学習済みモデル保存
│   ├── baseline/
│   ├── lightgbm/
│   └── ensemble/
├── notebooks/               # 探索的分析・実験ノート
├── scripts/                 # 実行スクリプト
│   ├── data_collection.py   # データ収集
│   ├── feature_engineering.py  # 特徴量生成
│   ├── train.py             # モデル訓練
│   ├── backtest.py          # Purged/Embargoed WF評価
│   └── predict.py           # 推論・デプロイ用
├── src/                     # 再利用可能モジュール
│   ├── data/
│   │   ├── collectors/      # 取引所・API別コレクター
│   │   ├── aggregators.py   # VWAP中央値・外れ値処理
│   │   └── validators.py    # データ品質チェック
│   ├── features/
│   │   ├── lags.py
│   │   ├── technical.py
│   │   ├── regime.py        # HMM/Drawdown/合成レジーム
│   │   └── cross_asset.py
│   ├── models/
│   │   ├── baseline.py      # Ridge/Lasso/TVP-Logit
│   │   ├── lgbm.py
│   │   ├── ensemble.py      # スタッキング・重み調整
│   │   └── calibration.py   # Platt/Isotonic
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── walk_forward.py  # Purged/Embargoed WF
│   │   └── trading_sim.py
│   └── utils/
│       ├── config.py
│       └── logging.py
├── tests/                   # ユニット・統合テスト
├── config/                  # 設定ファイル（YAML）
│   ├── data_sources.yaml
│   ├── features.yaml
│   └── models.yaml
├── dashboard/               # Streamlit監視ダッシュボード
├── mlruns/                  # MLflow実験管理
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Phase 0: 環境構築と基盤整備（1週間）

### 0.1 プロジェクト初期化
- [ ] リポジトリ構造作成（上記ディレクトリツリー）
- [ ] `.gitignore` 設定（`data/`, `models/`, `mlruns/`, `.env`）
- [ ] `requirements.txt` 作成（主要依存）
  ```
  pandas>=2.0
  numpy>=1.24
  scikit-learn>=1.3
  lightgbm>=4.0
  optuna>=3.0
  mlflow>=2.8
  streamlit>=1.28
  yfinance
  requests
  python-dotenv
  statsmodels
  hmmlearn
  ta-lib  # テクニカル指標
  ```
- [ ] Docker化（`Dockerfile` + `docker-compose.yml`）
- [ ] 開発環境セットアップ（venv/conda + pre-commit hooks）

### 0.2 設定管理
- [ ] `config/data_sources.yaml` 作成
  - 取引所エンドポイント、APIキー環境変数マッピング
  - 最小取引所数=7、外れ値閾値=3%、embargo=3日
- [ ] `config/features.yaml` 作成
  - ラグ範囲、テクニカル指標パラメータ、レジーム閾値
- [ ] `config/models.yaml` 作成
  - LightGBM推奨パラメータ、Optuna探索空間、WF設定
- [ ] `.env.example` 作成（APIキー例示）

### 0.3 ロギング・監視基盤
- [ ] `src/utils/logging.py` 実装（構造化ログ、MLflow連携）
- [ ] MLflow初期化（`mlruns/` ディレクトリ、実験名定義）
- [ ] データ品質監視の仕組み（PSI計算、欠損率、外れ値カウント）

---

## Phase 1: データ取得層の実装（2週間）

### 1.1 取引所データコレクター（BTC/ETH/SOL/BNB）
- [ ] `src/data/collectors/exchange_collector.py` 実装
  - `ref/crypt-arbitrage/crypt-arbitrage.py` をベースに拡張
  - 10社以上の取引所（Binance, Coinbase, Kraken, OKX等）
  - VWAP取得（価格×出来高加重）、USDT/USD正規化ロジック
  - レート制限・指数バックオフ・リトライ機構
  - 生データは `data/raw/exchanges/{date}/{exchange}_{symbol}.json` に保存
- [ ] 動的外れ値除外（中央値±3%）とVWAP中央値集約
- [ ] 最小7社チェック、2連続欠損でバックフィル打ち切り

### 1.2 マクロ経済データコレクター
- [ ] `src/data/collectors/macro_collector.py` 実装
  - FRED API（M2, 10Y, VIX）
  - Yahoo Finance（NASDAQ, DXY, Gold）
  - 発表UTC時刻管理、サプライズz-score計算（実績−予想の標準化）
  - 新値反映フラグ生成（M2等の低頻度系列）
- [ ] 前方埋め期限（日次7日、月次30日）実装
- [ ] 生データは `data/raw/macro/{date}/{indicator}.json` に保存

### 1.3 センチメント・ニュースコレクター
- [ ] `src/data/collectors/sentiment_collector.py` 実装
  - Alternative.me Fear & Greed Index
  - CryptoPanic API（ニュースセンチメント）
  - LunarCrush API（SNS活動指標）
- [ ] ソース多重度、重複除外、言語別バイアス補正
- [ ] 7日/14日ローリング平均の事前計算

### 1.4 オンチェーン・デリバティブコレクター（拡張）
- [ ] `src/data/collectors/onchain_collector.py`（CoinMetrics, CryptoQuant）
- [ ] `src/data/collectors/derivatives_collector.py`（Binance Futures, Bybit）
- [ ] キャッシュ・再取得ロジック（日次1回、APIクォータ管理）

### 1.5 データバリデーション
- [ ] `src/data/validators.py` 実装
  - 最小取引所数チェック
  - タイムスタンプ連続性検証
  - 外れ値・欠損率モニタリング
  - PSI計算（新旧データ分布比較）
- [ ] バリデーションレポート自動生成（Markdown + MLflow）

### 1.6 統合データパイプライン
- [ ] `scripts/data_collection.py` 実装
  - 全コレクターを並列実行（ThreadPoolExecutor）
  - エラーハンドリングと部分成功時の継続
  - `data/processed/{date}/consolidated.parquet` に集約
- [ ] Airflow DAG作成（毎日UTC 00:15実行）

---

## Phase 2: 特徴量エンジニアリング（2週間）

### 2.1 基本特徴量生成
- [ ] `src/features/lags.py` 実装
  - 価格lag 1〜7、出来高lag 1〜7
  - クロスアセット（ETH/SOL/BNB）lag 1〜3
  - マクロlag 1〜3、センチメントlag 1〜5
  - 変化率（Δ1, Δ3）、対数差分
- [ ] `src/features/technical.py` 実装
  - SMA7/21, EMA9/26, Bollinger Bands
  - RSI(14), MACD, ATR
  - 日中レンジ比、価格vs SMAの乖離率

### 2.2 ボラティリティ・レジーム特徴
- [ ] `src/features/regime.py` 実装
  - 10日/30日ローリング標準偏差
  - Garman-Klass, Parkinson, Realized Vol
  - HMM（2-state: 低ボラ/高ボラ）
  - Drawdownレジーム（過去最高値からの下落率）
  - SPX/VIX・DXY合成レジームID
  - 過去90日75パーセンタイル超フラグ
- [ ] レジームIDをメタ特徴として保存

### 2.3 PCAと多重共線性対策
- [ ] `src/features/pca.py` 実装
  - 価格系列PCA（累積分散90%）
  - マクロ系列PCA（累積分散90%）
  - 線形モデル用PC特徴、木系モデル用原変数の並行保存

### 2.4 ターゲットエンコーディング
- [ ] `src/features/target_encoding.py` 実装
  - `y_dir` 7日ローリング勝率（lag 8以上でシフト: t-15〜t-8）
  - `btc_log_return` 14日ローリング平均/標準偏差（t-14〜t-1）

### 2.5 メタ情報特徴
- [ ] `src/features/meta.py` 実装
  - 曜日ダミー、月初/月末フラグ
  - 経済イベントフラグ（FOMC, CPI, 雇用統計）
  - イベント前後3日間フラグ
  - 週末フラグ

### 2.6 特徴量パイプライン統合
- [ ] `scripts/feature_engineering.py` 実装
  - `data/processed/{date}/consolidated.parquet` を読込
  - 全特徴量生成器を順次実行
  - `data/features/{date}/features.parquet` に保存
  - 特徴量メタデータ（列名、型、欠損率）をMLflowにログ

---

## Phase 3: ベースラインモデル実装（1週間）

### 3.1 時系列モデル
- [ ] `src/models/baseline.py` 実装
  - ARIMA(2,0,2) for `y_1d`
  - VAR for 価格＋主要マクロ
  - 訓練・推論インターフェース統一

### 3.2 線形モデル
- [ ] Ridge/Lasso for `y_1d` 回帰
- [ ] TVP-ロジスティック回帰 for `y_dir` 分類
- [ ] 確率校正（Platt Scaling, Isotonic Regression）
  - `src/models/calibration.py` 実装

### 3.3 ベースライン評価
- [ ] 簡易WF評価（2023-11〜2024-10 train, 2024-11〜2025-11 test）
- [ ] 指標: RMSE, Directional Accuracy, Sharpe
- [ ] 結果を `models/baseline/` に保存、MLflowに記録

---

## Phase 4: LightGBM実装と最適化（2週間）

### 4.1 LightGBMモデル実装
- [ ] `src/models/lgbm.py` 実装
  - 回帰（`y_1d`, `y_3d_avg`）
  - 分類（`y_dir`、binary objective）
  - 推奨パラメータ適用（min_data_in_leaf≥20, max_depth≤6, 正則化）
  - SHAP値計算・保存

### 4.2 Optunaハイパーパラメータ探索
- [ ] `scripts/tune_lgbm.py` 実装
  - 探索空間: learning_rate, num_leaves, feature_fraction, lambda_l1/l2
  - 目的関数: valのDirectional Accuracy（方向性）、RMSE（回帰）
  - 探索回数: 100試行、TPESampler
  - 最良パラメータをMLflowに記録

### 4.3 特徴選択
- [ ] SHAP値による特徴重要度分析
- [ ] 上位30%特徴のみ使用（過学習対策）
- [ ] 選択特徴リストを `config/features.yaml` に保存

### 4.4 確率校正と閾値最適化
- [ ] `src/models/calibration.py` 拡張
  - Platt Scaling/Isotonic Regressionで`y_dir`の確率校正
  - 期待値最大化（手数料0.1%考慮）の最適閾値探索
  - valで閾値決定、testで検証

---

## Phase 5: Purged/Embargoed Walk-Forward評価（2週間）

### 5.1 WF評価フレームワーク
- [ ] `src/evaluation/walk_forward.py` 実装
  - Purged K-Fold（Lopez de Prado）
  - Embargo=3日（隣接期間の情報漏洩防止）
  - 月次分割（12ウィンドウ以上）
  - モデル選択はvalのみ、testは最後のホールドアウト1回

### 5.2 評価指標計算
- [ ] `src/evaluation/metrics.py` 実装
  - 回帰: RMSE, MAE, MAPE, R²
  - 方向性: Accuracy, Precision, Recall, MCC
  - トレード: Sharpe, MDD, Hit Ratio, PnL
  - 90日ローリング信頼区間（bootstrap）

### 5.3 統計検定
- [ ] Diebold-Mariano検定実装（ベースライン vs LightGBM）
- [ ] 有意水準5%で優位性確認

### 5.4 バックテストスクリプト
- [ ] `scripts/backtest.py` 実装
  - 全ウィンドウでLightGBM訓練・推論
  - トレードシミュレーション（手数料0.1%、スリッページ考慮）
  - エクイティカーブ・ドローダウン曲線を可視化
  - 結果を `models/lightgbm/backtest_results.csv` に保存

---

## Phase 6: レジーム適応アンサンブル（1週間）

### 6.1 レジーム別重み調整
- [ ] `src/models/ensemble.py` 実装
  - 高ボラ期: センチメント・ボラ指標の重み増加
  - 低ボラ期: マクロ指標の重み増加
  - 重みは検証スコアに基づき逆分散重みで算出

### 6.2 メタ学習（スタッキング）
- [ ] ベースライン、LightGBMの予測をメタ特徴化
- [ ] レジームIDもメタ特徴に注入
- [ ] メタ学習器: Logit/Elastic Net
- [ ] 5-fold CVでメタ学習器を訓練

### 6.3 アンサンブル評価
- [ ] WF評価でアンサンブル性能測定
- [ ] 単体モデルとの比較（Directional Accuracy, Sharpe改善度）

---

## Phase 7: 運用パイプライン構築（2週間）

### 7.1 推論スクリプト
- [ ] `scripts/predict.py` 実装
  - 最新データ読込（`data/features/latest/features.parquet`）
  - 訓練済みモデルロード（`models/lightgbm/best_model.pkl`）
  - 1日先/3日先/7日先の予測生成
  - 結果を `output/forecast_results.csv` に保存、MLflowにログ

### 7.2 自動再学習
- [ ] `scripts/train.py` 実装
  - 週次（月曜）: 最新データで再学習
  - 月次: Optuna探索で最適化
  - ベリフィケーション: RMSE < 0.025, Directional Accuracy > 0.55
  - 閾値未達時はロールバック

### 7.3 モデルバージョニング
- [ ] MLflow Model Registry統合
  - モデルを `Production` / `Staging` / `Archived` で管理
  - 各バージョンに実験ID・ハイパーパラメータ・評価指標を紐付け
- [ ] DVCで大容量データ・モデルをバージョン管理

### 7.4 監視ダッシュボード
- [ ] `dashboard/app.py`（Streamlit）実装
  - 最新予測値・実績値の時系列表示
  - 評価指標（RMSE, Directional Accuracy, Sharpe, MDD）
  - データ取得成功率、欠損率、PSIドリフト
  - モデル更新履歴、ロールバック履歴
  - レート制限状況、キャッシュヒット率

### 7.5 Airflow DAG統合
- [ ] データ収集→特徴量生成→推論の自動化DAG
- [ ] 週次再学習DAG、月次最適化DAG
- [ ] エラー通知（Slack/Email）

---

## Phase 8: テスト・ドキュメント・デプロイ（1週間）

### 8.1 ユニットテスト
- [ ] `tests/` ディレクトリ作成
- [ ] データコレクター、特徴量生成、モデルの単体テスト
- [ ] カバレッジ80%以上を目標

### 8.2 統合テスト
- [ ] データ収集→特徴量→訓練→推論のエンドツーエンドテスト
- [ ] モックAPIレスポンスで再現性確認

### 8.3 ドキュメント作成
- [ ] `README.md` 更新（セットアップ手順、実行例）
- [ ] `docs/` ディレクトリ作成
  - API仕様
  - 特徴量定義書
  - モデル選定根拠
  - 運用マニュアル

### 8.4 本番デプロイ
- [ ] Docker Compose for 本番環境（API + 監視 + MLflow）
- [ ] 秘密管理（AWS Secrets Manager / GCP Secret Manager）
- [ ] CI/CD（GitHub Actions: lint, test, build, deploy）
- [ ] ロールバック手順確立

---

## Phase 9: 拡張機能（今後の展開）

### 9.1 TFT導入（条件付き）
- [ ] LightGBMでDirectional Accuracy > 0.57達成後
- [ ] Temporal Fusion Transformer実装
- [ ] 56日入力ウィンドウ、1/3/7日出力ウィンドウ
- [ ] 動的要因（Fear & Greed, マクロ）組込み

### 9.2 強化学習エージェント
- [ ] DQN/PPO実装
- [ ] リスク調整後リターン（Sharpe）最大化の報酬設計
- [ ] バックテスト環境（gym wrapper）構築

### 9.3 クロスアセット予測
- [ ] ETH/SOL/BNB同時予測の集合的分類モデル
- [ ] グラフニューラルネットワーク（GNN）
- [ ] 通貨間相互依存性の明示的モデル化

---

## マイルストーン・スケジュール

| Phase | 期間 | 成果物 | 担当 |
|-------|------|--------|------|
| Phase 0 | Week 1 | 環境構築、設定管理 | DevOps |
| Phase 1 | Week 2-3 | データ収集層、バリデーション | Data Eng |
| Phase 2 | Week 4-5 | 特徴量エンジニアリング | Feature Eng |
| Phase 3 | Week 6 | ベースラインモデル | ML Eng |
| Phase 4 | Week 7-8 | LightGBM実装・最適化 | ML Eng |
| Phase 5 | Week 9-10 | Purged/Embargoed WF評価 | ML Eng |
| Phase 6 | Week 11 | レジーム適応アンサンブル | ML Eng |
| Phase 7 | Week 12-13 | 運用パイプライン構築 | MLOps |
| Phase 8 | Week 14 | テスト・ドキュメント・デプロイ | QA/DevOps |
| Phase 9 | 継続的 | 拡張機能（TFT, 強化学習, クロスアセット） | Research |

**総期間**: 約14週間（3.5ヶ月）でMVP完成

---

## リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| APIレート制限超過 | データ欠損 | 指数バックオフ、複数取引所の冗長化、キャッシュ活用 |
| モデル精度未達（DA < 0.55） | 実用性低下 | ベースライン複数手法、特徴量追加、レジーム適応強化 |
| 過学習（test精度がval比-10%以上低下） | 汎化性能低下 | Purged/Embargoed WF、正則化強化、特徴選択 |
| データドリフト（PSI > 0.25） | 予測劣化 | 週次再学習、ドリフト検知アラート、自動ロールバック |
| 外部ショック（規制発表等） | 予測精度急落 | イベントフラグ導入、レジーム適応、保守的閾値 |

---

## 成功指標（KPI）

### MVP完成時（Phase 8終了）
- [ ] Directional Accuracy（1D）> 0.57（90日ローリング平均）
- [ ] RMSE < 0.025（過去最良モデル比+10%以内）
- [ ] 年率Sharpe > 0.7（手数料0.1%込み）
- [ ] 最大ドローダウン < 15%
- [ ] データ取得成功率 > 95%（1週間平均）
- [ ] モデル再学習の自動化（週次実行、ロールバック機能完備）

### 本番運用3ヶ月後
- [ ] Directional Accuracy維持（> 0.55、信頼区間内）
- [ ] PSI < 0.25（データドリフト検知基準内）
- [ ] ダッシュボード稼働率 > 99%
- [ ] インシデント対応時間 < 4時間（中央値）

---

## 補足: 開発プラクティス

### コードレビュー
- Pull Request必須
- 2名以上の承認で main へマージ
- CI（lint, test）通過が条件

### 実験管理
- すべての実験をMLflowに記録（パラメータ、指標、アーティファクト）
- タグ付け（`baseline`, `lightgbm`, `ensemble`, `production`）
- 再現性: seed固定、Docker化、requirements.txt固定

### ドキュメント
- Markdown形式で `docs/` に集約
- API仕様はOpenAPI/Swagger
- モデルカードをMLflowに添付

### セキュリティ
- APIキー・秘密情報は `.env` (開発)、秘密管理基盤（本番）
- `.gitignore` で漏洩防止
- 定期的な依存ライブラリ脆弱性スキャン（Snyk/Dependabot）

---

## 次のアクション

1. **Phase 0の着手**: リポジトリ作成、`requirements.txt` 整備、Docker環境構築
2. **データ収集のプロトタイプ**: Binance/Coinbase/Krakenの3社でVWAP取得を試行
3. **ベースライン実装**: Ridge回帰でシンプルな1日先予測を実装し、精度ベンチマーク確立

このプランに沿って段階的に実装を進めることで、`spec2.md` の仕様を満たす堅牢な予測システムを構築できます。

