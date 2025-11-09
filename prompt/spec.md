# 仮想通貨KPI相関分析 仕様書

## 目的

仮想通貨（例：BTC, ETHなど）の価格変動要因を定量的に分析し、主要KPI（マクロ、オンチェーン、デリバティブ、センチメント等）との相関構造を明らかにする。これにより、価格予測モデル（例：GPR, LSTM, LightGBM等）の特徴量選定を支援する。

---

## 分析対象

* 対象通貨：BTC-USD（必要に応じてETH, SOL等も拡張）
* 対象期間：過去2年（例：2023-01-01 〜 現在）
* データ粒度：日次（1D）または4時間足（4H）

---

## 収集データ項目

### 1. 価格関連（Yahoo Finance / Binance API）

* Close, Open, High, Low, Volume
* 日次リターン（log return）

### 2. マクロ経済指標（FRED / Yahoo Finance）

* 米10年国債利回り（US10Y）
* NASDAQ指数（^IXIC）
* ドル指数（DXY）
* 金価格（GC=F）
* M2マネーサプライ

### 3. オンチェーンデータ（Glassnode / CryptoQuant）

* Exchange Balance (BTC残高)
* Miner to Exchange Flow
* Stablecoin Supply (USDT, USDC)
* MVRV比率
* Realized Cap

### 4. デリバティブデータ（Coinglass / Binance Futures API）

* Funding Rate
* Open Interest
* Liquidation Volume（Long/Short別）

### 5. センチメントデータ（Alternative.me / Santiment）

* Fear & Greed Index
* Twitter Volume（BTC関連投稿数）
* Positive/Negative Ratio（SNS感情分析）

---

## 分析手法

### 1. 前処理

* 各特徴量のスケーリング（StandardScaler or MinMaxScaler）
* 欠損値補完（線形補間または直近値維持）
* 時系列整合（共通のタイムスタンプでリサンプリング）

### 2. 相関分析

* ピアソン相関係数
* スピアマン順位相関（非線形関係の補完）
* ヒートマップ可視化（Seaborn Heatmap）

```python
sns.heatmap(df.corr(method='pearson'), annot=True, cmap='coolwarm')
```

### 3. 重要度評価（任意）

* RandomForest / LightGBMによる特徴重要度
* PCAによる潜在因子抽出（主成分寄与率）

### 4. 因果性検証

* グレンジャー因果性検定（Granger Causality Test）
* ラグ時間の最適化（1〜7日）

### 5. 結果出力

* 相関マトリクス（CSV形式）
* KPI間ネットワーク可視化（NetworkX）
* 要約レポート（各KPIとBTC変化率の相関上位5項目）

---

## 成果物

| ファイル名                    | 内容            |
| ------------------------ | ------------- |
| `correlation_matrix.csv` | KPI間相関係数表     |
| `btc_kpi_heatmap.png`    | 相関ヒートマップ画像    |
| `granger_results.csv`    | 因果関係検定結果      |
| `summary_report.md`      | 相関分析サマリ（テキスト） |

---

## 分析環境

* 言語：Python 3.11
* ライブラリ：pandas, numpy, seaborn, matplotlib, sklearn, statsmodels, yfinance
* 実行環境：Jupyter / Google Colab / VSCode + Jupyter

---

## 今後の展開

* KPI間の相関構造を踏まえた特徴量選択
* GPR（ガウス過程回帰）およびLSTMモデルへの入力最適化
* 相関変化の時系列モニタリング（ローリングウィンドウ分析）
* Streamlitダッシュボード化による可視化・運用化
