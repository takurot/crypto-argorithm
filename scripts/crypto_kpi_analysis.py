"""
Cryptocurrency KPI correlation analysis script.

仕様書: prompt/spec.md

主な機能:
    * データ収集 (価格, マクロ, オンチェーン, デリバティブ, センチメント)
    * 前処理 (時系列揃え, 欠損補完, スケーリング, 収益率算出)
    * 相関分析 (Pearson / Spearman, ヒートマップ出力)
    * 重要度評価 (RandomForest, PCA)
    * グレンジャー因果性検定
    * 結果出力 (CSV, PNG, Markdown)

必要な外部APIキー:
    * FRED:      環境変数 FRED_API_KEY
    * Glassnode: 環境変数 GLASSNODE_API_KEY
    * Coinglass: 環境変数 COINGLASS_API_KEY
    * Santiment: 環境変数 SANTIMENT_API_KEY (任意, Twitter Volumeなど)

APIキーが未設定の場合、該当指標はスキップし警告を表示します。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import yfinance as yf

# Suppress font warnings for CJK characters
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests

logger = logging.getLogger("crypto_kpi_analysis")


DATE_FMT = "%Y-%m-%d"
DEFAULT_INTERVAL = "1d"
DEFAULT_LAG_RANGE = (1, 7)


class DataFetchError(Exception):
    """データ取得エラーを表す例外。"""


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="仮想通貨KPI相関分析スクリプト (仕様書: prompt/spec.md)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now(timezone.utc) - timedelta(days=365 * 2)).strftime(DATE_FMT),
        help="分析開始日 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now(timezone.utc).strftime(DATE_FMT),
        help="分析終了日 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--interval",
        choices={"1d", "4h"},
        default=DEFAULT_INTERVAL,
        help="データ粒度 (1d or 4h)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="成果物出力ディレクトリ",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=DEFAULT_LAG_RANGE[1],
        help="グレンジャー因果性検定の最大ラグ",
    )
    parser.add_argument(
        "--only-core",
        action="store_true",
        help="必須データ(価格, マクロ, センチメント)のみ取得し、オプション指標をスキップ",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="ログレベル (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def daterange(start: datetime, end: datetime) -> Tuple[datetime, datetime]:
    if start >= end:
        raise ValueError("start-date は end-date より前である必要があります。")
    return start, end


def safe_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def parse_timestamp(value: str) -> datetime:
    candidates = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y",
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
    ]
    for fmt in candidates:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromtimestamp(int(value))
    except (TypeError, ValueError):
        pass
    raise ValueError(f"未知の日時形式: {value}")


def to_daily_index(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty:
        return df
    if interval == "4h":
        return (
            df.resample("1D")
            .agg("mean")
            .dropna(how="all")
        )
    return df


def add_log_return(series: pd.Series, periods: int = 1, suffix: str = "log_ret") -> pd.Series:
    return np.log(series / series.shift(periods)).rename(f"{series.name}_{suffix}")


def df_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    try:
        return df.to_markdown(index=index)
    except (ImportError, ValueError):
        return df.to_string(index=index)


def ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def read_env(key: str) -> Optional[str]:
    value = os.getenv(key)
    if value:
        return value.strip()
    return None


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        logger.debug(".env file not found: %s", env_path)
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:  # noqa: BLE001
        logger.debug(".env file read failed: %s (%s)", env_path, exc)
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key in os.environ:
            continue
        os.environ[key] = value
        logger.debug("Loaded %s from %s", key, env_path)


def fetch_yfinance_series(
    ticker: str,
    start: datetime,
    end: datetime,
    interval: str,
    fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    logger.info("Yahoo Financeからデータ取得: %s (%s)", ticker, interval)
    data = yf.download(
        ticker,
        start=start,
        end=end + timedelta(days=1),
        interval="1d" if interval == "1d" else "1h",
        progress=False,
        auto_adjust=False,
    )
    if data.empty:
        raise DataFetchError(f"Yahoo Finance: {ticker} のデータを取得できませんでした。")

    if isinstance(data.columns, pd.MultiIndex):
        try:
            data.columns = data.columns.droplevel(-1)
        except Exception:  # noqa: BLE001
            data.columns = ["_".join(str(part) for part in col if part) for col in data.columns]
    else:
        data.columns = [str(col) for col in data.columns]

    if interval == "4h":
        data = data.resample("4H").last().dropna(how="all")

    if fields:
        missing = [field for field in fields if field not in data.columns]
        if missing:
            logger.warning("指定フィールドが存在しません: %s (利用可能: %s)", missing, list(data.columns))
        selected_fields = [field for field in fields if field in data.columns]
        if not selected_fields:
            raise DataFetchError(f"Yahoo Finance: {ticker} で利用可能な列がありません。")
        data = data[selected_fields]
    data.columns = [f"{ticker}_{col.lower()}" for col in data.columns]
    return data


def fetch_fred_series(series_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    api_key = read_env("FRED_API_KEY")
    if not api_key:
        raise DataFetchError("FRED_API_KEY が設定されていないため取得をスキップします。")

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start.strftime(DATE_FMT),
        "observation_end": end.strftime(DATE_FMT),
    }
    url = "https://api.stlouisfed.org/fred/series/observations"
    logger.info("FREDからデータ取得: %s", series_id)
    res = requests.get(url, params=params, timeout=30)
    res.raise_for_status()
    data = res.json()
    observations = data.get("observations", [])
    if not observations:
        raise DataFetchError(f"FRED: {series_id} の観測値が取得できませんでした。")

    df = pd.DataFrame(
        {
            "date": [obs["date"] for obs in observations],
            series_id: [safe_float(obs["value"]) for obs in observations],
        }
    ).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


def fetch_glassnode_metric(
    metric: str,
    start: datetime,
    end: datetime,
    asset: str = "BTC",
    interval: str = "24h",
) -> pd.DataFrame:
    api_key = read_env("GLASSNODE_API_KEY")
    if not api_key:
        raise DataFetchError("GLASSNODE_API_KEY が設定されていないため取得をスキップします。")

    endpoint = f"https://api.glassnode.com/v1/metrics/{metric}"
    params = {
        "a": asset,
        "s": int(start.timestamp()),
        "u": int(end.timestamp()),
        "i": interval,
        "api_key": api_key,
    }
    logger.info("Glassnodeからデータ取得: %s (%s)", metric, asset)
    res = requests.get(endpoint, params=params, timeout=30)
    res.raise_for_status()
    data = res.json()
    if not data:
        raise DataFetchError(f"Glassnode: {metric} のデータが取得できませんでした。")
    df = pd.DataFrame(data)
    df["t"] = pd.to_datetime(df["t"], unit="s")
    df = df.rename(columns={"t": "date", "v": metric.replace("/", "_")})
    df = df.set_index("date")
    return df


def fetch_fear_greed_index(start: datetime, end: datetime) -> pd.DataFrame:
    logger.info("Alternative.me Fear & Greed Index取得")
    url = "https://api.alternative.me/fng/"
    params = {
        "limit": 0,
        "format": "json",
        "date_format": "iso",
    }
    res = requests.get(url, params=params, timeout=30)
    res.raise_for_status()
    data = res.json()
    values = data.get("data", [])
    if not values:
        raise DataFetchError("Fear & Greed Indexのデータが空でした。")

    records = []
    for item in values:
        try:
            date = parse_timestamp(item["timestamp"])
        except ValueError as exc:
            logger.debug("Fear & Greed Indexの日時解析失敗: %s (%s)", item.get("timestamp"), exc)
            continue
        if start <= date <= end:
            records.append(
                {
                    "date": date,
                    "fear_greed_index": safe_float(item["value"]),
                    "classification": item.get("value_classification"),
                }
            )
    if not records:
        raise DataFetchError("指定期間に該当するFear & Greed Indexがありません。")
    df = pd.DataFrame(records).set_index("date").sort_index()
    return df[["fear_greed_index"]]


def fetch_coinglass_metric(endpoint: str, market: str, start: datetime, end: datetime) -> pd.DataFrame:
    api_key = read_env("COINGLASS_API_KEY")
    if not api_key:
        raise DataFetchError("COINGLASS_API_KEY が設定されていないため取得をスキップします。")

    base_url = "https://open-api.coinglass.com/public/v2"
    headers = {"coinglassSecret": api_key}
    params = {"symbol": market}
    url = f"{base_url}/{endpoint}"
    logger.info("Coinglassからデータ取得: %s (%s)", endpoint, market)
    res = requests.get(url, headers=headers, params=params, timeout=30)
    res.raise_for_status()
    data = res.json()
    if data.get("code") != 0:
        raise DataFetchError(f"Coinglassエラー({endpoint}): {data.get('msg')}")
    records = data.get("data", [])
    df = pd.DataFrame(records)
    if df.empty:
        raise DataFetchError(f"Coinglass: {endpoint} データが空でした。")

    time_col = "time" if "time" in df.columns else "timestamp"
    df[time_col] = pd.to_datetime(df[time_col], unit="ms", utc=True).dt.tz_localize(None)
    df = df.set_index(time_col)
    df = df[(df.index >= start) & (df.index <= end)]
    df = df.sort_index()
    df.columns = [f"{endpoint.replace('/', '_')}"]
    return df


@dataclass
class DatasetBundle:
    name: str
    df: pd.DataFrame
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_daily(self, interval: str) -> "DatasetBundle":
        daily_df = to_daily_index(self.df, interval)
        return DatasetBundle(name=self.name, df=daily_df, metadata=self.metadata)


class CryptoKPIAnalyzer:
    def __init__(
        self,
        start: datetime,
        end: datetime,
        interval: str,
        output_dir: str,
        max_lag: int,
        only_core: bool = False,
    ) -> None:
        self.start, self.end = daterange(start, end)
        self.interval = interval
        self.output_dir = ensure_output_dir(output_dir)
        self.max_lag = max_lag
        self.only_core = only_core
        self.datasets: Dict[str, DatasetBundle] = {}

    # --------------------
    # データ取得
    # --------------------
    def fetch_all(self) -> None:
        self._fetch_price_data()
        self._fetch_macro_data()
        self._fetch_sentiment_data()
        if not self.only_core:
            self._fetch_onchain_data()
            self._fetch_derivative_data()

    def _store(self, name: str, df: pd.DataFrame, metadata: Optional[Dict[str, str]] = None) -> None:
        if df is None or df.empty:
            logger.warning("%s の取得結果が空のためスキップします。", name)
            return
        df = df.sort_index()
        self.datasets[name] = DatasetBundle(name=name, df=df, metadata=metadata or {})
        logger.info("%s: 取得完了 (レコード数=%d)", name, len(df))

    def _fetch_price_data(self) -> None:
        btc_df = fetch_yfinance_series("BTC-USD", self.start, self.end, self.interval, ["Open", "High", "Low", "Close", "Volume"])
        btc_df = btc_df.rename(columns={"BTC-USD_close": "btc_close"})
        btc_df["btc_log_return"] = add_log_return(btc_df["btc_close"], suffix="log_return")
        self._store("price", btc_df)

    def _fetch_macro_data(self) -> None:
        frames = []
        macro_def = {
            "us10y": {"ticker": "^TNX", "field": "Close"},
            "nasdaq": {"ticker": "^IXIC", "field": "Close"},
            "dxy": {"ticker": "DX-Y.NYB", "field": "Close"},
            "gold": {"ticker": "GC=F", "field": "Close"},
        }
        for name, cfg in macro_def.items():
            try:
                df = fetch_yfinance_series(cfg["ticker"], self.start, self.end, self.interval, [cfg["field"]])
                df = df.rename(columns={f"{cfg['ticker']}_{cfg['field'].lower()}": name})
                frames.append(df)
            except Exception as exc:  # noqa: BLE001
                logger.warning("マクロ指標 %s 取得失敗: %s", name, exc)
        try:
            fred_df = fetch_fred_series("M2SL", self.start, self.end)
            fred_df = fred_df.rename(columns={"M2SL": "m2_supply"})
            frames.append(fred_df)
        except Exception as exc:  # noqa: BLE001
            logger.warning("M2供給量取得失敗: %s", exc)

        if frames:
            macro_df = pd.concat(frames, axis=1)
            self._store("macro", macro_df)
        else:
            logger.warning("マクロ指標が1件も取得できませんでした。")

    def _fetch_onchain_data(self) -> None:
        metrics = {
            "supply/exchange_balance": {"asset": "BTC"},
            "flow/miner_to_exchange": {"asset": "BTC"},
            "supply/stablecoins": {"asset": "USDT"},
            "market/mvrv_z_score": {"asset": "BTC"},
            "market/realized_cap_usd": {"asset": "BTC"},
        }
        frames = []
        for metric, cfg in metrics.items():
            try:
                df = fetch_glassnode_metric(
                    metric,
                    self.start,
                    self.end,
                    asset=cfg.get("asset", "BTC"),
                    interval="24h" if self.interval == "1d" else "4h",
                )
                frames.append(df)
            except Exception as exc:  # noqa: BLE001
                logger.warning("オンチェーン指標 %s 取得失敗: %s", metric, exc)
        if frames:
            onchain_df = pd.concat(frames, axis=1)
            self._store("onchain", onchain_df)

    def _fetch_derivative_data(self) -> None:
        frames = []
        endpoints = {
            "futures/funding_rate": "BTC",
            "futures/open_interest/chart": "BTC",
        }
        for endpoint, market in endpoints.items():
            try:
                df = fetch_coinglass_metric(endpoint, market, self.start, self.end)
                frames.append(df)
            except Exception as exc:  # noqa: BLE001
                logger.warning("デリバティブ指標 %s 取得失敗: %s", endpoint, exc)
        if frames:
            derivative_df = pd.concat(frames, axis=1)
            self._store("derivative", derivative_df)

    def _fetch_sentiment_data(self) -> None:
        try:
            fear_greed_df = fetch_fear_greed_index(self.start, self.end)
            self._store("sentiment", fear_greed_df)
        except Exception as exc:  # noqa: BLE001
            logger.warning("センチメント指標取得失敗: %s", exc)

    # --------------------
    # 前処理
    # --------------------
    def build_feature_table(self) -> pd.DataFrame:
        if not self.datasets:
            raise RuntimeError("データセットが1件も存在しません。fetch_all()を実行してください。")

        aligned = []
        for bundle in self.datasets.values():
            daily_bundle = bundle.to_daily(self.interval)
            aligned.append(daily_bundle.df)

        df = pd.concat(aligned, axis=1)
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        df = df.loc[(df.index >= self.start) & (df.index <= self.end)]

        # 欠損補完: 線形補完 + 前方埋め
        df = df.interpolate(method="time").ffill().bfill()

        # ログリターン (BTC基準)
        if "btc_close" in df.columns and "btc_log_return" not in df.columns:
            df["btc_log_return"] = add_log_return(df["btc_close"])

        df = df.dropna(axis=0, how="any")
        logger.info("特徴量テーブル生成完了: shape=%s", df.shape)
        return df

    def scale_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        scaler = StandardScaler()
        scaled = pd.DataFrame(
            scaler.fit_transform(df),
            index=df.index,
            columns=df.columns,
        )
        return scaled, scaler

    # --------------------
    # 分析
    # --------------------
    def run_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        pearson = df.corr(method="pearson")
        spearman = df.corr(method="spearman")
        logger.info("相関行列計算完了")
        return {"pearson": pearson, "spearman": spearman}

    def plot_heatmap(self, corr: pd.DataFrame, title: str, path: str) -> None:
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, linewidths=0.5)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info("ヒートマップ保存: %s", path)

    def evaluate_feature_importance(
        self, df: pd.DataFrame, target_col: str = "btc_log_return"
    ) -> Optional[pd.Series]:
        if target_col not in df.columns:
            logger.warning("特徴重要度評価をスキップ: %s が存在しません。", target_col)
            return None

        features = df.drop(columns=[target_col])
        target = df[target_col]
        if len(features.columns) < 2:
            logger.warning("特徴量が不足しているため重要度評価をスキップします。")
            return None

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(features, target)
        importances = pd.Series(model.feature_importances_, index=features.columns)
        importances = importances.sort_values(ascending=False)
        logger.info("特徴重要度推定完了")
        return importances

    def run_pca(self, df: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
        pca = PCA(n_components=min(n_components, df.shape[1]))
        pca.fit(df)
        explained = pd.DataFrame(
            {
                "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
            }
        )
        logger.info("PCA計算完了")
        return explained

    def run_granger_causality(
        self,
        df: pd.DataFrame,
        target_col: str = "btc_log_return",
        max_lag: int = 7,
    ) -> pd.DataFrame:
        if target_col not in df.columns:
            raise RuntimeError(f"グレンジャー因果性検定対象列 {target_col} が存在しません。")

        results = []
        for col in df.columns:
            if col == target_col:
                continue
            try:
                test_df = df[[target_col, col]].dropna()
                if len(test_df) <= max_lag + 1:
                    logger.debug("Granger検定スキップ (%s): データ点不足", col)
                    continue
                granger = grangercausalitytests(test_df, maxlag=max_lag, verbose=False)
                best_lag = min(granger.keys(), key=lambda lag: granger[lag][0]["ssr_ftest"][1])
                pvalue = granger[best_lag][0]["ssr_ftest"][1]
                results.append(
                    {
                        "feature": col,
                        "best_lag": best_lag,
                        "p_value": pvalue,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Granger検定失敗 (%s): %s", col, exc)
        if not results:
            raise RuntimeError("グレンジャー因果性検定結果が得られませんでした。")
        result_df = pd.DataFrame(results).sort_values("p_value")
        logger.info("グレンジャー因果性検定完了")
        return result_df

    # --------------------
    # 出力
    # --------------------
    def export_correlation_matrix(self, corr: pd.DataFrame, path: str) -> None:
        corr.to_csv(path)
        logger.info("相関行列CSV出力: %s", path)

    def export_granger_results(self, df: pd.DataFrame, path: str) -> None:
        df.to_csv(path, index=False)
        logger.info("グレンジャー検定結果出力: %s", path)

    def export_summary_report(
        self,
        corr_dict: Dict[str, pd.DataFrame],
        feature_importance: Optional[pd.Series],
        pca_df: pd.DataFrame,
        granger_df: pd.DataFrame,
        path: str,
        target_col: str = "btc_log_return",
        top_n: int = 5,
    ) -> None:
        pearson = corr_dict["pearson"][target_col].drop(target_col).abs().sort_values(ascending=False).head(top_n)
        spearman = corr_dict["spearman"][target_col].drop(target_col).abs().sort_values(ascending=False).head(top_n)

        lines = [
            "# 仮想通貨KPI相関分析サマリ",
            "",
            f"- 解析期間: {self.start.strftime(DATE_FMT)} 〜 {self.end.strftime(DATE_FMT)}",
            f"- データ粒度: {self.interval}",
            "",
            "## BTC対主要KPI相関 (Pearson 上位5件)",
        ]
        for name, value in pearson.items():
            lines.append(f"- {name}: {value:.4f}")

        lines.append("")
        lines.append("## BTC対主要KPI相関 (Spearman 上位5件)")
        for name, value in spearman.items():
            lines.append(f"- {name}: {value:.4f}")

        if feature_importance is not None:
            lines.extend(["", "## RandomForest特徴重要度 (上位5件)"])
            for name, value in feature_importance.head(top_n).items():
                lines.append(f"- {name}: {value:.4f}")

        lines.extend(
            [
                "",
                "## PCA寄与率",
                df_to_markdown(pca_df, index=False),
                "",
                "## グレンジャー因果性検定 (p値昇順 上位10件)",
                df_to_markdown(granger_df.head(10), index=False),
            ]
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("サマリレポート出力: %s", path)

    def export_correlation_network(
        self,
        corr: pd.DataFrame,
        path: str,
        threshold: float = 0.6,
    ) -> None:
        try:
            import networkx as nx
        except ImportError:
            logger.warning("NetworkXが未インストールのためKPIネットワーク可視化をスキップします。")
            return

        nodes = corr.columns.tolist()
        graph = nx.Graph()
        for node in nodes:
            graph.add_node(node)

        for i, col_i in enumerate(nodes):
            for j, col_j in enumerate(nodes):
                if j <= i:
                    continue
                weight = corr.loc[col_i, col_j]
                if not np.isnan(weight) and abs(weight) >= threshold:
                    graph.add_edge(col_i, col_j, weight=float(weight))

        if graph.number_of_edges() == 0:
            logger.warning("閾値 %.2f を超える相関エッジが存在しません。ネットワーク描画をスキップします。", threshold)
            return

        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(graph, seed=42)
        weights = [abs(graph[u][v]["weight"]) for u, v in graph.edges()]
        nx.draw_networkx_nodes(graph, pos, node_color="skyblue", node_size=800)
        nx.draw_networkx_edges(graph, pos, width=[3 * w for w in weights], edge_color=weights, edge_cmap=plt.cm.coolwarm)
        nx.draw_networkx_labels(graph, pos, font_size=9)
        plt.title(f"KPI相関ネットワーク (|r| >= {threshold:.2f})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info("KPIネットワーク図保存: %s", path)

    # --------------------
    # 実行フロー
    # --------------------
    def run(self) -> None:
        self.fetch_all()
        feature_df = self.build_feature_table()
        features_only = feature_df.drop(columns=["btc_log_return"], errors="ignore")
        if features_only.empty:
            raise RuntimeError("BTCログリターン以外の特徴量がありません。")

        scaled_df, _ = self.scale_features(features_only)
        scaled_with_target = scaled_df.join(feature_df[["btc_log_return"]], how="left")
        scaled_with_target = scaled_with_target.dropna()
        if scaled_with_target.empty:
            raise RuntimeError("前処理後のデータが空となりました。")

        corr_dict = self.run_correlation_analysis(scaled_with_target)
        pearson_corr = corr_dict["pearson"]

        corr_path = os.path.join(self.output_dir, "correlation_matrix.csv")
        self.export_correlation_matrix(pearson_corr, corr_path)

        heatmap_path = os.path.join(self.output_dir, "btc_kpi_heatmap.png")
        self.plot_heatmap(pearson_corr, "BTC KPI Correlation (Pearson)", heatmap_path)

        network_path = os.path.join(self.output_dir, "btc_kpi_network.png")
        self.export_correlation_network(pearson_corr, network_path)

        feature_importance = self.evaluate_feature_importance(scaled_with_target)

        pca_df = self.run_pca(scaled_df)

        granger_df = self.run_granger_causality(scaled_with_target.dropna(), max_lag=self.max_lag)
        granger_path = os.path.join(self.output_dir, "granger_results.csv")
        self.export_granger_results(granger_df, granger_path)

        summary_path = os.path.join(self.output_dir, "summary_report.md")
        self.export_summary_report(
            corr_dict=corr_dict,
            feature_importance=feature_importance,
            pca_df=pca_df,
            granger_df=granger_df,
            path=summary_path,
        )


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)
    load_env_file()
    try:
        analyzer = CryptoKPIAnalyzer(
            start=datetime.strptime(args.start_date, DATE_FMT),
            end=datetime.strptime(args.end_date, DATE_FMT),
            interval=args.interval,
            output_dir=args.output_dir,
            max_lag=args.max_lag,
            only_core=args.only_core,
        )
        analyzer.run()
    except Exception as exc:  # noqa: BLE001
        logger.exception("分析処理でエラーが発生しました: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

