# 仮想通貨KPI相関分析サマリ

- 解析期間: 2023-11-10 〜 2025-11-09
- データ粒度: 1d

## BTC対主要KPI相関 (Pearson 上位5件)
- BTC-USD_open: 0.0721
- gold: 0.0353
- m2_supply: 0.0351
- BTC-USD_high: 0.0307
- BTC-USD_low: 0.0301

## BTC対主要KPI相関 (Spearman 上位5件)
- BTC-USD_open: 0.0785
- fear_greed_index: 0.0355
- m2_supply: 0.0306
- dxy: 0.0296
- BTC-USD_high: 0.0264

## RandomForest特徴重要度 (上位5件)
- BTC-USD_open: 0.1921
- btc_close: 0.1785
- BTC-USD_volume: 0.1713
- fear_greed_index: 0.0809
- BTC-USD_low: 0.0655

## PCA寄与率
component  explained_variance_ratio  cumulative_variance_ratio
      PC1                  0.653902                   0.653902
      PC2                  0.168669                   0.822571
      PC3                  0.069837                   0.892408
      PC4                  0.060579                   0.952987
      PC5                  0.030586                   0.983573

## グレンジャー因果性検定 (p値昇順 上位10件)
       feature  best_lag  p_value
   BTC-USD_low         2 0.010091
  BTC-USD_open         2 0.044331
     btc_close         1 0.049435
  BTC-USD_high         1 0.052414
        nasdaq         1 0.117694
BTC-USD_volume         6 0.125610
         us10y         2 0.184674
          gold         1 0.267769
     m2_supply         1 0.299630
           dxy         2 0.588816