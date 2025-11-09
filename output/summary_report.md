# 仮想通貨KPI相関分析サマリ

- 解析期間: 2023-11-10 〜 2025-11-09
- データ粒度: 1d

## BTC対主要KPI相関 (Pearson 上位5件)
- BTC-USD_open: 0.0723
- gold: 0.0357
- m2_supply: 0.0354
- BTC-USD_high: 0.0309
- BTC-USD_low: 0.0303

## BTC対主要KPI相関 (Spearman 上位5件)
- BTC-USD_open: 0.0790
- fear_greed_index: 0.0345
- m2_supply: 0.0315
- dxy: 0.0291
- BTC-USD_high: 0.0269

## RandomForest特徴重要度 (上位5件)
- BTC-USD_open: 0.1927
- btc_close: 0.1789
- BTC-USD_volume: 0.1723
- fear_greed_index: 0.0783
- BTC-USD_high: 0.0637

## PCA寄与率
component  explained_variance_ratio  cumulative_variance_ratio
      PC1                  0.653786                   0.653786
      PC2                  0.168758                   0.822544
      PC3                  0.069959                   0.892503
      PC4                  0.060498                   0.953001
      PC5                  0.030586                   0.983587

## グレンジャー因果性検定 (p値昇順 上位10件)
       feature  best_lag  p_value
   BTC-USD_low         2 0.009932
  BTC-USD_open         2 0.041872
     btc_close         1 0.048654
  BTC-USD_high         1 0.051556
        nasdaq         1 0.114395
BTC-USD_volume         6 0.117035
         us10y         2 0.184243
          gold         1 0.261253
     m2_supply         1 0.294439
           dxy         2 0.594305