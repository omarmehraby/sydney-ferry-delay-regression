# Temporal Features: Before vs After (Feature Sets B and C)

- **no_route_stop -> no_route_stop_temporal**, trip_grouped, XGBoost: R2 0.245 -> 0.254 (+0.009, stayed about the same); RMSE 105.0 -> 104.3; MAE 77.5 -> 77.0
- **no_route_stop -> no_route_stop_temporal**, trip_grouped, LightGBM: R2 0.245 -> 0.246 (+0.001, stayed about the same); RMSE 105.0 -> 104.9; MAE 77.6 -> 77.5
- **no_route_stop -> no_route_stop_temporal**, trip_grouped, TabPFN: R2 0.130 -> 0.150 (+0.020, improved); RMSE 112.7 -> 111.4; MAE 83.8 -> 82.4
- **no_route_stop -> no_route_stop_temporal**, route_grouped, XGBoost: R2 -0.112 -> -0.103 (+0.009, stayed about the same); RMSE 125.5 -> 124.9; MAE 93.6 -> 93.2
- **no_route_stop -> no_route_stop_temporal**, route_grouped, LightGBM: R2 -0.069 -> -0.075 (-0.006, stayed about the same); RMSE 123.0 -> 123.3; MAE 91.0 -> 91.3
- **no_route_stop -> no_route_stop_temporal**, route_grouped, TabPFN: R2 -0.098 -> -0.106 (-0.008, stayed about the same); RMSE 124.8 -> 125.2; MAE 91.9 -> 92.1
- **no_identity -> no_identity_temporal**, trip_grouped, XGBoost: R2 0.228 -> 0.240 (+0.012, improved); RMSE 106.1 -> 105.3; MAE 78.3 -> 77.6
- **no_identity -> no_identity_temporal**, trip_grouped, LightGBM: R2 0.228 -> 0.234 (+0.006, stayed about the same); RMSE 106.1 -> 105.8; MAE 78.4 -> 78.1
- **no_identity -> no_identity_temporal**, trip_grouped, TabPFN: R2 0.118 -> 0.131 (+0.013, improved); RMSE 113.5 -> 112.6; MAE 84.5 -> 83.6
- **no_identity -> no_identity_temporal**, route_grouped, XGBoost: R2 -0.136 -> -0.125 (+0.011, improved); RMSE 126.8 -> 126.8; MAE 95.0 -> 94.9
- **no_identity -> no_identity_temporal**, route_grouped, LightGBM: R2 -0.092 -> -0.089 (+0.003, stayed about the same); RMSE 124.3 -> 124.8; MAE 92.3 -> 92.5
- **no_identity -> no_identity_temporal**, route_grouped, TabPFN: R2 -0.088 -> -0.102 (-0.014, got worse); RMSE 124.2 -> 129.2; MAE 91.6 -> 99.2
