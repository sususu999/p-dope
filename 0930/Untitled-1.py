# 確認: predict_df が既にある想定。なければ再作成してください
import numpy as np
import pandas as pd

# predict_df がなければ作成
if 'predict_df' not in globals():
    import itertools
    halfgold_vals = list(range(90, -1, -10))
    doping_vals = np.round(np.arange(0.5, 16.0 + 1e-9, 1.0), 3)
    predict_df = pd.DataFrame(list(itertools.product(halfgold_vals, doping_vals)),
                              columns=['halfgold_ratio', 'doping_concentration'])

# 型を確認・変換
predict_df['halfgold_ratio'] = pd.to_numeric(predict_df['halfgold_ratio'], errors='coerce')
predict_df['doping_concentration'] = pd.to_numeric(predict_df['doping_concentration'], errors='coerce')

# 80 と 70 のサンプル表示
print("== 80 のサンプル ==")
print(predict_df[predict_df['halfgold_ratio'] == 80].head())
print("== 70 のサンプル ==")
print(predict_df[predict_df['halfgold_ratio'] == 70].head())

# 予測列がなければ作成（gs または rf を使う）
if 'predicted_poly' not in predict_df.columns:
    if 'gs' in globals():
        predict_df['predicted_poly'] = gs.predict(predict_df[['halfgold_ratio','doping_concentration']])
    elif 'rf' in globals():
        predict_df['predicted_rf'] = rf.predict(predict_df[['halfgold_ratio','doping_concentration']])

# どの予測列を使うか決定
if 'predicted_poly' in predict_df.columns:
    col = 'predicted_poly'
elif 'predicted_rf' in predict_df.columns:
    col = 'predicted_rf'
else:
    raise RuntimeError("予測列がありません。gs または rf を定義して予測を作成してください。")

# 整列して差分を計算
pred80 = predict_df.loc[predict_df['halfgold_ratio'] == 80, :].sort_values('doping_concentration')[col].values
pred70 = predict_df.loc[predict_df['halfgold_ratio'] == 70, :].sort_values('doping_concentration')[col].values

if len(pred80) != len(pred70):
    print("80 と 70 のサンプル数が一致しません。")
else:
    diff = pred80 - pred70
    print("差分の統計: min, mean, max =", np.min(diff), np.mean(diff), np.max(diff))
    # 最初の数行を表示
    print("doping, pred80, pred70, diff (先頭10):")
    vals = predict_df.loc[predict_df['halfgold_ratio'].isin([80,70])].pivot_table(
        index='doping_concentration', columns='halfgold_ratio', values=col
    ).sort_index()
    print(vals.head(10))

# モデルの中身を確認（多項式 + 線形の場合）
if 'gs' in globals():
    est = gs.best_estimator_
    if hasattr(est, 'named_steps') and 'lr' in est.named_steps:
        poly = est.named_steps.get('poly', None)
        lr = est.named_steps['lr']
        print("線形係数（lr.coef_） shape:", getattr(lr, 'coef_', None).shape)
        if poly is not None and hasattr(poly, 'get_feature_names_out'):
            feat_names = poly.get_feature_names_out(['halfgold_ratio','doping_concentration'])
            coef = lr.coef_
            print("特徴量名と係数（上位20まで）:")
            for n, c in zip(feat_names, coef)[:20]:
                print(n, c)
        else:
            print("PolynomialFeatures がないか get_feature_names_out が使えません。")
# RandomForest の場合
if 'rf' in globals():
    print("RF feature_importances_:", rf.feature_importances_)