Evet, notebook’a baktım. **Submit kodun doğru.** `cb_oof_preds` bug’ı da düzeltilmiş:

```python
cb_oof_preds = np.zeros(len(X_cb))
...
cb_oof_preds[val_idx] = cb_val_preds
```

Ve final submit de mantıklı:

```python
final_blended_preds = np.clip(
    (test_preds_catboost * 0.45) + (test_preds_tabpfn * 0.55), 
    0, 100
)
```

Yani public 87.4’ün sebebi büyük ihtimalle **submission formülü değil**.

Ama notebook’ta benim gördüğüm asıl problem şu:

## En büyük sorun: CV fazla iyimser olabilir

Senin CV tamamen random `KFold`:

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

Eğer public test dağılımı yıl, target role, üniversite tier, department veya başka bir segmentte train’den farklıysa random KFold seni kandırır. Public farkının ana sebebi bu gibi duruyor.

Bence şu an yapılacak ilk şey model eklemek değil, **validation’ı sertleştirmek**.

## Hemen denenecek şey

Random KFold yerine target-bin stratified CV kur:

```python
from sklearn.model_selection import StratifiedKFold
import pandas as pd

y_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

CatBoost loop:

```python
for fold, (train_idx, val_idx) in enumerate(kf.split(X_cb, y_bins)):
    ...
```

TabPFN loop:

```python
for fold, (train_idx, val_idx) in enumerate(kf.split(X_enc, y_bins)):
    ...
```

Bu değişiklikten sonra blend OOF kötüleşirse şaşırma. Ama public’e daha yakın olur.

## İkinci önemli nokta

Notebook’ta CatBoost TF-IDF kullanıyor, TabPFN ise TF-IDF’i drop edip SVD kullanıyor:

```python
X_tr_tab = X_enc.iloc[train_idx].drop(columns=tfidf_cols)
```

Bu iyi fikir. Ama CatBoost tarafında 500 TF-IDF feature biraz noisy olabilir. Public’te kötüleşme yapıyor olabilir. O yüzden 3 submit karşılaştırması çok değerli:

```python
# 1) mevcut
CatBoost TF-IDF + TabPFN SVD blend

# 2) CatBoost TF-IDF'siz + TabPFN SVD blend

# 3) sadece daha regularized CatBoost + TabPFN
```

CatBoost TF-IDF’siz versiyon için:

```python
drop_cols = svd_cols + tfidf_cols + ['student_id', 'career_success_score', 'mentor_feedback_text']
cb_features = [c for c in train_df.columns if c not in drop_cols]
```

Bu CV’de biraz kötüleşebilir ama public’te iyileşebilir. Çünkü SHAP’ta TF-IDF kelimeleri var ama ana sinyalin yanında zayıf ve biraz leaderboard overfit riski taşıyor.

## Üçüncü nokta: CatBoost’u regularize et

Şu an model train’de 6 RMSE’ye kadar düşüyor, val 9 civarı. Overfit var. Şunu dene:

```python
cb_model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.025,
    depth=5,
    l2_leaf_reg=8,
    random_strength=1.5,
    bagging_temperature=0.5,
    cat_features=cb_cat_features,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=250
)
```

Daha da güvenli alternatif:

```python
cb_model = CatBoostRegressor(
    iterations=4000,
    learning_rate=0.02,
    depth=4,
    l2_leaf_reg=10,
    random_strength=2,
    bagging_temperature=1,
    cat_features=cb_cat_features,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=250
)
```

## Benim net kararım

Submit kodunda hata aramayı bırakırdım. Hata orada değil.

Şimdi sırayla:

1. `StratifiedKFold` target-bin CV’ye geç.
2. CatBoost’u TF-IDF’siz dene.
3. CatBoost’u depth 4/5 ve daha yüksek regularization ile dene.
4. Blend ağırlığını yine OOF ile seç ama public için 0.50, 0.55, 0.60 ayrı dene.

En hızlı deney şu olur:

```python
drop_cols = svd_cols + tfidf_cols + ['student_id', 'career_success_score', 'mentor_feedback_text']
```

Yani CatBoost’tan TF-IDF’i de çıkar. Eğer public artarsa, sorun metin feature noise’uymuş demektir.
