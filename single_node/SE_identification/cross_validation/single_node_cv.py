import os
import time
import pickle
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


# ============================================================
# Repeated stratified 5-fold cross-validation
# 5 random seeds × 5 folds = 25 runs
# ============================================================

random_seeds = [123, 456, 789, 1024, 2026]
n_splits = 5

train = 1

macro_f1_lr_all = []
macro_f1_knn_all = []
macro_f1_rf_all = []
macro_f1_xgb_all = []

train_LR_all = []
train_KNN_all = []
train_RF_all = []
train_XGB_all = []

test_LR_all = []
test_KNN_all = []
test_RF_all = []
test_XGB_all = []


def select_rows(data, indices):
    """同时兼容 pandas DataFrame/Series 和 NumPy array。"""
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


for seed in random_seeds:

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    for fold, (train_index, test_index) in enumerate(
        skf.split(X, y), start=1
    ):

        print(
            f"\nSeed: {seed}, Fold: {fold}/{n_splits}"
        )

        X_train = select_rows(X, train_index)
        X_test = select_rows(X, test_index)

        y_train = select_rows(y, train_index)
        y_test = select_rows(y, test_index)

        # 转成一维，避免部分模型收到列向量
        y_train = np.asarray(y_train).ravel()
        y_test = np.asarray(y_test).ravel()

        # ====================================================
        # Training
        # ====================================================

        if train == 1:

            train_LR = -time.time()

            LR(
                X_train,
                y_train,
                normalize=False,
                save_model=(
                    '\\SE_identification'
                    '\\classification_models_1'
                    '\\lr_model.pickle'
                ),
                save_param=(
                    '\\SE_identification'
                    '\\classification_models_1'
                    '\\lr_parametres.pickle'
                )
            )

            train_LR += time.time()
            train_LR_all.append(train_LR)

            print('LR')


            train_KNN = -time.time()

            KNN(
                X_train,
                y_train,
                normalize=False,
                save_model=(
                    '\\SE_identification'
                    '\\classification_models_1'
                    '\\knn_model.pickle'
                ),
                save_param=(
                    '\\SE_identification'
                    '\\classification_models_1'
                    '\\knn_parametres.pickle'
                )
            )

            train_KNN += time.time()
            train_KNN_all.append(train_KNN)

            print('KNN')


            train_RF = -time.time()

            RF(
                X_train,
                y_train,
                normalize=False,
                save_model=(
                    '\\SE_identification'
                    '\\classification_models_1'
                    '\\rf_model.pickle'
                ),
                save_param=(
                    '\\SE_identification'
                    '\\classification_models_1'
                    '\\rf_parametres.pickle'
                )
            )

            train_RF += time.time()
            train_RF_all.append(train_RF)

            print('RF')


            train_XGB = -time.time()

            XGB(
                X_train,
                y_train,
                normalize=False,
                save_model=(
                    '\\SE_identification'
                    '\\classification_models_1'
                    '\\xgb_model.pickle'
                ),
                save_param=(
                    '\\SE_identification'
                    '\\classification_models_1'
                    '\\xgb_parametres.pickle'
                ),
                save_encoder=(
                    '\\SE_identification'
                    '\\classification_models_1'
                    '\\xgb_label_encoder.pickle'
                )
            )

            train_XGB += time.time()
            train_XGB_all.append(train_XGB)

            print('XGB')

        # ====================================================
        # Load models trained in the current fold
        # ====================================================

        model_path = os.path.join(
            module_path,
            'single_attack',
            'SE_identification',
            'classification_models_1'
        )

        lr = pickle.load(
            open(
                os.path.join(
                    model_path,
                    'lr_model.pickle'
                ),
                'rb'
            )
        )

        knn = pickle.load(
            open(
                os.path.join(
                    model_path,
                    'knn_model.pickle'
                ),
                'rb'
            )
        )

        rf = pickle.load(
            open(
                os.path.join(
                    model_path,
                    'rf_model.pickle'
                ),
                'rb'
            )
        )

        xgb = pickle.load(
            open(
                os.path.join(
                    model_path,
                    'xgb_model.pickle'
                ),
                'rb'
            )
        )

        le = pickle.load(
            open(
                os.path.join(
                    model_path,
                    'xgb_label_encoder.pickle'
                ),
                'rb'
            )
        )

        # ====================================================
        # Prediction
        # ====================================================

        test_LR = -time.time()
        y_pred_lr = lr.predict(X_test)
        test_LR += time.time()
        test_LR_all.append(test_LR)

        test_KNN = -time.time()
        y_pred_knn = knn.predict(X_test)
        test_KNN += time.time()
        test_KNN_all.append(test_KNN)

        test_RF = -time.time()
        y_pred_rf = rf.predict(X_test)
        test_RF += time.time()
        test_RF_all.append(test_RF)

        test_XGB = -time.time()
        y_pred_xgb = xgb.predict(X_test)
        test_XGB += time.time()
        test_XGB_all.append(test_XGB)

        y_pred_xgb = le.inverse_transform(
            np.asarray(y_pred_xgb).astype(int)
        )

        # ====================================================
        # Macro-F1 of the current fold
        # ====================================================

        macro_f1_lr = f1_score(
            y_test,
            y_pred_lr,
            average='macro'
        )

        macro_f1_knn = f1_score(
            y_test,
            y_pred_knn,
            average='macro'
        )

        macro_f1_rf = f1_score(
            y_test,
            y_pred_rf,
            average='macro'
        )

        macro_f1_xgb = f1_score(
            y_test,
            y_pred_xgb,
            average='macro'
        )

        macro_f1_lr_all.append(macro_f1_lr)
        macro_f1_knn_all.append(macro_f1_knn)
        macro_f1_rf_all.append(macro_f1_rf)
        macro_f1_xgb_all.append(macro_f1_xgb)

        print(
            f"Macro-F1: "
            f"LR={macro_f1_lr:.4f}, "
            f"KNN={macro_f1_knn:.4f}, "
            f"RF={macro_f1_rf:.4f}, "
            f"XGB={macro_f1_xgb:.4f}"
        )


# ============================================================
# Mean and standard deviation over 25 runs
# ============================================================

macro_f1_lr = np.mean(macro_f1_lr_all)
macro_f1_knn = np.mean(macro_f1_knn_all)
macro_f1_rf = np.mean(macro_f1_rf_all)
macro_f1_xgb = np.mean(macro_f1_xgb_all)

macro_f1_lr_std = np.std(
    macro_f1_lr_all,
    ddof=1
)

macro_f1_knn_std = np.std(
    macro_f1_knn_all,
    ddof=1
)

macro_f1_rf_std = np.std(
    macro_f1_rf_all,
    ddof=1
)

macro_f1_xgb_std = np.std(
    macro_f1_xgb_all,
    ddof=1
)


print('\nRepeated stratified 5-fold CV results')
print('-------------------------------------')

print(
    f'LR   Macro-F1: '
    f'{macro_f1_lr * 100:.2f} '
    f'± {macro_f1_lr_std * 100:.2f}%'
)

print(
    f'KNN  Macro-F1: '
    f'{macro_f1_knn * 100:.2f} '
    f'± {macro_f1_knn_std * 100:.2f}%'
)

print(
    f'RF   Macro-F1: '
    f'{macro_f1_rf * 100:.2f} '
    f'± {macro_f1_rf_std * 100:.2f}%'
)

print(
    f'XGB  Macro-F1: '
    f'{macro_f1_xgb * 100:.2f} '
    f'± {macro_f1_xgb_std * 100:.2f}%'
)


# ============================================================
# Average training time per fold
# ============================================================

train_time = data_preparation.to_dict(
    np.mean(train_LR_all),
    np.mean(train_KNN_all),
    np.mean(train_RF_all),
    np.mean(train_XGB_all)
)

data_preparation.save_model(
    train_time,
    'time_1/train_time'
)