import os
import json
import hashlib
import importlib
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    xgb = None
    HAS_XGB = False

try:
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf = importlib.import_module("tensorflow")
    keras = importlib.import_module("tensorflow.keras")
    layers = importlib.import_module("tensorflow.keras.layers")
    tf.get_logger().setLevel(logging.ERROR)

    HAS_TF = True
except Exception:
    HAS_TF = False
    tf = None
    keras = None
    layers = None


TARGETS = ["number", "color", "big_small"]
BASE_MODELS = ["rf", "knn", "lr"]
SYNTHESIS_WEIGHTS = {"xgb": 0.35, "rf": 0.30, "knn": 0.15, "lr": 0.10, "lstm": 0.10}

DEFAULT_DATA_PATH = "Untitled spreadsheet (1).xlsx"
ARTIFACT_DIR = "model_artifacts"
STATE_PATH = os.path.join(ARTIFACT_DIR, "model_state.joblib")
LSTM_PATH = os.path.join(ARTIFACT_DIR, "lstm_multi_output.keras")
META_PATH = os.path.join(ARTIFACT_DIR, "meta.json")
STATE_VERSION = 2


@dataclass
class TargetPrediction:
    label: str
    confidence: float
    scores: Dict[str, float]


class TimeSeriesPredictor:
    def __init__(self, data_path: str = DEFAULT_DATA_PATH):
        self.data_path = data_path
        self.seq_len = 8
        self.retrain_batch_size = 5
        self.pending_updates = 0
        self.last_train_size = 0
        self.feature_dim = 0
        self.lstm_model = None

        self.pattern_len = 5
        self.pattern_memory = defaultdict(Counter)
        self.evaluation: Dict[str, Dict[str, object]] = {}

        self.models: Dict[str, Dict[str, object]] = {t: {} for t in TARGETS}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.performance_stats: Dict[str, Dict[str, int]] = {t: {"correct": 0, "wrong": 0} for t in TARGETS}

        os.makedirs(ARTIFACT_DIR, exist_ok=True)

    @staticmethod
    def _normalize_big_small(value: str) -> str:
        if pd.isna(value):
            return "small"
        v = str(value).strip().lower()
        if v in {"big", "b", "1", "high", "large"}:
            return "big"
        return "small"

    @staticmethod
    def _normalize_color(value: str) -> str:
        if pd.isna(value):
            return "green"
        text = str(value).strip().lower().replace(" ", "")
        parts = [p for p in text.replace("|", "/").replace(",", "/").split("/") if p]
        allowed = {"green", "red", "pink"}
        parts = [p for p in parts if p in allowed]
        if not parts:
            return "green"
        return "/".join(sorted(set(parts)))

    @staticmethod
    def _normalize_number(value) -> int:
        if pd.isna(value):
            return 0
        return int(np.clip(round(float(value)), 0, 9))

    @staticmethod
    def _normalize_period_value(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, str):
            text = value.strip()
            if text.endswith(".0"):
                text = text[:-2]
            return text
        try:
            if isinstance(value, (int, np.integer)):
                return str(int(value))
            if isinstance(value, float):
                if value.is_integer():
                    return str(int(value))
                return format(value, ".15g")
            return str(value).strip()
        except Exception:
            return str(value).strip()

    @staticmethod
    def _num_to_big_small(num: int) -> str:
        return "big" if int(num) >= 5 else "small"

    def _read_dataset(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        p = self.data_path.lower()
        if p.endswith(".csv"):
            return pd.read_csv(self.data_path)
        if p.endswith(".xlsx"):
            try:
                return pd.read_excel(self.data_path, engine="openpyxl")
            except Exception:
                return pd.read_csv(self.data_path)
        if p.endswith(".xls"):
            try:
                return pd.read_excel(self.data_path, engine="xlrd")
            except Exception:
                return pd.read_excel(self.data_path, engine="openpyxl")

        try:
            return pd.read_excel(self.data_path, engine="openpyxl")
        except Exception:
            return pd.read_csv(self.data_path)

    def _save_dataset(self, df: pd.DataFrame) -> None:
        p = self.data_path.lower()
        if p.endswith(".csv"):
            df.to_csv(self.data_path, index=False)
            return
        if p.endswith(".xlsx"):
            df.to_excel(self.data_path, index=False, engine="openpyxl")
            return
        if p.endswith(".xls"):
            df.to_excel(self.data_path, index=False)
            return
        df.to_csv(self.data_path, index=False)

    def load_data(self) -> pd.DataFrame:
        df = self._read_dataset()
        required = {"Period", "Number", "Big/Small", "Color"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = df[["Period", "Number", "Big/Small", "Color"]].copy()
        df["Period"] = df["Period"].apply(self._normalize_period_value)
        period_numeric = pd.to_numeric(df["Period"], errors="coerce")
        df["Number"] = pd.to_numeric(df["Number"], errors="coerce")

        period_numeric = period_numeric.ffill().bfill()
        if period_numeric.isna().any():
            period_numeric = pd.Series(np.arange(1, len(df) + 1), index=df.index, dtype=float)

        df["Number"] = df["Number"].apply(self._normalize_number)

        bs_mode = "small"
        color_mode = "green"
        if df["Big/Small"].notna().any():
            bs_mode = self._normalize_big_small(df["Big/Small"].dropna().mode().iloc[0])
        if df["Color"].notna().any():
            color_mode = self._normalize_color(df["Color"].dropna().mode().iloc[0])

        df["Big/Small"] = df["Big/Small"].fillna(bs_mode).map(self._normalize_big_small)
        df["Color"] = df["Color"].fillna(color_mode).map(self._normalize_color)

        df["_period_numeric"] = period_numeric
        df = df.sort_values("_period_numeric").reset_index(drop=True)
        df = df[["Period", "Number", "Big/Small", "Color"]]
        return df

    def _estimate_cycle_len(self, periods: np.ndarray) -> int:
        if len(periods) < 12:
            return 1
        diffs = np.diff(periods)
        max_cycle = min(20, len(diffs) // 2)
        best_cycle = 1
        best_score = 0.0
        for c in range(2, max_cycle + 1):
            a = diffs[-c:]
            b = diffs[-2 * c:-c]
            if len(a) != len(b) or len(a) == 0:
                continue
            score = float(np.mean(a == b))
            if score > best_score:
                best_score = score
                best_cycle = c
        return best_cycle

    @staticmethod
    def _recent_sample_weights(n: int) -> np.ndarray:
        if n <= 0:
            return np.array([])
        w = np.exp(np.linspace(-2.2, 0.0, n))
        if n >= 20:
            w[-20:] *= 1.5
        w /= w.mean()
        return w

    def _fit_encoders(self, y_number: List[int], y_color: List[str], y_bs: List[str]) -> None:
        num_enc = LabelEncoder()
        num_enc.fit(sorted({str(v) for v in y_number}))
        self.label_encoders["number"] = num_enc

        color_enc = LabelEncoder()
        color_enc.fit(sorted(set(y_color)))
        self.label_encoders["color"] = color_enc

        bs_enc = LabelEncoder()
        bs_enc.fit(sorted(set(y_bs)))
        self.label_encoders["big_small"] = bs_enc

    def _build_pattern_memory(self, df: pd.DataFrame) -> None:
        self.pattern_memory = defaultdict(Counter)
        if len(df) <= self.pattern_len:
            return

        for i in range(self.pattern_len, len(df)):
            window = df.iloc[i - self.pattern_len:i]
            key = (
                tuple(window["Number"].tolist()),
                tuple(window["Color"].tolist()),
                tuple(window["Big/Small"].tolist()),
            )
            nxt = (df.loc[i, "Number"], df.loc[i, "Color"], df.loc[i, "Big/Small"])
            self.pattern_memory[key][nxt] += 1

    @staticmethod
    def _stable_hash(value: str, modulo: int = 10000) -> float:
        digest = hashlib.md5(value.encode("utf-8")).hexdigest()
        return (int(digest[:12], 16) % modulo) / float(modulo)

    def _window_feature_vector(
        self,
        window: pd.DataFrame,
        period_numbers: np.ndarray,
        idx: int,
        next_period: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        color_vocab = ["green", "red", "pink"]
        color_map = {c: i for i, c in enumerate(color_vocab)}

        seq_rows = []
        for row_idx, (_, row) in enumerate(window.iterrows()):
            num_norm = float(row["Number"]) / 9.0
            bs_bin = 1.0 if str(row["Big/Small"]) == "big" else 0.0
            color_norm = float(color_map.get(str(row["Color"]), 0)) / max(1.0, len(color_vocab) - 1)
            prev_period = period_numbers[max(0, idx - self.seq_len + row_idx - 1)] if idx - self.seq_len + row_idx - 1 >= 0 else period_numbers[0]
            cur_period = period_numbers[idx - self.seq_len + row_idx]
            period_delta = (cur_period - prev_period) / 1000.0
            period_mod_5 = float(cur_period % 5) / 5.0
            period_mod_10 = float(cur_period % 10) / 10.0
            cycle_pos = float(cur_period % max(1, self._estimate_cycle_len(period_numbers))) / max(1.0, float(max(1, self._estimate_cycle_len(period_numbers))))
            seq_rows.append([num_norm, bs_bin, color_norm, period_delta, period_mod_5, period_mod_10, cycle_pos])

        seq_arr = np.array(seq_rows, dtype=np.float32)

        nums = window["Number"].astype(float).to_numpy()
        bs_vals = (window["Big/Small"].astype(str) == "big").astype(float).to_numpy()
        color_vals = window["Color"].astype(str).to_numpy()

        last3_num = np.pad(nums[-3:] / 9.0, (3 - min(3, len(nums)), 0), mode="constant")
        last3_bs = np.pad(bs_vals[-3:], (3 - min(3, len(bs_vals)), 0), mode="constant")
        last3_color = np.pad(np.array([color_map.get(c, 0) for c in color_vals[-3:]], dtype=float) / 2.0, (3 - min(3, len(color_vals)), 0), mode="constant")

        last5 = nums[-5:] if len(nums) >= 5 else nums
        last10 = nums[-10:] if len(nums) >= 10 else nums
        color10 = color_vals[-10:] if len(color_vals) >= 10 else color_vals
        bs10 = bs_vals[-10:] if len(bs_vals) >= 10 else bs_vals

        num_mean5 = float(np.mean(last5) / 9.0) if len(last5) else 0.0
        num_mean10 = float(np.mean(last10) / 9.0) if len(last10) else 0.0
        num_std5 = float(np.std(last5) / 9.0) if len(last5) else 0.0
        num_std10 = float(np.std(last10) / 9.0) if len(last10) else 0.0
        num_trend5 = float((last5[-1] - last5[0]) / max(1.0, len(last5) - 1) / 9.0) if len(last5) >= 2 else 0.0
        num_ma5 = num_mean5
        num_ma10 = num_mean10

        color_freq = np.array([np.mean(color10 == c) if len(color10) else 0.0 for c in color_vocab], dtype=float)
        bs_freq = np.array([float(np.mean(bs10 == 1.0)) if len(bs10) else 0.0, float(np.mean(bs10 == 0.0)) if len(bs10) else 0.0], dtype=float)
        num_hist = np.array([np.mean(last10 == i) if len(last10) else 0.0 for i in range(10)], dtype=float)

        transition_repeat_num = float(np.mean(np.diff(last5) == 0)) if len(last5) >= 2 else 0.0
        transition_repeat_color = float(np.mean(np.diff(np.array([color_map.get(c, 0) for c in color10])) == 0)) if len(color10) >= 2 else 0.0
        transition_repeat_bs = float(np.mean(np.diff(bs10) == 0)) if len(bs10) >= 2 else 0.0

        pattern_key = "|".join([f"{int(n)}-{c}-{int(b)}" for n, c, b in zip(nums[-self.pattern_len:], color_vals[-self.pattern_len:], bs_vals[-self.pattern_len:])])
        pattern_code = self._stable_hash(pattern_key)

        features = np.concatenate([
            last3_num,
            last3_bs,
            last3_color,
            np.array([
                num_mean5,
                num_mean10,
                num_std5,
                num_std10,
                num_trend5,
                num_ma5,
                num_ma10,
                transition_repeat_num,
                transition_repeat_color,
                transition_repeat_bs,
                pattern_code,
                float(nums[-1] / 9.0) if len(nums) else 0.0,
                float(nums[-2] / 9.0) if len(nums) >= 2 else 0.0,
                float(nums[-3] / 9.0) if len(nums) >= 3 else 0.0,
                float(bs_vals[-1]) if len(bs_vals) else 0.0,
                float(bs_vals[-2]) if len(bs_vals) >= 2 else 0.0,
                float(bs_vals[-3]) if len(bs_vals) >= 3 else 0.0,
            ], dtype=float),
            color_freq,
            bs_freq,
            num_hist,
        ])

        return seq_arr, features.astype(np.float32)

    def _pattern_probs(self, history_df: pd.DataFrame, target: str, classes: List[str]) -> Optional[np.ndarray]:
        if len(history_df) < self.pattern_len:
            return None

        window = history_df.iloc[-self.pattern_len:]
        key = (
            tuple(window["Number"].tolist()),
            tuple(window["Color"].tolist()),
            tuple(window["Big/Small"].tolist()),
        )
        c = self.pattern_memory.get(key)
        if not c:
            return None

        agg = Counter()
        total = 0
        for k, v in c.items():
            if target == "number":
                agg[str(k[0])] += v
            elif target == "color":
                agg[str(k[1])] += v
            else:
                agg[str(k[2])] += v
            total += v

        if total == 0:
            return None

        out = np.zeros(len(classes), dtype=float)
        idx_map = {classes[i]: i for i in range(len(classes))}
        for label, cnt in agg.items():
            if label in idx_map:
                out[idx_map[label]] = cnt / total
        if out.sum() == 0:
            return None
        return out / out.sum()

    def _build_matrices(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[int], List[str], List[str]]:
        if len(df) <= self.seq_len:
            raise ValueError(f"Need at least {self.seq_len + 5} rows for sequence training")

        period_numbers = pd.to_numeric(df["Period"], errors="coerce").ffill().bfill().to_numpy()
        X_seq = []
        X_flat = []
        y_number = []
        y_color = []
        y_bs = []

        for i in range(self.seq_len, len(df)):
            window = df.iloc[i - self.seq_len:i]
            seq_arr, flat_arr = self._window_feature_vector(window, period_numbers, i)
            X_seq.append(seq_arr)
            X_flat.append(flat_arr)
            y_number.append(int(df.iloc[i]["Number"]))
            y_color.append(str(df.iloc[i]["Color"]))
            y_bs.append(str(df.iloc[i]["Big/Small"]))

        X_seq = np.array(X_seq, dtype=np.float32)
        X_flat = np.array(X_flat, dtype=np.float32)
        self.feature_dim = int(X_seq.shape[-1])
        return X_seq, X_flat, y_number, y_color, y_bs

    def _build_lstm_model(self, n_num: int, n_color: int, n_bs: int):
        inp = keras.Input(shape=(self.seq_len, self.feature_dim), name="seq_input")
        x = layers.LSTM(32, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)(inp)
        x = layers.LayerNormalization()(x)
        x = layers.LSTM(16, dropout=0.25)(x)
        shared = layers.Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        shared = layers.Dropout(0.2)(shared)

        out_num = layers.Dense(n_num, activation="softmax", name="number_head")(shared)
        out_color = layers.Dense(n_color, activation="softmax", name="color_head")(shared)
        out_bs = layers.Dense(n_bs, activation="softmax", name="big_small_head")(shared)

        model = keras.Model(inputs=inp, outputs={"number_head": out_num, "color_head": out_color, "big_small_head": out_bs})
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0007),
            loss={
                "number_head": "sparse_categorical_crossentropy",
                "color_head": "sparse_categorical_crossentropy",
                "big_small_head": "sparse_categorical_crossentropy",
            },
            metrics={
                "number_head": ["accuracy"],
                "color_head": ["accuracy"],
                "big_small_head": ["accuracy"],
            },
        )
        return model

    def _make_models(self, n_classes: int) -> Dict[str, object]:
        if HAS_XGB:
            xgb_model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                objective="multi:softprob",
                num_class=max(2, n_classes),
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            )
        else:
            xgb_model = GradientBoostingClassifier(random_state=42)

        rf = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
        )

        knn = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        )

        lr = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1500, random_state=42)),
            ]
        )

        return {"xgb": xgb_model, "rf": rf, "knn": knn, "lr": lr}

    def train(self, force: bool = True) -> Dict[str, str]:
        df = self.load_data()
        self._build_pattern_memory(df)

        X_seq, X_flat, y_num_raw, y_color_raw, y_bs_raw = self._build_matrices(df)

        y_combo = np.array([f"{a}|{b}|{c}" for a, b, c in zip(y_num_raw, y_color_raw, y_bs_raw)])
        combo_counts = Counter(y_combo.tolist())
        stratify_combo = y_combo if combo_counts and min(combo_counts.values()) >= 2 else None
        X_train_idx, X_test_idx = train_test_split(
            np.arange(len(X_flat)),
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=stratify_combo if len(np.unique(y_combo)) > 1 else None,
        )

        X_seq_train, X_seq_test = X_seq[X_train_idx], X_seq[X_test_idx]
        X_flat_train, X_flat_test = X_flat[X_train_idx], X_flat[X_test_idx]

        y_num_train_raw = [y_num_raw[i] for i in X_train_idx]
        y_color_train_raw = [y_color_raw[i] for i in X_train_idx]
        y_bs_train_raw = [y_bs_raw[i] for i in X_train_idx]
        y_num_test_raw = [y_num_raw[i] for i in X_test_idx]
        y_color_test_raw = [y_color_raw[i] for i in X_test_idx]
        y_bs_test_raw = [y_bs_raw[i] for i in X_test_idx]

        self._fit_encoders(y_num_train_raw, y_color_train_raw, y_bs_train_raw)

        enc_num = self.label_encoders["number"]
        enc_color = self.label_encoders["color"]
        enc_bs = self.label_encoders["big_small"]

        y_num = enc_num.transform([str(v) for v in y_num_train_raw])
        y_color = enc_color.transform(y_color_train_raw)
        y_bs = enc_bs.transform(y_bs_train_raw)

        y_train = {
            "number": y_num,
            "color": y_color,
            "big_small": y_bs,
        }
        y_test = {
            "number": y_num_test_raw,
            "color": y_color_test_raw,
            "big_small": y_bs_test_raw,
        }

        sw = self._recent_sample_weights(len(X_flat_train))
        summary = {}

        self.evaluation = {}

        if HAS_TF:
            self.lstm_model = self._build_lstm_model(len(enc_num.classes_), len(enc_color.classes_), len(enc_bs.classes_))
            callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
            self.lstm_model.fit(
                X_seq_train,
                {
                    "number_head": y_train["number"],
                    "color_head": y_train["color"],
                    "big_small_head": y_train["big_small"],
                },
                sample_weight={
                    "number_head": sw,
                    "color_head": sw,
                    "big_small_head": sw,
                },
                validation_split=0.2,
                epochs=25,
                batch_size=32,
                verbose=0,
                callbacks=callbacks,
            )
            self.lstm_model.save(LSTM_PATH)
            summary["lstm"] = "trained"
        else:
            self.lstm_model = None
            summary["lstm"] = "unavailable_tensorflow_missing"

        for t_name, y_tr, y_te in [("number", y_train["number"], y_test["number"]), ("color", y_train["color"], y_test["color"]), ("big_small", y_train["big_small"], y_test["big_small"] )]:
            models = self._make_models(len(np.unique(y_tr)))
            trained_models = {}
            for m_name, model in models.items():
                if m_name == "knn":
                    model.fit(X_flat_train, y_tr)
                elif m_name == "lr":
                    model.fit(X_flat_train, y_tr, clf__sample_weight=sw)
                else:
                    if hasattr(model, "fit"):
                        try:
                            model.fit(X_flat_train, y_tr, sample_weight=sw)
                        except TypeError:
                            model.fit(X_flat_train, y_tr)
                trained_models[m_name] = model

            self.models[t_name] = trained_models

            encoder = self.label_encoders[t_name]
            known_classes = set(encoder.classes_.tolist())
            test_mask = np.array([str(v) in known_classes for v in y_te], dtype=bool)
            y_te_filtered_raw = [y_te[i] for i in range(len(y_te)) if test_mask[i]]
            X_flat_test_filtered = X_flat_test[test_mask]
            X_seq_test_filtered = X_seq_test[test_mask]

            eval_pred = {}
            test_accs = {}
            for m_name, model in trained_models.items():
                if len(y_te_filtered_raw) > 0:
                    y_te_filtered = encoder.transform([str(v) for v in y_te_filtered_raw])
                    preds = self._coerce_class_predictions(model.predict(X_flat_test_filtered))
                    test_accs[m_name] = self._simple_accuracy(y_te_filtered, preds)
                else:
                    preds = np.array([])
                    test_accs[m_name] = float("nan")
                eval_pred[m_name] = preds

            if HAS_TF and self.lstm_model is not None:
                if len(y_te_filtered_raw) > 0:
                    lstm_probs_dict = self.lstm_model.predict(X_seq_test_filtered, verbose=0)
                    head_map = {"number": "number_head", "color": "color_head", "big_small": "big_small_head"}
                    head_name = head_map[t_name]
                    lstm_preds = np.argmax(lstm_probs_dict[head_name], axis=1)
                    test_accs["lstm"] = self._simple_accuracy(y_te_filtered, lstm_preds)
                else:
                    test_accs["lstm"] = float("nan")

            cv_model = models["xgb"]
            cv_scores = self._safe_cv_scores(cv_model, X_flat_train, y_tr)

            self.evaluation[t_name] = {
                "test_accuracy": test_accs,
                "cv_accuracy_mean": float(np.mean(cv_scores)),
                "cv_accuracy_std": float(np.std(cv_scores)),
                "test_size": int(len(X_test_idx)),
            }
            summary[t_name] = "trained_xgb_rf_knn_lr"

        self.pending_updates = 0
        self.last_train_size = len(df)
        self.save_state()
        return summary

    @staticmethod
    def _safe_cv_scores(model, x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        if len(unique_classes) < 2 or len(y_train) < 2:
            return np.array([float("nan")])

        min_class_count = int(np.min(class_counts))
        if min_class_count >= 2:
            cv_splits = min(5, min_class_count)
            cv_splits = max(2, cv_splits)
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        else:
            n_splits = min(5, len(y_train))
            if n_splits < 2:
                return np.array([float("nan")])
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        try:
            def safe_accuracy(y_true, y_pred):
                return TimeSeriesPredictor._simple_accuracy(y_true, TimeSeriesPredictor._coerce_class_predictions(y_pred))

            scorer = make_scorer(safe_accuracy)
            return cross_val_score(model, x_train, y_train, cv=cv, scoring=scorer)
        except Exception:
            return np.array([float("nan")])

    def _extract_single_sample(self, df: pd.DataFrame, next_period: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(df) < self.seq_len:
            raise ValueError(f"Need at least {self.seq_len} rows to predict")

        period_numbers = pd.to_numeric(df["Period"], errors="coerce").ffill().bfill().to_numpy()
        window = df.iloc[-self.seq_len:]
        seq_arr, flat_arr = self._window_feature_vector(window, period_numbers, len(df) - 1, next_period=next_period)
        return seq_arr[np.newaxis, :, :].reshape(1, self.seq_len, self.feature_dim), flat_arr.reshape(1, -1)

    @staticmethod
    def _align_probs(model, x_flat: np.ndarray, n_classes: int) -> np.ndarray:
        p = model.predict_proba(x_flat)[0]
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            classes = model.named_steps["clf"].classes_
        out = np.zeros(n_classes, dtype=float)
        for i, cls in enumerate(classes):
            out[int(cls)] = p[i]
        if out.sum() == 0:
            return np.ones(n_classes) / n_classes
        return out / out.sum()

    @staticmethod
    def _coerce_class_predictions(preds: np.ndarray) -> np.ndarray:
        preds = np.asarray(preds)
        if preds.ndim == 2:
            return np.argmax(preds, axis=1)
        return preds.reshape(-1)

    @staticmethod
    def _simple_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        if len(y_true) == 0 or len(y_pred) == 0:
            return float("nan")
        limit = min(len(y_true), len(y_pred))
        return float(np.mean(y_true[:limit] == y_pred[:limit]))

    @staticmethod
    def _apply_diversity_penalty(probs: np.ndarray, class_labels: List[str], last_value: str) -> np.ndarray:
        out = probs.copy()
        top_idx = int(np.argmax(out))
        if class_labels[top_idx] == last_value:
            out[top_idx] *= 0.55
            out = out / out.sum()
        return out

    def _target_prediction(
        self,
        target: str,
        seq_x: np.ndarray,
        flat_x: np.ndarray,
        history_df: pd.DataFrame,
    ) -> TargetPrediction:
        enc = self.label_encoders[target]
        classes = enc.classes_.tolist()
        n_classes = len(classes)

        available_weights = {}
        probs_parts = {}

        if self.lstm_model is not None:
            lstm_probs_dict = self.lstm_model.predict(seq_x, verbose=0)
            head_map = {"number": "number_head", "color": "color_head", "big_small": "big_small_head"}
            head_name = head_map[target]
            probs_parts["lstm"] = lstm_probs_dict[head_name][0]
            available_weights["lstm"] = SYNTHESIS_WEIGHTS["lstm"]

        for m_name, m in self.models[target].items():
            probs_parts[m_name] = self._align_probs(m, flat_x, n_classes)
            available_weights[m_name] = SYNTHESIS_WEIGHTS[m_name]

        total_w = sum(available_weights.values())
        if total_w <= 0:
            total_w = 1.0
        for k in list(available_weights.keys()):
            available_weights[k] /= total_w

        final = np.zeros(n_classes, dtype=float)
        for m_name, p in probs_parts.items():
            final += available_weights[m_name] * p

        pattern = self._pattern_probs(history_df, target, classes)
        if pattern is not None:
            final = 0.85 * final + 0.15 * pattern

        last_label = ""
        if target == "number":
            last_label = str(int(history_df.iloc[-1]["Number"]))
        elif target == "color":
            last_label = str(history_df.iloc[-1]["Color"])
        else:
            last_label = str(history_df.iloc[-1]["Big/Small"])

        target_column = "Number" if target == "number" else ("Color" if target == "color" else "Big/Small")
        recent_vals = history_df[target_column].astype(str).tail(5).tolist()

        final = self._apply_diversity_penalty(final, classes, last_label)
        if len(recent_vals) >= 3 and recent_vals.count(last_label) >= 3:
            final[np.argmax(final)] *= 0.7
        final = final / final.sum()

        best_idx = int(np.argmax(final))
        pred_label = classes[best_idx]
        conf = float(final[best_idx])
        scores = {classes[i]: float(final[i]) for i in range(len(classes))}
        scores = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
        return TargetPrediction(label=pred_label, confidence=conf, scores=scores)

    def predict_next(self, period: int) -> Dict[str, object]:
        if not self.models["number"]:
            raise RuntimeError("Model is not trained")

        history = self.load_data()
        seq_x, flat_x = self._extract_single_sample(history, int(period))

        num_pred = self._target_prediction("number", seq_x, flat_x, history)
        color_pred = self._target_prediction("color", seq_x, flat_x, history)
        bs_pred = self._target_prediction("big_small", seq_x, flat_x, history)

        def _top2(scores: Dict[str, float]) -> List[Dict[str, object]]:
            items = list(scores.items())[:2]
            return [{"label": k, "probability": round(float(v), 4)} for k, v in items]

        return {
            "number_prediction": int(num_pred.label),
            "color_prediction": color_pred.label,
            "big_small_prediction": bs_pred.label,
            "synthesis_scores": {
                "number": num_pred.scores,
                "color": color_pred.scores,
                "big_small": bs_pred.scores,
            },
            "confidence": {
                "number": round(num_pred.confidence, 4),
                "color": round(color_pred.confidence, 4),
                "big_small": round(bs_pred.confidence, 4),
            },
            "top2_predictions": {
                "number": _top2(num_pred.scores),
                "color": _top2(color_pred.scores),
                "big_small": _top2(bs_pred.scores),
            },
            "probability_distribution": {
                "number": num_pred.scores,
                "color": color_pred.scores,
                "big_small": bs_pred.scores,
            },
            "_legacy": {
                "Predicted Number": int(num_pred.label),
                "Predicted Color": color_pred.label,
                "Predicted Big/Small": bs_pred.label,
                "Confidence": round(float(np.mean([num_pred.confidence, color_pred.confidence, bs_pred.confidence])) * 100, 2),
            },
        }

    def add_actual_and_retrain(
        self,
        period: int,
        number: int,
        big_small: str,
        color: str,
        force_retrain: bool = False,
    ) -> Tuple[bool, int, str]:
        df = self.load_data()
        requested_period = int(period)
        row = {
            "Period": self._normalize_period_value(requested_period),
            "Number": self._normalize_number(number),
            "Big/Small": self._normalize_big_small(big_small),
            "Color": self._normalize_color(color),
        }

        dup = df["Period"].astype(str) == row["Period"]
        if dup.any():
            df.loc[dup, ["Number", "Big/Small", "Color"]] = [row["Number"], row["Big/Small"], row["Color"]]
            save_period = row["Period"]
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            save_period = row["Period"]

        df = df.copy()
        df["_period_numeric"] = pd.to_numeric(df["Period"], errors="coerce").ffill().bfill()
        df = df.sort_values("_period_numeric").drop(columns=["_period_numeric"]).reset_index(drop=True)
        self._save_dataset(df)

        self.pending_updates += 1
        retrained = False
        if force_retrain or self.pending_updates >= self.retrain_batch_size:
            self.train(force=True)
            retrained = True
        else:
            self._build_pattern_memory(df)
            self.save_state()

        return retrained, self.pending_updates, save_period

    def save_state(self) -> None:
        payload = {
            "state_version": STATE_VERSION,
            "data_path": self.data_path,
            "seq_len": self.seq_len,
            "feature_dim": self.feature_dim,
            "retrain_batch_size": self.retrain_batch_size,
            "pending_updates": self.pending_updates,
            "last_train_size": self.last_train_size,
            "models": self.models,
            "label_encoders": self.label_encoders,
            "performance_stats": self.performance_stats,
            "evaluation": self.evaluation,
            "pattern_len": self.pattern_len,
            "pattern_memory": dict(self.pattern_memory),
        }
        joblib.dump(payload, STATE_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump({"data_path": self.data_path, "last_train_size": self.last_train_size}, f, indent=2)

    def load_state(self) -> bool:
        if not os.path.exists(STATE_PATH):
            return False
        payload = joblib.load(STATE_PATH)
        if payload.get("state_version") != STATE_VERSION:
            return False

        self.data_path = payload.get("data_path", self.data_path)
        self.seq_len = payload.get("seq_len", self.seq_len)
        self.feature_dim = payload.get("feature_dim", self.feature_dim)
        self.retrain_batch_size = payload.get("retrain_batch_size", self.retrain_batch_size)
        self.pending_updates = payload.get("pending_updates", 0)
        self.last_train_size = payload.get("last_train_size", 0)
        loaded_models = payload.get("models", {})
        self.models = {t: {} for t in TARGETS}
        self.models["number"] = loaded_models.get("number", loaded_models.get("Number", {}))
        self.models["color"] = loaded_models.get("color", loaded_models.get("Color", {}))
        self.models["big_small"] = loaded_models.get("big_small", loaded_models.get("Big/Small", {}))

        loaded_enc = payload.get("label_encoders", {})
        self.label_encoders = {}
        self.label_encoders["number"] = loaded_enc.get("number", loaded_enc.get("Number"))
        self.label_encoders["color"] = loaded_enc.get("color", loaded_enc.get("Color"))
        self.label_encoders["big_small"] = loaded_enc.get("big_small", loaded_enc.get("Big/Small"))

        loaded_perf = payload.get("performance_stats", {})
        self.performance_stats = {t: {"correct": 0, "wrong": 0} for t in TARGETS}
        self.performance_stats["number"] = loaded_perf.get("number", loaded_perf.get("Number", {"correct": 0, "wrong": 0}))
        self.performance_stats["color"] = loaded_perf.get("color", loaded_perf.get("Color", {"correct": 0, "wrong": 0}))
        self.performance_stats["big_small"] = loaded_perf.get("big_small", loaded_perf.get("Big/Small", {"correct": 0, "wrong": 0}))
        self.evaluation = payload.get("evaluation", {})
        self.pattern_len = payload.get("pattern_len", 5)

        self.pattern_memory = defaultdict(Counter)
        for k, v in payload.get("pattern_memory", {}).items():
            self.pattern_memory[k] = Counter(v)

        if HAS_TF and os.path.exists(LSTM_PATH):
            try:
                self.lstm_model = keras.models.load_model(LSTM_PATH)
            except Exception:
                self.lstm_model = None
        else:
            self.lstm_model = None

        return True


def get_predictor(data_path: str) -> TimeSeriesPredictor:
    p = TimeSeriesPredictor(data_path=data_path)
    loaded = p.load_state()
    if not loaded:
        p.train(force=True)
        return p

    try:
        df = p.load_data()
        if len(df) != p.last_train_size or not p.models["number"] or "xgb" not in p.models["number"]:
            p.train(force=True)
    except Exception:
        p.train(force=True)
    return p


def show_prediction(pred: Dict[str, object]) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Number", str(pred["number_prediction"]))
    c2.metric("Predicted Color", str(pred["color_prediction"]))
    c3.metric("Predicted Big/Small", str(pred["big_small_prediction"]))

    st.markdown("### Confidence")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Number", f"{pred['confidence']['number'] * 100:.2f}%")
    cc2.metric("Color", f"{pred['confidence']['color'] * 100:.2f}%")
    cc3.metric("Big/Small", f"{pred['confidence']['big_small'] * 100:.2f}%")

    if "top2_predictions" in pred:
        st.markdown("### Top 2 Predictions")
        for target in ["number", "color", "big_small"]:
            top2 = pred["top2_predictions"][target]
            st.write(f"{target}: {top2[0]['label']} ({top2[0]['probability']:.4f}), {top2[1]['label']} ({top2[1]['probability']:.4f})")


def main() -> None:
    st.set_page_config(page_title="Time-Series Outcome Predictor", layout="wide")
    st.title("Time-Series Outcome Prediction System")

    data_path = st.sidebar.text_input("Dataset path (CSV/XLSX)", value=DEFAULT_DATA_PATH)

    if "predictor" not in st.session_state or st.session_state.get("active_data_path") != data_path:
        with st.spinner("Model load/train ho raha hai... thoda wait karein"):
            st.session_state["predictor"] = get_predictor(data_path)
            st.session_state["active_data_path"] = data_path

    predictor: TimeSeriesPredictor = st.session_state["predictor"]

    if "prediction_by_period" not in st.session_state:
        st.session_state["prediction_by_period"] = {}
    if "actual_pred_table" not in st.session_state:
        st.session_state["actual_pred_table"] = []

    for t in TARGETS:
        if t not in predictor.performance_stats:
            predictor.performance_stats[t] = {"correct": 0, "wrong": 0}
        if t not in predictor.models:
            predictor.models[t] = {}

    if not HAS_TF:
        st.warning("TensorFlow not detected. LSTM branch is disabled until TensorFlow is installed.")

    try:
        latest_period = int(predictor.load_data()["Period"].iloc[-1])
    except Exception:
        latest_period = 0

    st.sidebar.markdown("### Model Controls")
    st.sidebar.write(f"Pending updates before auto retrain: {predictor.pending_updates}/{predictor.retrain_batch_size}")
    if st.sidebar.button("Retrain Model", width="stretch"):
        try:
            predictor.train(force=True)
            st.sidebar.success("Model retrained successfully")
        except Exception as e:
            st.sidebar.error(str(e))

    st.markdown("## Predict Next Outcome")
    left, right = st.columns([2, 1])

    with left:
        period_input = st.text_input("Period", value=str(latest_period + 1))
        if st.button("Predict", type="primary"):
            try:
                period_val = int(str(period_input).strip())
                if period_val < 0:
                    raise ValueError("Period must be a non-negative integer")
                pred = predictor.predict_next(period_val)
                st.session_state["last_prediction"] = pred
                st.session_state["prediction_by_period"][str(period_val)] = {
                    "Period": str(period_val),
                    "Predicted Number": int(pred["number_prediction"]),
                    "Predicted Color": str(pred["color_prediction"]),
                    "Predicted Big/Small": str(pred["big_small_prediction"]),
                }
                show_prediction(pred)
            except Exception as e:
                st.error(str(e))

    with right:
        st.markdown("### Performance")
        for t in TARGETS:
            c = predictor.performance_stats[t]["correct"]
            w = predictor.performance_stats[t]["wrong"]
            total = c + w
            acc = (100.0 * c / total) if total > 0 else 0.0
            st.write(f"{t}: {acc:.2f}% ({c}/{total})")
            if t in predictor.evaluation:
                st.caption(
                    f"test={predictor.evaluation[t]['test_accuracy']} | cv={predictor.evaluation[t]['cv_accuracy_mean']:.3f}±{predictor.evaluation[t]['cv_accuracy_std']:.3f}"
                )

    if "last_prediction" in st.session_state:
        show_prediction(st.session_state["last_prediction"])

    st.markdown("## Add Actual Result and Live Learn")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        actual_period = st.text_input("Actual Period", value=str(latest_period + 1))
    with a2:
        actual_number = st.number_input("Actual Number", min_value=0, max_value=9, value=0, step=1)
    with a3:
        actual_bs = st.selectbox("Actual Big/Small", ["big", "small"])
    with a4:
        actual_color = st.text_input("Actual Color", value="green")

    if st.button("Add Actual Result", width="stretch"):
        try:
            actual_period_val = int(str(actual_period).strip())
            if actual_period_val < 0:
                raise ValueError("Actual Period must be a non-negative integer")

            retrained, pending, saved_period = predictor.add_actual_and_retrain(
                period=actual_period_val,
                number=int(actual_number),
                big_small=actual_bs,
                color=actual_color,
                force_retrain=False,
            )
            st.session_state["last_actual_added"] = {
                "Period": saved_period,
                "Number": int(actual_number),
                "Big/Small": actual_bs,
                "Color": actual_color,
            }

            pred_row = st.session_state["prediction_by_period"].get(str(saved_period))
            if pred_row:
                row = {
                    "Period": str(saved_period),
                    "Actual Number": int(actual_number),
                    "Predicted Number": int(pred_row["Predicted Number"]),
                    "Number Match": bool(int(actual_number) == int(pred_row["Predicted Number"])),
                    "Actual Color": str(actual_color),
                    "Predicted Color": str(pred_row["Predicted Color"]),
                    "Color Match": bool(str(actual_color).strip().lower() == str(pred_row["Predicted Color"]).strip().lower()),
                    "Actual Big/Small": str(actual_bs),
                    "Predicted Big/Small": str(pred_row["Predicted Big/Small"]),
                    "Big/Small Match": bool(str(actual_bs).strip().lower() == str(pred_row["Predicted Big/Small"]).strip().lower()),
                }
                st.session_state["actual_pred_table"] = [
                    row,
                    *[
                        r
                        for r in st.session_state["actual_pred_table"]
                        if str(r.get("Period", "")) != str(saved_period)
                    ],
                ]

            if retrained:
                st.success("Actual result added. Batch threshold reached, model retrained.")
            else:
                st.success(f"Actual result added. Pending updates for retrain: {pending}/{predictor.retrain_batch_size}")
        except Exception as e:
            st.error(str(e))

    st.markdown("## Actual vs Predicted Table")
    if st.session_state["actual_pred_table"]:
        ap_df = pd.DataFrame(st.session_state["actual_pred_table"])
        st.dataframe(ap_df, width="stretch")
    else:
        st.info("Pehle Predict karein, phir same Period ka Actual add karein. Table yahin show hogi.")

    st.markdown("## Recent Actual Results")
    try:
        rec = predictor.load_data().copy()
        rec["_period_sort"] = pd.to_numeric(rec["Period"], errors="coerce")
        rec = rec.sort_values("_period_sort", ascending=False).drop(columns=["_period_sort"]).head(10).reset_index(drop=True)
        st.dataframe(rec, width="stretch")
        if "last_actual_added" in st.session_state:
            v = st.session_state["last_actual_added"]
            st.info(
                f"Last Added -> Period: {v['Period']}, Number: {v['Number']}, Big/Small: {v['Big/Small']}, Color: {v['Color']}"
            )
    except Exception as e:
        st.warning(f"Could not load recent actual results: {e}")


if __name__ == "__main__":
    main()
