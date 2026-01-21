# web.py
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib, pickle
import matplotlib
import matplotlib.pyplot as plt

# 兼容 numpy 旧别名
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int

import shap

# ============== 字体/中文显示 ==================
def setup_chinese_font():
    """设置中文字体（优先系统字体，其次 ./fonts 目录）"""
    try:
        import matplotlib.font_manager as fm
        chinese_fonts = [
            'WenQuanYi Zen Hei','WenQuanYi Micro Hei','SimHei','Microsoft YaHei',
            'PingFang SC','Hiragino Sans GB','Noto Sans CJK SC','Source Han Sans SC'
        ]
        available = [f.name for f in fm.fontManager.ttflist]
        for f in chinese_fonts:
            if f in available:
                matplotlib.rcParams['font.sans-serif'] = [f, 'DejaVu Sans', 'Arial']
                matplotlib.rcParams['font.family'] = 'sans-serif'
                return f

        # 尝试加载 ./fonts 下自带字体
        fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        candidates = [
            'NotoSansSC-Regular.otf','NotoSansCJKsc-Regular.otf',
            'SourceHanSansSC-Regular.otf','SimHei.ttf','MicrosoftYaHei.ttf'
        ]
        if os.path.isdir(fonts_dir):
            import matplotlib.font_manager as fm
            for fname in candidates:
                fpath = os.path.join(fonts_dir, fname)
                if os.path.exists(fpath):
                    fm.fontManager.addfont(fpath)
                    fam = fm.FontProperties(fname=fpath).get_name()
                    matplotlib.rcParams['font.sans-serif'] = [fam, 'DejaVu Sans', 'Arial']
                    matplotlib.rcParams['font.family'] = 'sans-serif'
                    return fam
    except Exception:
        pass

    # 兜底
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    return None

chinese_font = setup_chinese_font()
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = matplotlib.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============== 页面配置 ==================
st.set_page_config(
    page_title="GDM Gestational Diabetes Mellitus Risk Prediction Tool",
    page_icon="GDM",
    layout="wide"
)

# ============== Feature names & display labels ==================
MODEL_FEATURES = [
    "TC", "TG", "HDL-C", "LDL-C", "ChE",
    "Glucose", "SF", "Tg", "GGT", "PTA",
    "lnCu+10", "lnAs+10", "lnCd+10", "lnPb+10",
]

FEATURE_LABELS = {
    "TC": "TC (Total Cholesterol)",
    "TG": "TG (Triglycerides)",
    "HDL-C": "HDL-C (High-Density Lipoprotein Cholesterol)",
    "LDL-C": "LDL-C (Low-Density Lipoprotein Cholesterol)",
    "ChE": "ChE (Cholinesterase)",
    "Glucose": "Glu (Glucose)",
    "SF": "SF (Ferritin)",
    "Tg": "Tg (Thyroglobulin)",
    "GGT": "GGT (Gamma-Glutamyl Transferase)",
    "PTA": "PTA (Prothrombin Time Activity)",
    "lnCu+10": "Cu (ln(value)+10)",
    "lnAs+10": "As (ln(value)+10)",
    "lnCd+10": "Cd (ln(value)+10)",
    "lnPb+10": "Pb (ln(value)+10)",
}

VARIABLE_DESCRIPTIONS = {
    "TC": "Total Cholesterol",
    "TG": "Triglycerides",
    "HDL-C": "High-Density Lipoprotein Cholesterol",
    "LDL-C": "Low-Density Lipoprotein Cholesterol",
    "ChE": "Cholinesterase",
    "Glucose": "Glucose",
    "SF": "Ferritin",
    "Tg": "Thyroglobulin",
    "GGT": "Gamma-Glutamyl Transferase",
    "PTA": "Prothrombin Time Activity",
    "lnCu+10": "Cu value from report, model uses ln(Cu) + 10",
    "lnAs+10": "As value from report, model uses ln(As) + 10",
    "lnCd+10": "Cd value from report, model uses ln(Cd) + 10",
    "lnPb+10": "Pb value from report, model uses ln(Pb) + 10",
}

_model_root = os.path.dirname(__file__)
MODEL_PATH = os.path.join(_model_root, "XGBoost_model.pkl")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(_model_root, "XGBoost_model.pkl")
GDM_THRESHOLD = 0.30


# ============== 工具函数 ==================
def _clean_number(x):
    """把 '[3.3101046E-1]'、'3,210'、' 12. ' 等转成 float；失败返回 NaN"""
    if isinstance(x, str):
        s = x.strip().strip('[](){}').replace(',', '')
        try:
            return float(s)
        except Exception:
            return np.nan
    return x

def unwrap_model(model):
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        return model.named_steps["clf"]
    return model

@st.cache_resource
def load_model(model_path: str = MODEL_PATH):
    """加载 xgboost 模型，兼容旧版训练产物：补 use_label_encoder / gpu_id / n_gpus / predictor 等缺失属性"""
    try:
        try:
            model = joblib.load(model_path)
        except Exception:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

        base_model = unwrap_model(model)

        # 兼容补丁：老版本 XGBoost 训练的模型里常见的已废弃/迁移属性
        try:
            if hasattr(base_model, "__class__") and base_model.__class__.__name__.startswith("XGB"):
                # 这些属性的存在只为避免 get_params() getattr 报错；值不影响 1.7.6 推理
                defaults = {
                    "use_label_encoder": False,   # 1.x 时代参数，2.x 已废弃
                    "gpu_id": 0,                  # 老版本 GPU 选择；1.7.6 不再需要
                    "n_gpus": 1,                  # 有些旧代码保存过这个
                    "predictor": None,            # 旧参数：cpu_predictor/gpu_predictor
                    "tree_method": getattr(base_model, "tree_method", None),
                }
                for k, v in defaults.items():
                    if not hasattr(base_model, k):
                        setattr(base_model, k, v)
        except Exception:
            pass

        # 尝试获取特征名（优先 sklearn 风格，再退 Booster）
        model_feature_names = None
        for candidate in (model, base_model):
            try:
                if hasattr(candidate, 'feature_names_in_'):
                    model_feature_names = list(candidate.feature_names_in_)
                    break
            except Exception:
                pass
        if model_feature_names is None:
            try:
                booster = getattr(base_model, 'get_booster', lambda: None)()
                if booster is not None and hasattr(booster, 'feature_names'):
                    model_feature_names = list(booster.feature_names)
            except Exception:
                model_feature_names = None

        return model, model_feature_names
    except Exception as e:
        raise RuntimeError(f"无法加载模型: {e}")

def predict_proba_safe(model, X_df):
    """优先用 sklearn predict_proba；失败则补属性重试；仍失败则回退到 booster 直接预测概率"""
    # 第一次尝试
    try:
        return model.predict_proba(X_df)
    except AttributeError:
        # 再补一次容错属性（如果模型是从别处传来的）
        base_model = unwrap_model(model)
        for k, v in {"use_label_encoder": False, "gpu_id": 0, "n_gpus": 1, "predictor": None}.items():
            if not hasattr(base_model, k):
                setattr(base_model, k, v)
        return base_model.predict_proba(X_df)
    except Exception:
        # 回退：直接用 booster 预测（要求模型 objective 为二/多分类概率）
        import xgboost as xgb
        base_model = unwrap_model(model)
        booster = getattr(base_model, "get_booster", lambda: None)()
        if booster is None:
            raise
        dm = xgb.DMatrix(X_df.values, feature_names=list(X_df.columns))
        pred = booster.predict(dm, output_margin=False)
        # pred 形状：二分类通常 (n,), 多分类 (n, K)
        if isinstance(pred, np.ndarray):
            if pred.ndim == 1:  # 二分类概率（正类）
                proba_pos = pred.astype(float)
                return np.vstack([1 - proba_pos, proba_pos]).T
            elif pred.ndim == 2:
                return pred.astype(float)
        raise RuntimeError("Booster fallback failed: unknown output shape")

# ============== 主逻辑 ==================
def main():
    st.sidebar.title("GDM Gestational Diabetes Mellitus Risk Prediction Tool")
    st.sidebar.markdown("""
    ### About
    This calculator uses an XGBoost model to predict GDM risk.

    **Outputs:**
    - Predicted probability of GDM vs. non-GDM
    - Optional SHAP-based explanation
    """)
    with st.sidebar.expander("Variable description"):
        for f in MODEL_FEATURES:
            label = FEATURE_LABELS.get(f, f)
            desc = VARIABLE_DESCRIPTIONS.get(f, "")
            st.markdown(f"**{label}**: {desc}")

    st.title("GDM Gestational Diabetes Mellitus Risk Prediction Tool")

    try:
        model, model_feature_names = load_model(MODEL_PATH)
        st.sidebar.success(f"Model loaded: {os.path.basename(MODEL_PATH)}")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        return

    st.header("Patient information")
    with st.form("input_form"):
        st.subheader("Laboratory results")
        c1, c2, c3 = st.columns(3)
        with c1:
            tc = st.number_input("TC (Total Cholesterol)", value=4.5, step=0.1, min_value=0.0)
            tg = st.number_input("TG (Triglycerides)", value=1.5, step=0.1, min_value=0.0)
            hdl = st.number_input("HDL-C", value=1.2, step=0.1, min_value=0.0)
            ldl = st.number_input("LDL-C", value=2.5, step=0.1, min_value=0.0)
        with c2:
            che = st.number_input("ChE (Cholinesterase)", value=7.0, step=0.1, min_value=0.0)
            glucose = st.number_input("Glu (Glucose)", value=4.5, step=0.1, min_value=0.0)
            sf = st.number_input("SF (Ferritin)", value=30.0, step=1.0, min_value=0.0)
            tg_thyro = st.number_input("Tg (Thyroglobulin)", value=5.0, step=0.1, min_value=0.0)
        with c3:
            ggt = st.number_input("GGT (Gamma-Glutamyl Transferase)", value=20.0, step=1.0, min_value=0.0)
            pta = st.number_input("PTA (Prothrombin Time Activity)", value=90.0, step=1.0, min_value=0.0)

        st.subheader("Metals (raw values from report)")
        c9, c10 = st.columns(2)
        with c9:
            cu = st.number_input("Cu (raw value, ln(Cu)+10)", value=1.0, step=0.1, min_value=0.0001)
            arsenic = st.number_input("As (raw value, ln(As)+10)", value=1.0, step=0.1, min_value=0.0001)
        with c10:
            cd = st.number_input("Cd (raw value, ln(Cd)+10)", value=1.0, step=0.1, min_value=0.0001)
            pb = st.number_input("Pb (raw value, ln(Pb)+10)", value=1.0, step=0.1, min_value=0.0001)

        submitted = st.form_submit_button("Run prediction")
    if submitted:
        if any(v <= 0 for v in (cu, arsenic, cd, pb)):
            st.error("Cu/As/Cd/Pb must be positive numbers to compute ln(value)+10.")
            return

        ln_cu = float(np.log(cu) + 10)
        ln_as = float(np.log(arsenic) + 10)
        ln_cd = float(np.log(cd) + 10)
        ln_pb = float(np.log(pb) + 10)

        user_inputs = {
            "TC": tc,
            "TG": tg,
            "HDL-C": hdl,
            "LDL-C": ldl,
            "ChE": che,
            "Glucose": glucose,
            "SF": sf,
            "Tg": tg_thyro,
            "GGT": ggt,
            "PTA": pta,
            "lnCu+10": ln_cu,
            "lnAs+10": ln_as,
            "lnCd+10": ln_cd,
            "lnPb+10": ln_pb,
        }

        features = model_feature_names or MODEL_FEATURES
        missing_features = [f for f in features if f not in user_inputs]
        if missing_features:
            st.error(f"Missing model features in UI: {missing_features}")
            with st.expander("Debug: compare model feature names and UI keys"):
                st.write("Model feature names:", features)
                st.write("UI input keys:", list(user_inputs.keys()))
            return

        input_df = pd.DataFrame([[user_inputs[c] for c in features]], columns=features)

        input_df = input_df.applymap(_clean_number)
        for c in input_df.columns:
            input_df[c] = pd.to_numeric(input_df[c], errors='coerce')
        if input_df.isnull().any().any():
            st.error("There are missing or unparsable input values. Please check all fields.")
            with st.expander("Debug: current input DataFrame"):
                st.write(input_df)
            return

        try:
            proba = predict_proba_safe(model, input_df)[0]
            if len(proba) != 2:
                raise ValueError("Unexpected probability shape")
            no_gdm_prob = float(proba[0])
            gdm_prob = float(proba[1])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        st.header("GDM risk prediction result")
        a, b = st.columns(2)
        with a:
            st.subheader("Probability of non-GDM")
            st.progress(no_gdm_prob)
            st.write(f"{no_gdm_prob:.2%}")
        with b:
            st.subheader("Probability of GDM")
            st.progress(gdm_prob)
            st.write(f"{gdm_prob:.2%}")

        st.subheader("Prediction label")
        if gdm_prob >= GDM_THRESHOLD:
            st.error(f"High risk (>= {GDM_THRESHOLD:.2f})")
        else:
            st.success(f"Low risk (< {GDM_THRESHOLD:.2f})")

        st.write("---")
        st.subheader("Model explanation (SHAP)")
        try:
            model_for_shap = unwrap_model(model)
            try:
                explainer = shap.Explainer(model_for_shap)
                sv = explainer(input_df)
                shap_value = np.array(sv.values[0])
                expected_value = sv.base_values[0] if np.ndim(sv.base_values) else sv.base_values
            except Exception:
                explainer = shap.TreeExplainer(model_for_shap)
                shap_values = explainer.shap_values(input_df)
                if isinstance(shap_values, list):
                    shap_value = np.array(shap_values[1][0])
                    ev = explainer.expected_value
                    expected_value = ev[1] if isinstance(ev, (list, np.ndarray)) else ev
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    shap_value = shap_values[0, :, 1]
                    ev = explainer.expected_value
                    expected_value = ev[1] if isinstance(ev, (list, np.ndarray)) else ev
                else:
                    shap_value = np.array(shap_values[0])
                    expected_value = explainer.expected_value

            current_features = list(input_df.columns)
            st.subheader("SHAP waterfall plot")
            fig_waterfall = plt.figure(figsize=(12, 8))
            display_data = input_df.iloc[0].copy()

            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_value,
                    base_values=expected_value,
                    data=display_data.values,
                    feature_names=[FEATURE_LABELS.get(f, f) for f in current_features],
                ),
                max_display=len(current_features),
                show=False,
            )

            for ax in fig_waterfall.get_axes():
                for text in ax.texts:
                    s = text.get_text()
                    if '?' in s:
                        text.set_text(s.replace('?', '-'))
                    if chinese_font:
                        text.set_fontfamily(chinese_font)
                for label in ax.get_yticklabels() + ax.get_xticklabels():
                    t = label.get_text()
                    if '?' in t:
                        label.set_text(t.replace('?', '-'))
                    if chinese_font:
                        label.set_fontfamily(chinese_font)
                if chinese_font:
                    ax.set_xlabel(ax.get_xlabel(), fontfamily=chinese_font)
                    ax.set_ylabel(ax.get_ylabel(), fontfamily=chinese_font)
                    ax.set_title(ax.get_title(), fontfamily=chinese_font)

            plt.tight_layout()
            st.pyplot(fig_waterfall)
            plt.close(fig_waterfall)

            st.subheader("SHAP force plot")
            try:
                import streamlit.components.v1 as components
                force_plot = shap.force_plot(
                    expected_value,
                    shap_value,
                    display_data,
                    feature_names=[FEATURE_LABELS.get(f, f) for f in current_features],
                )
                shap_html = f"""
                <head>{shap.getjs()}</head>
                <body><div class='force-plot-container'>{force_plot.html()}</div></body>
                """
                components.html(shap_html, height=400, scrolling=False)
            except Exception as e:
                st.warning(f"Failed to generate SHAP force plot: {e}")
        except Exception as e:
            st.error(f"Failed to generate SHAP explanation: {e}")

    st.write("---")
    st.caption("GDM Risk Prediction Tool (XGBoost)")
if __name__ == "__main__":
    main()
