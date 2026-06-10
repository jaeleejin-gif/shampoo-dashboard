import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ─────────────────────────────────────────────────────────────
# 하드코딩 상수
# ─────────────────────────────────────────────────────────────
BASF_STRUCTURAL = {
    'Texapon SB 3 KC':       {'C_0':69.3,'C_neg1':0,'C_neg2':0,'C_neg3':0,'C_neg4':0,'C_neg5':0,'C_neg6':0,'C_neg7':0,'C_neg8':0,'C_pos1':0,'C_pos2':30.7,'C_pos3':0,'C_pos4':0,'C_pos5':0,'C_pos6':0,'TOTAL':100,'CS(=O)(=O)[O-]':1,'CC(=O)[O-]':1,'[H]C(C)(OC)OC':0,'C[N+](C)(C)C':0},
    'Plantapon ACG 50':      {'C_0':40.2,'C_neg1':0,'C_neg2':5.4,'C_neg3':0,'C_neg4':11.5,'C_neg5':0,'C_neg6':0,'C_neg7':0,'C_neg8':0,'C_pos1':0,'C_pos2':24.4,'C_pos3':0,'C_pos4':17.9,'C_pos5':0.6,'C_pos6':0,'TOTAL':100,'CS(=O)(=O)[O-]':0,'CC(=O)[O-]':2,'[H]C(C)(OC)OC':0,'C[N+](C)(C)C':0},
    'Plantapon LC 7':        {'C_0':100,'C_neg1':0,'C_neg2':0,'C_neg3':0,'C_neg4':0,'C_neg5':0,'C_neg6':0,'C_neg7':0,'C_neg8':0,'C_pos1':0,'C_pos2':0,'C_pos3':0,'C_pos4':0,'C_pos5':0,'C_pos6':0,'TOTAL':100,'CS(=O)(=O)[O-]':0,'CC(=O)[O-]':0,'[H]C(C)(OC)OC':0,'C[N+](C)(C)C':0},
    'Plantacare 818':        {'C_0':24.8,'C_neg1':0,'C_neg2':29,'C_neg3':0,'C_neg4':15,'C_neg5':0,'C_neg6':24.1,'C_neg7':0,'C_neg8':0,'C_pos1':0,'C_pos2':0,'C_pos3':0,'C_pos4':2.9,'C_pos5':0,'C_pos6':3.7,'TOTAL':99.5,'CS(=O)(=O)[O-]':0,'CC(=O)[O-]':0,'[H]C(C)(OC)OC':1,'C[N+](C)(C)C':0},
    'Plantacare 2000':       {'C_0':32.1,'C_neg1':0,'C_neg2':0,'C_neg3':11,'C_neg4':31,'C_neg5':5.8,'C_neg6':0,'C_neg7':0,'C_neg8':0,'C_pos1':0,'C_pos2':17.1,'C_pos3':0,'C_pos4':0,'C_pos5':0,'C_pos6':2.1,'TOTAL':99.1,'CS(=O)(=O)[O-]':0,'CC(=O)[O-]':0,'[H]C(C)(OC)OC':1,'C[N+](C)(C)C':0},
    'Dehyton MC':            {'C_0':36.2,'C_neg1':0,'C_neg2':8.5,'C_neg3':0,'C_neg4':9.2,'C_neg5':0,'C_neg6':0,'C_neg7':0,'C_neg8':0,'C_pos1':0,'C_pos2':20.7,'C_pos3':0,'C_pos4':12.7,'C_pos5':0,'C_pos6':12.6,'TOTAL':100,'CS(=O)(=O)[O-]':0,'CC(=O)[O-]':1,'[H]C(C)(OC)OC':0,'C[N+](C)(C)C':0},
    'Dehyton PK 45':         {'C_0':38.3,'C_neg1':0,'C_neg2':9,'C_neg3':0,'C_neg4':6.2,'C_neg5':0,'C_neg6':0,'C_neg7':0,'C_neg8':0,'C_pos1':0,'C_pos2':21.4,'C_pos3':0,'C_pos4':7.4,'C_pos5':0,'C_pos6':17.7,'TOTAL':99.9,'CS(=O)(=O)[O-]':0,'CC(=O)[O-]':1,'[H]C(C)(OC)OC':0,'C[N+](C)(C)C':1},
    'Dehyton ML':            {'C_0':97.5,'C_neg1':0,'C_neg2':2.5,'C_neg3':0,'C_neg4':0,'C_neg5':0,'C_neg6':0,'C_neg7':0,'C_neg8':0,'C_pos1':0,'C_pos2':0,'C_pos3':0,'C_pos4':0,'C_pos5':0,'C_pos6':0,'TOTAL':100,'CS(=O)(=O)[O-]':0,'CC(=O)[O-]':1,'[H]C(C)(OC)OC':0,'C[N+](C)(C)C':0},
    'Dehyton AB 30':         {'C_0':61.2,'C_neg1':0,'C_neg2':3.2,'C_neg3':0,'C_neg4':0,'C_neg5':0,'C_neg6':0,'C_neg7':0,'C_neg8':0,'C_pos1':0,'C_pos2':35.7,'C_pos3':0,'C_pos4':0,'C_pos5':0,'C_pos6':0,'TOTAL':100,'CS(=O)(=O)[O-]':0,'CC(=O)[O-]':1,'[H]C(C)(OC)OC':0,'C[N+](C)(C)C':1},
    'Plantapon Amino SCG-L': {'C_0':8.4,'C_neg1':0,'C_neg2':26.5,'C_neg3':0,'C_neg4':46,'C_neg5':0,'C_neg6':8.7,'C_neg7':0,'C_neg8':10.5,'C_pos1':0,'C_pos2':0,'C_pos3':0,'C_pos4':0,'C_pos5':0,'C_pos6':0,'TOTAL':100,'CS(=O)(=O)[O-]':0,'CC(=O)[O-]':2,'[H]C(C)(OC)OC':0,'C[N+](C)(C)C':0},
    'Plantapon Amino KG-L':  {'C_0':51,'C_neg1':0,'C_neg2':12.1,'C_neg3':0,'C_neg4':11.5,'C_neg5':0,'C_neg6':0,'C_neg7':0,'C_neg8':0,'C_pos1':0,'C_pos2':25.3,'C_pos3':0,'C_pos4':0,'C_pos5':0,'C_pos6':0,'TOTAL':100,'CS(=O)(=O)[O-]':0,'CC(=O)[O-]':1,'[H]C(C)(OC)OC':0,'C[N+](C)(C)C':0},
    'Dehyquart A-CA':        {'C_0':100,'C_neg1':0,'C_neg2':0,'C_neg3':0,'C_neg4':0,'C_neg5':0,'C_neg6':0,'C_neg7':0,'C_neg8':0,'C_pos1':0,'C_pos2':0,'C_pos3':0,'C_pos4':0,'C_pos5':0,'C_pos6':0,'TOTAL':100,'CS(=O)(=O)[O-]':0,'CC(=O)[O-]':0,'[H]C(C)(OC)OC':0,'C[N+](C)(C)C':1},
}

SURFACTANTS = list(BASF_STRUCTURAL.keys())
POLYMERS    = ['Luviquat Excellence', 'Salcare Super 7', 'Dehyquart CC6', 'Dehyquart CC7 Benz']
THICKENERS  = ['Arlypon TT', 'Arlypon F']

MODEL_FEATURES_RAW = [
    'Arlypon TT', 'Arlypon F', 'Luviquat Excellence', 'Dehyquart CC6',
    'Dehyquart CC7 Benz', 'Salcare Super 7',
    'C[N+](C)(C)C_derived', 'CS(=O)(=O)[O-]_derived', 'CC(=O)[O-]_derived',
    '[H]C(C)(OC)OC_derived', 'C_0_Ratio', 'C_neg_Total_Ratio', 'C_pos_Total_Ratio',
]

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Combined_data.csv')


# ─────────────────────────────────────────────────────────────
# 데이터 로딩
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# ─────────────────────────────────────────────────────────────
# 가이드라인 계산 (성공 제형 기준)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def compute_guidelines():
    df = load_data()

    stab_mask  = df['Stability_Test'].isin([True, 'True', 'true', 1])
    turb_mask  = df['Turbidity_NTU'].notna() & (df['Turbidity_NTU'] <= 100)
    visc_mask  = df['Viscosity'].isin(['MEDIUM', 'HIGH'])
    df_ok = df[stab_mask & turb_mask & visc_mask].copy()

    ingredient_ranges = {}
    for ing in SURFACTANTS + POLYMERS + THICKENERS:
        if ing not in df_ok.columns:
            continue
        vals = df_ok[df_ok[ing] > 0][ing].dropna()
        if len(vals) > 0:
            ingredient_ranges[ing] = (
                round(float(vals.min()), 2),
                round(float(vals.max()), 2),
                round(float(vals.mean()), 2),
            )

    def get_combo(row):
        return tuple(sorted([s for s in SURFACTANTS if row.get(s, 0) > 0]))

    df_ok['_combo'] = df_ok.apply(get_combo, axis=1)
    df_2 = df_ok[df_ok['_combo'].apply(len) == 2].copy()

    combo_guidelines = {}
    for combo, grp in df_2.groupby('_combo'):
        s1, s2 = combo
        m1, m2 = grp[s1].mean(), grp[s2].mean()
        ratio_str = f"1:{m2/m1:.2f}" if m1 > 0 else "N/A"

        surf_range = {
            s: (round(float(grp[s].min()), 2), round(float(grp[s].max()), 2))
            for s in combo
        }

        polymer_data = {}
        for p in POLYMERS:
            if p not in grp.columns:
                continue
            v = grp[grp[p] > 0][p].dropna()
            if len(v) > 0:
                polymer_data[p] = (round(float(v.min()), 2), round(float(v.max()), 2))

        thickener_data = {}
        for t in THICKENERS:
            if t not in grp.columns:
                continue
            v = grp[grp[t] > 0][t].dropna()
            if len(v) > 0:
                thickener_data[t] = (round(float(v.min()), 2), round(float(v.max()), 2))

        combo_guidelines[combo] = {
            'count':      len(grp),
            'ratio':      ratio_str,
            'surf_range': surf_range,
            'polymer':    polymer_data,
            'thickener':  thickener_data,
        }

    return ingredient_ranges, combo_guidelines


# ─────────────────────────────────────────────────────────────
# 모델 학습
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_models():
    df = load_data()
    stab = df['Stability_Test'].isin([True, 'True', 'true', 1])

    def _add_ratio_cols(d, scale100):
        c_neg_cols = sorted([c for c in d.columns if re.match(r'^C_neg\d+_derived$', c)])
        c_pos_cols = sorted([c for c in d.columns if re.match(r'^C_pos\d+_derived$', c)])
        d['C_neg_Total_derived'] = d[c_neg_cols].sum(axis=1) if c_neg_cols else 0.0
        d['C_pos_Total_derived'] = d[c_pos_cols].sum(axis=1) if c_pos_cols else 0.0
        tot = d['TOTAL_derived'] if 'TOTAL_derived' in d.columns else pd.Series(0.0, index=d.index)
        if scale100:
            d['C_0_Ratio']         = (d['C_0_derived'] / (tot + 1e-5)) * 100
            d['C_neg_Total_Ratio'] = (d['C_neg_Total_derived'] / (tot + 1e-5)) * 100
            d['C_pos_Total_Ratio'] = (d['C_pos_Total_derived'] / (tot + 1e-5)) * 100
        else:
            tot_safe = tot.replace(0, np.nan)
            d['C_0_Ratio']         = d['C_0_derived'] / tot_safe
            d['C_neg_Total_Ratio'] = d['C_neg_Total_derived'] / tot_safe
            d['C_pos_Total_Ratio'] = d['C_pos_Total_derived'] / tot_safe
        return d

    df_v = df[stab & df['Viscosity'].notna()].copy()
    df_v = _add_ratio_cols(df_v, scale100=True)
    X_v = df_v[MODEL_FEATURES_RAW].fillna(0).copy()
    X_v.columns = X_v.columns.str.replace(r'[\[\]<>,]', '_', regex=True)
    y_v = df_v['Viscosity'].isin(['LOW', 'VERY-LOW']).astype(int)

    vm = XGBClassifier(learning_rate=0.1, max_depth=10, n_estimators=100,
                       subsample=0.8, random_state=42, eval_metric='logloss')
    vm.fit(X_v, y_v, sample_weight=compute_sample_weight('balanced', y_v))
    visc_feats = list(X_v.columns)

    df_t = df[df['Turbidity_NTU'].notna()].copy()
    df_t = _add_ratio_cols(df_t, scale100=False)
    X_t = df_t[MODEL_FEATURES_RAW].fillna(0).copy()
    X_t.columns = pd.Index([re.sub(r'[^a-zA-Z0-9_]', '_', c) for c in X_t.columns])
    y_t = (df_t['Turbidity_NTU'] > 100).astype(int)

    tm = LGBMClassifier(n_estimators=387, learning_rate=0.10208424050050095,
                        max_depth=10, num_leaves=51, class_weight='balanced',
                        random_state=42, verbose=-1)
    tm.fit(X_t, y_t)
    turb_feats = list(X_t.columns)

    return vm, visc_feats, tm, turb_feats


# ─────────────────────────────────────────────────────────────
# 예측
# ─────────────────────────────────────────────────────────────
def make_prediction(surf1, conc1, surf2, conc2, polymer, p_conc, thickener, t_conc,
                    vm, visc_feats, tm, turb_feats):
    c0 = c_neg = c_pos = tot = c_np = c_su = c_ac = c_gl = 0.0

    for surf, conc in [(surf1, conc1), (surf2, conc2)]:
        if surf and conc > 0:
            s = BASF_STRUCTURAL[surf]
            c0    += s['C_0'] * conc
            c_neg += sum(s.get(f'C_neg{i}', 0) for i in range(1, 9)) * conc
            c_pos += sum(s.get(f'C_pos{i}', 0) for i in range(1, 7)) * conc
            tot   += s['TOTAL'] * conc
            c_np  += s['C[N+](C)(C)C'] * conc
            c_su  += s['CS(=O)(=O)[O-]'] * conc
            c_ac  += s['CC(=O)[O-]'] * conc
            c_gl  += s['[H]C(C)(OC)OC'] * conc

    direct = {k: 0.0 for k in POLYMERS + THICKENERS}
    if polymer:   direct[polymer]   = p_conc
    if thickener: direct[thickener] = t_conc

    base = {
        'Arlypon TT':             direct['Arlypon TT'],
        'Arlypon F':              direct['Arlypon F'],
        'Luviquat Excellence':    direct['Luviquat Excellence'],
        'Dehyquart CC6':          direct['Dehyquart CC6'],
        'Dehyquart CC7 Benz':     direct['Dehyquart CC7 Benz'],
        'Salcare Super 7':        direct['Salcare Super 7'],
        'C[N+](C)(C)C_derived':   c_np,
        'CS(=O)(=O)[O-]_derived': c_su,
        'CC(=O)[O-]_derived':     c_ac,
        '[H]C(C)(OC)OC_derived':  c_gl,
    }

    eps = 1e-5
    v_row = dict(base)
    v_row['C_0_Ratio']         = (c0   / (tot + eps)) * 100
    v_row['C_neg_Total_Ratio'] = (c_neg / (tot + eps)) * 100
    v_row['C_pos_Total_Ratio'] = (c_pos / (tot + eps)) * 100
    df_v = pd.DataFrame([v_row])[MODEL_FEATURES_RAW]
    df_v.columns = df_v.columns.str.replace(r'[\[\]<>,]', '_', regex=True)
    df_v = df_v[visc_feats]
    v_pred = int(vm.predict(df_v)[0])

    denom = tot if tot > 0 else np.nan
    t_row = dict(base)
    t_row['C_0_Ratio']         = c0   / denom if denom else 0.0
    t_row['C_neg_Total_Ratio'] = c_neg / denom if denom else 0.0
    t_row['C_pos_Total_Ratio'] = c_pos / denom if denom else 0.0
    df_t = pd.DataFrame([t_row])[MODEL_FEATURES_RAW]
    df_t.columns = pd.Index([re.sub(r'[^a-zA-Z0-9_]', '_', c) for c in df_t.columns])
    df_t = df_t[turb_feats]
    t_pred = int(tm.predict(df_t)[0])

    return v_pred, t_pred


# ─────────────────────────────────────────────────────────────
# 가이드라인 렌더링  (민트 헤더)
# ─────────────────────────────────────────────────────────────
def render_guideline(surf1, surf2, ingredient_ranges, combo_guidelines):
    st.markdown(
        "<div style='background:#e0f2f1;border:1px solid #80cbc4;border-radius:8px;padding:16px'>"
        "<b style='font-size:1.05em'>배합비 가이드라인</b>"
        "<span style='color:#555;font-size:0.85em;margin-left:8px'>"
        "안정성 통과 + 점도 보통/높음 + 탁도 ≤100 NTU 기준 성공 제형 분석</span>",
        unsafe_allow_html=True,
    )

    if not surf1:
        st.markdown(
            "<p style='color:#888;margin:8px 0 4px 0'>계면활성제 1을 선택하면 추천 조합이 표시됩니다.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    matching = {k: v for k, v in combo_guidelines.items() if surf1 in k}
    if not matching:
        st.markdown(
            f"<p style='color:#888;margin:8px 0 4px 0'>{surf1}를 포함한 성공 제형 조합 데이터가 없습니다.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    selected_combo = tuple(sorted([surf1, surf2])) if surf2 else None

    rows = []
    for combo, info in sorted(matching.items(), key=lambda x: -x[1]['count']):
        partner = [s for s in combo if s != surf1][0]
        is_sel  = selected_combo is not None and combo == selected_combo

        s1, s2 = combo
        sr = info['surf_range']
        surf_txt = f"{s1}: {sr[s1][0]}~{sr[s1][1]}% &nbsp;|&nbsp; {s2}: {sr[s2][0]}~{sr[s2][1]}%"

        pol_parts = [f"{p}: {lo}~{hi}%" for p, (lo, hi) in info['polymer'].items()]
        pol_txt = " | ".join(pol_parts) if pol_parts else "—"

        thi_parts = [f"{t}: {lo}~{hi}%" for t, (lo, hi) in info['thickener'].items()]
        thi_txt = " | ".join(thi_parts) if thi_parts else "—"

        bg     = "#fffbe6" if is_sel else "white"
        border = "2px solid #f5a623" if is_sel else "1px solid #e0e0e0"
        star   = "★ " if is_sel else ""

        rows.append(
            f"<div style='background:{bg};border:{border};border-radius:6px;"
            f"padding:10px 14px;margin:6px 0'>"
            f"<b>{star}{partner}</b>"
            f"<span style='color:#666;font-size:0.85em;margin-left:8px'>"
            f"성공 {info['count']}건 &nbsp;·&nbsp; 비율 {info['ratio']}</span><br>"
            f"<span style='color:#333;font-size:0.82em'>계면활성제: {surf_txt}</span><br>"
            f"<span style='color:#333;font-size:0.82em'>폴리머: {pol_txt}</span><br>"
            f"<span style='color:#333;font-size:0.82em'>증점제: {thi_txt}</span>"
            f"</div>"
        )

    st.markdown("".join(rows) + "</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 앱 시작
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="샴푸 배합비 설계 대시보드", layout="wide")

st.markdown(
    "<h2 style='margin-bottom:4px'>샴푸 배합비 설계 대시보드</h2>",
    unsafe_allow_html=True,
)

with st.spinner("데이터 로딩 및 모델 학습 중…"):
    ingredient_ranges, combo_guidelines = compute_guidelines()
    vm, visc_feats, tm, turb_feats = train_models()


# ─────────────────────────────────────────────────────────────
# 입력 섹션  (연초록 헤더)
# ─────────────────────────────────────────────────────────────
st.markdown(
    "<div style='background:#e8f5e9;border-radius:8px;padding:10px 16px;margin-bottom:12px'>"
    "<b>성분 배합비 입력란</b></div>",
    unsafe_allow_html=True,
)

# ── 행 1: 계면활성제 1 | 계면활성제 2 ──
st.markdown("**계면활성제**")
s_col1, s_col2 = st.columns(2)

with s_col1:
    surf1_opts = [None] + SURFACTANTS
    surf1 = st.selectbox(
        "계면활성제 1",
        surf1_opts,
        format_func=lambda x: "— 선택 —" if x is None else x,
        key="surf1",
    )
    if surf1 and surf1 in ingredient_ranges:
        lo, hi, mu = ingredient_ranges[surf1]
        st.markdown(
            f"<span style='color:#333;font-size:0.82em'>추천 범위 {lo}~{hi}%&nbsp;&nbsp;(평균 {mu}%)</span>",
            unsafe_allow_html=True,
        )
    conc1 = st.number_input("투입량 (%)", min_value=0.0, max_value=50.0,
                             value=10.0, step=0.5, key="conc1")

with s_col2:
    surf2_opts = ["없음"] + [s for s in SURFACTANTS if s != surf1]
    surf2_sel  = st.selectbox("계면활성제 2", surf2_opts, key="surf2")
    surf2 = None if surf2_sel == "없음" else surf2_sel
    if surf2 and surf2 in ingredient_ranges:
        lo, hi, mu = ingredient_ranges[surf2]
        st.markdown(
            f"<span style='color:#333;font-size:0.82em'>추천 범위 {lo}~{hi}%&nbsp;&nbsp;(평균 {mu}%)</span>",
            unsafe_allow_html=True,
        )
    conc2 = st.number_input("투입량 (%)", min_value=0.0, max_value=50.0,
                             value=10.0, step=0.5, key="conc2",
                             disabled=(surf2 is None))

# ── 행 2: 폴리머 | 증점제 ──
st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
p_col, t_col = st.columns(2)

with p_col:
    st.markdown("**폴리머**")
    polymer_sel = st.selectbox("폴리머", ["없음"] + POLYMERS, key="polymer")
    polymer = None if polymer_sel == "없음" else polymer_sel
    if polymer and polymer in ingredient_ranges:
        lo, hi, mu = ingredient_ranges[polymer]
        st.markdown(
            f"<span style='color:#333;font-size:0.82em'>추천 범위 {lo}~{hi}%&nbsp;&nbsp;(평균 {mu}%)</span>",
            unsafe_allow_html=True,
        )
    p_conc = st.number_input("투입량 (%)", min_value=0.0, max_value=20.0,
                              value=2.0, step=0.1, key="pconc",
                              disabled=(polymer is None))

with t_col:
    st.markdown("**증점제**")
    thick_sel = st.selectbox("증점제", ["없음"] + THICKENERS, key="thickener")
    thickener = None if thick_sel == "없음" else thick_sel
    if thickener and thickener in ingredient_ranges:
        lo, hi, mu = ingredient_ranges[thickener]
        st.markdown(
            f"<span style='color:#333;font-size:0.82em'>추천 범위 {lo}~{hi}%&nbsp;&nbsp;(평균 {mu}%)</span>",
            unsafe_allow_html=True,
        )
    t_conc = st.number_input("투입량 (%)", min_value=0.0, max_value=20.0,
                              value=2.0, step=0.1, key="tconc",
                              disabled=(thickener is None))

# ─────────────────────────────────────────────────────────────
# 가이드라인 박스 (reactive)
# ─────────────────────────────────────────────────────────────
st.markdown("<div style='margin:16px 0 8px 0'></div>", unsafe_allow_html=True)
render_guideline(surf1, surf2, ingredient_ranges, combo_guidelines)

# ─────────────────────────────────────────────────────────────
# 예측 버튼  →  "샴푸 예측"
# ─────────────────────────────────────────────────────────────
st.markdown("<div style='margin:20px 0 8px 0'></div>", unsafe_allow_html=True)
st.markdown("""
<style>
div[data-testid="stButton"] > button[kind="primary"] {
    font-size: 1.3em;
    padding: 0.65em 0;
}
</style>
""", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1.5, 1, 1.5])
with btn_col:
    predict_clicked = st.button("샴푸 예측", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 예측 결과  →  "샴푸 예측 결과"
# ─────────────────────────────────────────────────────────────
if predict_clicked:
    if not surf1:
        st.warning("계면활성제 1을 선택해 주세요.")
    elif conc1 <= 0:
        st.warning("계면활성제 1의 투입량을 입력해 주세요.")
    else:
        s2  = surf2     if surf2     else None
        c2  = conc2     if surf2     else 0.0
        pol = polymer   if polymer   else None
        pc  = p_conc    if polymer   else 0.0
        thi = thickener if thickener else None
        tc  = t_conc    if thickener else 0.0

        with st.spinner("예측 중…"):
            v_pred, t_pred = make_prediction(
                surf1, conc1, s2, c2, pol, pc, thi, tc,
                vm, visc_feats, tm, turb_feats,
            )

        st.markdown(
            "<div style='background:#e3f2fd;border-radius:8px;padding:10px 16px;margin:12px 0 8px 0'>"
            "<b>샴푸 예측 결과</b></div>",
            unsafe_allow_html=True,
        )

        rc1, rc2 = st.columns(2)

        # 점도 등급 (한국어)
        if v_pred == 0:
            visc_label  = "보통 / 높음"
            visc_color  = "#e8f5e9"
            visc_border = "#66bb6a"
            visc_icon   = "✅"
            visc_note   = "적정 점도 범위"
        else:
            visc_label  = "매우 낮음 / 낮음"
            visc_color  = "#fff3e0"
            visc_border = "#ffa726"
            visc_icon   = "⚠️"
            visc_note   = "점도 부족 가능성"

        card_height = "180px"

        with rc1:
            st.markdown(
                f"<div style='background:{visc_color};border:2px solid {visc_border};"
                f"border-radius:10px;padding:28px;text-align:center;"
                f"height:{card_height};display:flex;flex-direction:column;"
                f"justify-content:center;align-items:center'>"
                f"<div style='font-size:1em;color:#555;margin-bottom:8px'>점도 등급</div>"
                f"<div style='font-size:1.8em;font-weight:bold'>{visc_icon} {visc_label}</div>"
                f"<div style='font-size:0.85em;color:#777;margin-top:8px'>{visc_note}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # 탁도 등급
        if t_pred == 0:
            turb_label  = "맑음"
            turb_sub    = "≤100 NTU"
            turb_color  = "#e8f5e9"
            turb_border = "#66bb6a"
            turb_icon   = "✅"
            turb_note   = "투명도 양호"
        else:
            turb_label  = "탁함"
            turb_sub    = ">100 NTU"
            turb_color  = "#fce4ec"
            turb_border = "#ef5350"
            turb_icon   = "❌"
            turb_note   = "탁도 초과 가능성"

        with rc2:
            st.markdown(
                f"<div style='background:{turb_color};border:2px solid {turb_border};"
                f"border-radius:10px;padding:28px;text-align:center;"
                f"height:{card_height};display:flex;flex-direction:column;"
                f"justify-content:center;align-items:center'>"
                f"<div style='font-size:1em;color:#555;margin-bottom:8px'>탁도 등급</div>"
                f"<div style='font-size:1.8em;font-weight:bold'>{turb_icon} {turb_label}</div>"
                f"<div style='font-size:0.85em;color:#777;margin-top:6px'>{turb_sub}</div>"
                f"<div style='font-size:0.85em;color:#777;margin-top:2px'>{turb_note}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # 모델 정보 — 우측 하단
        st.markdown(
            "<div style='font-size:0.78em;color:#aaa;margin-top:8px;text-align:right'>"
            "점도 모델: XGBoost (정확도 87.2%) &nbsp;|&nbsp; 탁도 모델: LightGBM (F1 0.797)"
            "</div>",
            unsafe_allow_html=True,
        )
