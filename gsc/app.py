import io
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="GSC Brand vs Nonbrand Analyzer", layout="wide")

# ---------- Column mapping ----------
CANDIDATES = {
    "query": ["Query", "query", "자연 Google 검색어", "검색어"],
    "clicks": ["Clicks", "clicks", "자연 Google 검색 클릭수", "클릭수"],
    "impressions": ["Impressions", "impressions", "자연 Google 검색 노출수", "노출수"],
    "ctr": ["CTR", "ctr", "자연 Google 검색 클릭률", "클릭률"],
    "position": ["Position", "position", "자연 Google 검색 평균 게재순위", "평균 게재순위", "평균게재순위"],
}

def find_col(df, key):
    for c in CANDIDATES[key]:
        if c in df.columns:
            return c
    return None

def read_csv_safely(file) -> pd.DataFrame:
    content = file.getvalue()
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc, comment="#")
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", comment="#", engine="python")

# ---------- Brand rule ----------
DEFAULT_ADD = [
    "k t", "k. t. m", "k t mobile", "k-t", "k.t", "k.t.", "k/t",
    "k t engineering", "k-t event", "\"f와 g\" \"n, k, t\""
]
DEFAULT_REMOVE = [
    "www.ktmmobile",
    "kt 해외 로밍 데이터 무제한 요금제"
]

def build_brand_mask(q: pd.Series, add_list, remove_list) -> pd.Series:
    q = q.fillna("").astype(str)

    base = (
        q.str.contains("케이티", regex=False)
        | q.str.contains(r"(?:^|[^a-z0-9])kt(?:[^a-z0-9]|$)", case=False, regex=True)
        | q.str.contains(r"^kt[가-힣]", case=False, regex=True)
        | q.str.contains(r"ktm|kt\s*m|kt엠|케이티\s*엠|케이티엠|ktmmobile", case=False, regex=True)
        | q.str.startswith("엠모바일")
        | q.str.startswith("m모바일")
        | q.str.startswith("m 모바일")
        | q.str.lower().str.startswith("mmobile")
    )

    all_add = DEFAULT_ADD + [x.strip() for x in add_list if x.strip()]
    all_remove = DEFAULT_REMOVE + [x.strip() for x in remove_list if x.strip()]

    return (base | q.isin(all_add)) & (~q.isin(all_remove))

def summarize(df_std: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        impressions = g["impressions"].sum()
        clicks = g["clicks"].sum()
        top3_impr = g.loc[g["position"] <= 3, "impressions"].sum()
        return pd.Series({
            "키워드수": g["query"].nunique(),
            "노출수": int(impressions),
            "클릭수": int(clicks),
            "CTR": (clicks / impressions) if impressions else np.nan,
            "Top3 노출 비중": (top3_impr / impressions) if impressions else np.nan
        })

    out = df_std.groupby("brand_flag").apply(agg).reset_index()
    total = pd.DataFrame([{
        "brand_flag": "총합",
        "키워드수": df_std["query"].nunique(),
        "노출수": int(df_std["impressions"].sum()),
        "클릭수": int(df_std["clicks"].sum()),
        "CTR": (df_std["clicks"].sum() / df_std["impressions"].sum()) if df_std["impressions"].sum() else np.nan,
        "Top3 노출 비중": (df_std.loc[df_std["position"] <= 3, "impressions"].sum() / df_std["impressions"].sum()) if df_std["impressions"].sum() else np.nan
    }])
    return pd.concat([out, total], ignore_index=True)

# ---------- UI ----------
st.markdown("#### GSC 검색어 업로드 → 브랜드/일반 자동 분류 & 지표 산출")

uploaded = st.file_uploader("GSC 쿼리 CSV 업로드", type=["csv"])

with st.expander("브랜드 분류 예외(선택)"):
    add_text = st.text_area("강제 포함 query (줄바꿈)", value="")
    remove_text = st.text_area("강제 제외 query (줄바꿈)", value="")

if uploaded:
    df = read_csv_safely(uploaded)

    q_col = find_col(df, "query")
    c_col = find_col(df, "clicks")
    i_col = find_col(df, "impressions")
    ctr_col = find_col(df, "ctr")
    p_col = find_col(df, "position")

    missing = [k for k, col in [("query", q_col), ("clicks", c_col), ("impressions", i_col), ("position", p_col)] if col is None]
    if missing:
        st.error(f"필수 컬럼을 못 찾았어: {missing}. (파일이 GSC 쿼리 export인지 확인 필요)")
        st.stop()

    df_std = pd.DataFrame({
        "query": df[q_col].astype(str),
        "clicks": pd.to_numeric(df[c_col], errors="coerce").fillna(0),
        "impressions": pd.to_numeric(df[i_col], errors="coerce").fillna(0),
        "ctr": pd.to_numeric(df[ctr_col], errors="coerce") if ctr_col else np.nan,
        "position": pd.to_numeric(df[p_col], errors="coerce"),
    })
    df_std["clicks"] = df_std["clicks"].astype(int)
    df_std["impressions"] = df_std["impressions"].astype(int)

    add_list = add_text.splitlines()
    remove_list = remove_text.splitlines()

    brand_mask = build_brand_mask(df_std["query"], add_list, remove_list)
    df_std["brand_flag"] = np.where(brand_mask, "브랜드/준브랜드(kt 포함)", "일반(비브랜드)")

    summary = summarize(df_std)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("요약 지표")
        fmt = {"키워드수": "{:.0f}", "노출수": "{:.0f}", "클릭수": "{:.0f}", "CTR": "{:.2%}", "Top3 노출 비중": "{:.2%}"}
        st.dataframe(summary.style.format(fmt, na_rep="-"), use_container_width=True)
    with col2:
        st.subheader("샘플 raw")
        st.dataframe(df_std.head(30), use_container_width=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_std.to_excel(writer, index=False, sheet_name="raw")
        summary.to_excel(writer, index=False, sheet_name="summary")
        pd.DataFrame([{
            "note": "CTR은 클릭/노출로 재계산. Top3 노출 비중 = position<=3 노출합 / 전체노출합"
        }]).to_excel(writer, index=False, sheet_name="notes")

    st.download_button(
        label="엑셀 다운로드 (raw + summary)",
        data=output.getvalue(),
        file_name="gsc_brand_nonbrand_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

else:
    st.info("CSV 업로드하면 자동으로 결과가 나와.")
