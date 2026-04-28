import io
import json
import re
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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

def read_csv_safely(content: bytes) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc, comment="#")
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", comment="#", engine="python")

def extract_dates(content: bytes) -> str:
    """# 시작일 / 종료일 주석에서 날짜 문자열 추출"""
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            text = content.decode(enc)
            break
        except Exception:
            continue
    else:
        return ""

    start = re.search(r"#\s*시작일[:\s]+(\d{8})", text)
    end   = re.search(r"#\s*종료일[:\s]+(\d{8})", text)

    def fmt(d):
        return f"{d[:4]}.{d[4:6]}.{d[6:]}" if d else "?"

    if start or end:
        s = fmt(start.group(1)) if start else "?"
        e = fmt(end.group(1))   if end   else "?"
        return f"{s} ~ {e}"
    return ""

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
        # 노출수 가중 평균 게재순위
        avg_pos = (g["position"] * g["impressions"]).sum() / impressions if impressions else np.nan
        return pd.Series({
            "키워드수": g["query"].nunique(),
            "노출수": int(impressions),
            "클릭수": int(clicks),
            "CTR": (clicks / impressions) if impressions else np.nan,
            "평균 게재순위": avg_pos,
            "Top3 노출 비중": (top3_impr / impressions) if impressions else np.nan,
        })

    out = df_std.groupby("brand_flag").apply(agg).reset_index()
    total_impr = df_std["impressions"].sum()
    total_clicks = df_std["clicks"].sum()
    total_top3 = df_std.loc[df_std["position"] <= 3, "impressions"].sum()
    total_avg_pos = (df_std["position"] * df_std["impressions"]).sum() / total_impr if total_impr else np.nan
    total = pd.DataFrame([{
        "brand_flag": "총합",
        "키워드수": df_std["query"].nunique(),
        "노출수": int(total_impr),
        "클릭수": int(total_clicks),
        "CTR": (total_clicks / total_impr) if total_impr else np.nan,
        "평균 게재순위": total_avg_pos,
        "Top3 노출 비중": (total_top3 / total_impr) if total_impr else np.nan,
    }])
    return pd.concat([out, total], ignore_index=True)

def excel_copy_button(summary: pd.DataFrame, key: str):
    """브랜드/비브랜드 10개 지표를 탭 구분으로 클립보드에 복사하는 버튼"""
    BRAND_LABEL = "브랜드/준브랜드(kt 포함)"
    NB_LABEL    = "일반(비브랜드)"

    b_rows = summary[summary["brand_flag"] == BRAND_LABEL]
    n_rows = summary[summary["brand_flag"] == NB_LABEL]

    if b_rows.empty or n_rows.empty:
        return

    b = b_rows.iloc[0]
    n = n_rows.iloc[0]

    def pct(v):
        return round(float(v) * 100, 2) if pd.notna(v) else ""

    def dec(v):
        return round(float(v), 2) if pd.notna(v) else ""

    vals = [
        int(b["노출수"]),
        int(b["클릭수"]),
        pct(b["CTR"]),
        dec(b["평균 게재순위"]),
        pct(b["Top3 노출 비중"]),
        int(n["노출수"]),
        int(n["클릭수"]),
        pct(n["CTR"]),
        dec(n["평균 게재순위"]),
        pct(n["Top3 노출 비중"]),
    ]

    tab_str = "\t".join(str(v) for v in vals)
    js_val  = json.dumps(tab_str)  # safely escaped

    headers = [
        "브랜드 노출수", "브랜드 클릭수", "브랜드 CTR(%)",
        "브랜드 평균 게재순위", "브랜드 Top3 노출 비중",
        "비브랜드 노출수", "비브랜드 클릭수", "비브랜드 CTR(%)",
        "비브랜드 평균 게재순위", "비브랜드 Top3 노출 비중",
    ]
    preview_rows = "".join(
        f"<tr><td style='padding:2px 10px;color:#888;font-size:11px;'>{h}</td>"
        f"<td style='padding:2px 10px;font-size:12px;font-weight:600;'>{v}</td></tr>"
        for h, v in zip(headers, vals)
    )

    html = f"""
    <div style="font-family:sans-serif;">
      <button id="copybtn_{key}"
        onclick="navigator.clipboard.writeText({js_val}).then(()=>{{
          var b=document.getElementById('copybtn_{key}');
          b.textContent='복사됨!';
          b.style.background='#d4edda';
          setTimeout(()=>{{b.textContent='엑셀 복사';b.style.background='#f0f2f6';}},2000);
        }})"
        style="padding:7px 18px;border-radius:6px;border:1px solid #ccc;
               cursor:pointer;background:#f0f2f6;font-size:13px;margin-bottom:8px;">
        엑셀 복사
      </button>
      <table style="border-collapse:collapse;">{preview_rows}</table>
    </div>
    """
    components.html(html, height=40 + len(vals) * 22 + 16)

# ---------- UI ----------
st.markdown("#### GSC 검색어 업로드 → 브랜드/일반 자동 분류 & 지표 산출")

uploaded_files = st.file_uploader("GSC 쿼리 CSV 업로드 (여러 파일 동시 가능)", type=["csv"], accept_multiple_files=True)

with st.expander("브랜드 분류 예외(선택)"):
    add_text = st.text_area("강제 포함 query (줄바꿈)", value="")
    remove_text = st.text_area("강제 제외 query (줄바꿈)", value="")

fmt = {
    "키워드수": "{:.0f}",
    "노출수": "{:.0f}",
    "클릭수": "{:.0f}",
    "CTR": "{:.2%}",
    "평균 게재순위": "{:.2f}",
    "Top3 노출 비중": "{:.2%}",
}

if uploaded_files:
    add_list = add_text.splitlines()
    remove_list = remove_text.splitlines()

    excel_output = io.BytesIO()
    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
        pd.DataFrame([{
            "note": "CTR은 클릭/노출로 재계산. Top3 노출 비중 = position<=3 노출합 / 전체노출합. 평균 게재순위 = 노출수 가중평균."
        }]).to_excel(writer, index=False, sheet_name="notes")

        for i, uploaded in enumerate(uploaded_files, start=1):
            content = uploaded.getvalue()
            date_range = extract_dates(content)
            label = f"파일 {i}  {date_range}" if date_range else f"파일 {i}  {uploaded.name}"

            df = read_csv_safely(content)

            q_col   = find_col(df, "query")
            c_col   = find_col(df, "clicks")
            i_col   = find_col(df, "impressions")
            ctr_col = find_col(df, "ctr")
            p_col   = find_col(df, "position")

            missing = [k for k, col in [("query", q_col), ("clicks", c_col), ("impressions", i_col), ("position", p_col)] if col is None]
            if missing:
                st.error(f"[{uploaded.name}] 필수 컬럼 없음: {missing}")
                continue

            df_std = pd.DataFrame({
                "query":       df[q_col].astype(str),
                "clicks":      pd.to_numeric(df[c_col], errors="coerce").fillna(0).astype(int),
                "impressions": pd.to_numeric(df[i_col], errors="coerce").fillna(0).astype(int),
                "ctr":         pd.to_numeric(df[ctr_col], errors="coerce") if ctr_col else np.nan,
                "position":    pd.to_numeric(df[p_col], errors="coerce"),
            })

            brand_mask = build_brand_mask(df_std["query"], add_list, remove_list)
            df_std["brand_flag"] = np.where(brand_mask, "브랜드/준브랜드(kt 포함)", "일반(비브랜드)")

            summary = summarize(df_std)

            st.markdown(f"---\n#### {label}")
            col1, col2, col3 = st.columns([1.2, 1, 0.8])
            with col1:
                st.markdown("**요약 지표**")
                st.dataframe(summary.style.format(fmt, na_rep="-"), use_container_width=True)
            with col2:
                st.markdown("**샘플 raw**")
                st.dataframe(df_std.head(30), use_container_width=True)
            with col3:
                st.markdown("**엑셀 붙여넣기용**")
                excel_copy_button(summary, key=f"file{i}")

            # 엑셀 시트명 (최대 31자 제한)
            sheet_prefix = f"f{i}"
            df_std.to_excel(writer, index=False, sheet_name=f"{sheet_prefix}_raw")
            summary.to_excel(writer, index=False, sheet_name=f"{sheet_prefix}_summary")

    st.download_button(
        label="엑셀 다운로드 (전체 파일)",
        data=excel_output.getvalue(),
        file_name="gsc_brand_nonbrand_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

else:
    st.info("CSV 업로드하면 자동으로 결과가 나와.")
