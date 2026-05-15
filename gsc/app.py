import io
import json
import re
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange, Dimension, Metric, RunReportRequest,
    FilterExpression, FilterExpressionList, Filter,
)
from google.oauth2 import service_account

st.set_page_config(page_title="GSC Brand vs Nonbrand Analyzer", layout="wide")

# ============================================================
# GA4 Property ID — 여기에 숫자만 입력하세요
# ============================================================
GA4_PROPERTY_ID = "YOUR_PROPERTY_ID"
# ============================================================

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

def excel_copy_section(summary: pd.DataFrame, key: str):
    BRAND_LABEL = "브랜드/준브랜드(kt 포함)"
    NB_LABEL    = "일반(비브랜드)"
    b_rows = summary[summary["brand_flag"] == BRAND_LABEL]
    n_rows = summary[summary["brand_flag"] == NB_LABEL]
    if b_rows.empty or n_rows.empty:
        return
    b = b_rows.iloc[0]
    n = n_rows.iloc[0]
    def pct(v):
        return f"{round(float(v) * 100, 2)}%" if pd.notna(v) else ""
    def dec(v):
        return round(float(v), 2) if pd.notna(v) else ""
    vals = [
        int(b["노출수"]), int(b["클릭수"]), pct(b["CTR"]),
        dec(b["평균 게재순위"]), pct(b["Top3 노출 비중"]),
        int(n["노출수"]), int(n["클릭수"]), pct(n["CTR"]),
        dec(n["평균 게재순위"]), pct(n["Top3 노출 비중"]),
    ]
    headers = [
        "브랜드 노출수", "브랜드 클릭수", "브랜드 CTR(%)",
        "브랜드 평균 게재순위", "브랜드 Top3 노출 비중",
        "비브랜드 노출수", "비브랜드 클릭수", "비브랜드 CTR(%)",
        "비브랜드 평균 게재순위", "비브랜드 Top3 노출 비중",
    ]
    preview_df = pd.DataFrame({"항목": headers, "값": [str(v) for v in vals]})
    st.dataframe(preview_df, hide_index=True, use_container_width=True)
    tab_str = "\t".join(str(v) for v in vals)
    st.text_area("전체 선택(Ctrl+A) 후 복사 → 엑셀에 붙여넣기",
                 value=tab_str, height=68, key=key)

# ---------- GA4 ----------
def get_ga4_client(credentials_bytes: bytes) -> BetaAnalyticsDataClient:
    creds_dict = json.loads(credentials_bytes)
    credentials = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/analytics.readonly"],
    )
    return BetaAnalyticsDataClient(credentials=credentials)

def ga4_response_to_df(response) -> pd.DataFrame:
    dim_names = [h.name for h in response.dimension_headers]
    met_names = [h.name for h in response.metric_headers]
    rows = []
    for row in response.rows:
        dims = [d.value for d in row.dimension_values]
        mets = [m.value for m in row.metric_values]
        rows.append(dims + mets)
    df = pd.DataFrame(rows, columns=dim_names + met_names)
    for col in met_names:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def fetch_sessions(client, property_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """이벤트별 세션수 (가입신청서 | session_start | 유심_배송신청서)"""
    request = RunReportRequest(
        property=f"properties/{property_id}",
        date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        dimensions=[
            Dimension(name="date"),
            Dimension(name="sessionDefaultChannelGroup"),
            Dimension(name="sessionSourceMedium"),
            Dimension(name="eventName"),
        ],
        metrics=[Metric(name="sessions")],
        dimension_filter=FilterExpression(
            and_group=FilterExpressionList(expressions=[
                FilterExpression(filter=Filter(
                    field_name="eventName",
                    string_filter=Filter.StringFilter(
                        match_type=Filter.StringFilter.MatchType.PARTIAL_REGEXP,
                        value="가입신청서|session_start|유심_배송신청서",
                    ),
                )),
                FilterExpression(filter=Filter(
                    field_name="eventName",
                    string_filter=Filter.StringFilter(
                        match_type=Filter.StringFilter.MatchType.PARTIAL_REGEXP,
                        value="가입신청서|session_start",
                    ),
                )),
            ])
        ),
        limit=100000,
    )
    return ga4_response_to_df(client.run_report(request))

def fetch_users(client, property_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """session_start 이벤트 총 사용자수"""
    request = RunReportRequest(
        property=f"properties/{property_id}",
        date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        dimensions=[
            Dimension(name="date"),
            Dimension(name="sessionDefaultChannelGroup"),
            Dimension(name="sessionSourceMedium"),
            Dimension(name="eventName"),
        ],
        metrics=[Metric(name="totalUsers")],
        dimension_filter=FilterExpression(
            filter=Filter(
                field_name="eventName",
                string_filter=Filter.StringFilter(
                    match_type=Filter.StringFilter.MatchType.CONTAINS,
                    value="session_start",
                ),
            )
        ),
        limit=100000,
    )
    return ga4_response_to_df(client.run_report(request))

# ================================================================
# UI
# ================================================================
st.markdown("#### GSC Brand vs Nonbrand Analyzer")

tab_gsc, tab_ga4 = st.tabs(["GSC 분석", "GA4 분석"])

# ── GSC 탭 ──────────────────────────────────────────────────────
with tab_gsc:
    uploaded_files = st.file_uploader(
        "GSC 쿼리 CSV 업로드 (여러 파일 동시 가능)",
        type=["csv"], accept_multiple_files=True, key="gsc_upload"
    )
    with st.expander("브랜드 분류 예외(선택)"):
        add_text    = st.text_area("강제 포함 query (줄바꿈)", value="")
        remove_text = st.text_area("강제 제외 query (줄바꿈)", value="")

    fmt = {
        "키워드수": "{:.0f}", "노출수": "{:.0f}", "클릭수": "{:.0f}",
        "CTR": "{:.2%}", "평균 게재순위": "{:.2f}", "Top3 노출 비중": "{:.2%}",
    }

    if uploaded_files:
        add_list    = add_text.splitlines()
        remove_list = remove_text.splitlines()

        excel_output = io.BytesIO()
        with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
            pd.DataFrame([{
                "note": "CTR은 클릭/노출로 재계산. Top3 노출 비중 = position<=3 노출합 / 전체노출합. 평균 게재순위 = 노출수 가중평균."
            }]).to_excel(writer, index=False, sheet_name="notes")

            for i, uploaded in enumerate(uploaded_files, start=1):
                content    = uploaded.getvalue()
                date_range = extract_dates(content)
                label      = f"파일 {i}  {date_range}" if date_range else f"파일 {i}  {uploaded.name}"
                df         = read_csv_safely(content)

                q_col   = find_col(df, "query")
                c_col   = find_col(df, "clicks")
                i_col   = find_col(df, "impressions")
                ctr_col = find_col(df, "ctr")
                p_col   = find_col(df, "position")

                missing = [k for k, col in [("query", q_col), ("clicks", c_col),
                           ("impressions", i_col), ("position", p_col)] if col is None]
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
                brand_mask       = build_brand_mask(df_std["query"], add_list, remove_list)
                df_std["brand_flag"] = np.where(brand_mask, "브랜드/준브랜드(kt 포함)", "일반(비브랜드)")
                summary          = summarize(df_std)

                st.markdown(f"---\n#### {label}")
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("**요약 지표**")
                    st.dataframe(summary.style.format(fmt, na_rep="-"), use_container_width=True)
                with col2:
                    st.markdown("**엑셀 붙여넣기용**")
                    excel_copy_section(summary, key=f"file{i}")
                with st.expander("샘플 raw 보기", expanded=False):
                    st.dataframe(df_std.head(30), use_container_width=True)

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

# ── GA4 탭 ──────────────────────────────────────────────────────
with tab_ga4:
    st.markdown("#### GA4 데이터 조회")

    # 인증
    creds_file = st.file_uploader(
        "서비스 계정 JSON 업로드", type=["json"], key="ga4_creds"
    )

    # 날짜 선택
    today     = datetime.date.today()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("시작일", value=today - datetime.timedelta(days=14))
    with col2:
        end_date = st.date_input("종료일", value=today - datetime.timedelta(days=1))

    if start_date > end_date:
        st.error("시작일이 종료일보다 늦을 수 없어요.")
    elif GA4_PROPERTY_ID == "YOUR_PROPERTY_ID":
        st.warning("app.py 상단의 `GA4_PROPERTY_ID`에 실제 Property ID를 입력해주세요.")
    elif creds_file is None:
        st.info("서비스 계정 JSON을 업로드하면 데이터를 불러올 수 있어요.")
    else:
        if st.button("GA4 데이터 불러오기", type="primary"):
            try:
                client    = get_ga4_client(creds_file.getvalue())
                start_str = start_date.strftime("%Y-%m-%d")
                end_str   = end_date.strftime("%Y-%m-%d")

                with st.spinner("GA4 데이터 불러오는 중..."):
                    sessions_df = fetch_sessions(client, GA4_PROPERTY_ID, start_str, end_str)
                    users_df    = fetch_users(client, GA4_PROPERTY_ID, start_str, end_str)

                st.session_state["ga4_sessions"] = sessions_df
                st.session_state["ga4_users"]    = users_df

            except Exception as e:
                st.error(f"GA4 API 오류: {e}")

    # 결과 표시 (버튼 클릭 후에도 유지)
    if "ga4_sessions" in st.session_state:
        sessions_df = st.session_state["ga4_sessions"]
        users_df    = st.session_state["ga4_users"]

        st.markdown("---")
        st.markdown(f"**조회 기간:** {start_date} ~ {end_date}")

        st.markdown("**세션수 (이벤트별)**")
        st.dataframe(sessions_df, use_container_width=True, hide_index=True)

        st.markdown("**총 사용자수 (session_start)**")
        st.dataframe(users_df, use_container_width=True, hide_index=True)

        # 엑셀 다운로드
        ga4_excel = io.BytesIO()
        with pd.ExcelWriter(ga4_excel, engine="openpyxl") as writer:
            sessions_df.to_excel(writer, index=False, sheet_name="sessions")
            users_df.to_excel(writer, index=False, sheet_name="users")
        st.download_button(
            label="GA4 엑셀 다운로드",
            data=ga4_excel.getvalue(),
            file_name=f"ga4_{start_date}_{end_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
