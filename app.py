"""Streamlit UI for the Permission-to-Policy Checker."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.analyzer import AnalyzerConfig, analyze
from src.extractor import PolicyFetchError, extract_policy_text
from src.scraper import cache_metadata, fetch_app_metadata, load_cached

CACHE = Path("data/cache")

st.set_page_config(page_title="Permission-to-Policy Checker", layout="wide")
st.title("🔐 Permission-to-Policy Checker")
st.caption("LLM-based mismatch detection between Android permissions and privacy policies.")

with st.sidebar:
    st.header("Settings")
    app_id = st.text_input("Play Store App ID", value="com.whatsapp")
    backend = st.selectbox(
        "Backend",
        [
            "auto",
            "claude",
            "groq",
            "llama",
            "heuristic",
            "finetuned",
            "agent-claude",
            "agent-groq",
            "agent-llama",
        ],
        index=0,
    )
    use_cache = st.checkbox("Use cached metadata", value=True)
    run = st.button("Audit App", type="primary")

if run and app_id.strip():
    with st.spinner("Fetching app metadata..."):
        meta = load_cached(app_id, CACHE) if use_cache else None
        if meta is None:
            try:
                meta = fetch_app_metadata(app_id)
                cache_metadata(meta, CACHE)
            except Exception as e:
                st.error(f"Could not fetch app metadata: {e}")
                st.stop()

    cols = st.columns([2, 1])
    with cols[0]:
        st.subheader(meta.title)
        st.write(f"**Developer:** {meta.developer}")
        st.write(f"**Content rating:** {meta.content_rating}")
        if meta.privacy_policy_url:
            st.write(f"**Policy:** {meta.privacy_policy_url}")
    with cols[1]:
        st.metric("Permissions requested", len(meta.permissions))

    with st.spinner("Fetching and cleaning privacy policy..."):
        try:
            policy_text = extract_policy_text(meta.privacy_policy_url)
        except PolicyFetchError as e:
            st.warning(f"Policy fetch failed: {e}")
            policy_text = ""

    with st.spinner(f"Auditing with {backend} backend..."):
        cfg = AnalyzerConfig(backend=backend)
        report = analyze(
            app_title=meta.title,
            app_id=meta.app_id,
            permissions=meta.permissions,
            policy_text=policy_text,
            cfg=cfg,
            policy_url=meta.privacy_policy_url,
        )

    if report.error:
        st.error(report.error)
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ Covered", report.covered_count)
        c2.metric("⚠️ Unclear", report.unclear_count)
        c3.metric("❌ Mismatch", report.mismatch_count)
        c4.metric("Policy chars", report.policy_chars)

        st.markdown("### Plain-language summary")
        st.info(report.summary)

        st.markdown("### Per-permission verdicts")
        for v in report.permissions:
            icon = {"covered": "✅", "unclear": "⚠️", "mismatch": "❌"}[v.verdict.value]
            with st.expander(f"{icon} {v.permission} — {v.verdict.value.upper()}"):
                st.write(f"**Data access:** {v.data_access}")
                st.write(f"**Verdict reasoning:** {v.reasoning}")
                st.write(f"**Policy span:**")
                if v.policy_mention == "NOT MENTIONED":
                    st.error("NOT MENTIONED in policy")
                else:
                    st.code(v.policy_mention, language=None)

        with st.expander("Raw JSON report"):
            st.json(report.model_dump())

st.markdown("---")
st.caption(
    "⚖️ This is a screening instrument, not a legal verdict. Verdicts must be reviewed by a "
    "human expert before drawing conclusions about actual policy violations."
)
