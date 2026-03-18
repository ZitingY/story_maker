"""StoryWeaver – Streamlit chat-based text adventure UI.

Layout (Streamlit):
    Sidebar:    KG visualisation · consistency trend · debug info · download
    Main area:  controls · story chat · option buttons · evaluation dashboard
"""
from __future__ import annotations

import os
import sys
import time
import logging
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine.game_engine import GameEngine, TurnResult
from src.nlg.option_generator import StoryOption
from src.evaluation.metrics import full_evaluation
from src.evaluation.llm_judge import judge as llm_judge

logger = logging.getLogger(__name__)


# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="StoryWeaver - AI 故事生成器",
    page_icon="🎭",
    layout="wide",
)

# Lock UI to dark mode to avoid readability issues from theme switching.
st.session_state.ui_mode = "dark"


def _theme_tokens(mode: str) -> dict[str, str]:
    """Return CSS tokens for light/dark unified visual modes."""
    if mode == "dark":
        return {
            "bg": "radial-gradient(circle at 8% 8%, #1b2b47 0%, #0d1526 45%, #0a0f1c 100%)",
            "text": "#e8eefc",
            "muted": "#b5c0db",
            "hero1": "#1e5dbc",
            "hero2": "#174a96",
            "hero3": "#12356e",
            "hero_shadow": "rgba(6, 17, 39, 0.45)",
            "panel": "rgba(15, 24, 42, 0.82)",
            "panel_border": "#2d436f",
            "title": "#9fc3ff",
            "primary": "#4b8fff",
            "primary_hover": "#74a7ff",
            "input_bg": "#111b2e",
            "input_border": "#35517f",
            "chat_bg": "rgba(255, 255, 255, 0.03)",
        }

    return {
        "bg": "radial-gradient(circle at 5% 10%, #e9f5ff 0%, #f6fbff 38%, #ffffff 100%)",
        "text": "#10213d",
        "muted": "#4b5563",
        "hero1": "#0b3d91",
        "hero2": "#1768c3",
        "hero3": "#2f8fed",
        "hero_shadow": "rgba(11, 61, 145, 0.24)",
        "panel": "rgba(255, 255, 255, 0.86)",
        "panel_border": "#dce8f8",
        "title": "#0b3d91",
        "primary": "#1f6fff",
        "primary_hover": "#0f57d4",
        "input_bg": "#ffffff",
        "input_border": "#b7cdee",
        "chat_bg": "rgba(255, 255, 255, 0.75)",
    }


tokens = _theme_tokens(st.session_state.ui_mode)

st.markdown(
    f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700;800&display=swap');

    html, body, [class*="css"]  {{
        font-family: 'Noto Sans SC', sans-serif;
        color: {tokens['text']};
    }}

    .stApp {{
        background: {tokens['bg']};
        color: {tokens['text']};
    }}

    .hero {{
        background: linear-gradient(120deg, {tokens['hero1']} 0%, {tokens['hero2']} 55%, {tokens['hero3']} 100%);
        color: white;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 12px;
        box-shadow: 0 10px 28px {tokens['hero_shadow']};
    }}

    .hero h2 {{
        margin: 0 0 8px 0;
        font-size: 1.6rem;
        font-weight: 800;
    }}

    .hero p {{
        margin: 0;
        opacity: 0.96;
        font-size: 0.95rem;
    }}

    .section-title {{
        font-size: 1.08rem;
        font-weight: 800;
        color: {tokens['title']};
        margin-top: 2px;
        margin-bottom: 10px;
    }}

    .muted-note {{
        color: {tokens['muted']};
        font-size: 0.9rem;
    }}

    .metric-card {{
        background: {tokens['panel']};
        border: 1px solid {tokens['panel_border']};
        border-radius: 12px;
        padding: 10px 12px;
        backdrop-filter: blur(7px);
    }}

    .stChatMessage, .st-expander, section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {{
        background: {tokens['chat_bg']};
        border: 1px solid {tokens['panel_border']};
        border-radius: 12px;
    }}

    .stTextInput > div > div > input,
    div[data-baseweb="select"] > div,
    .stTextArea textarea {{
        background: {tokens['input_bg']};
        border: 1px solid {tokens['input_border']};
        color: {tokens['text']};
    }}

    .stButton > button {{
        border-radius: 10px;
        border: 1px solid {tokens['input_border']};
        transition: all 0.22s ease;
    }}

    .stButton > button[kind="primary"] {{
        background: {tokens['primary']};
        border: 1px solid {tokens['primary']};
        color: #ffffff;
    }}

    .stButton > button:hover {{
        border-color: {tokens['primary_hover']};
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.16);
    }}

    .stMarkdown, .stCaption, label, p, span {{
        color: {tokens['text']};
    }}

    section[data-testid="stSidebar"] {{
        background: {tokens['panel']};
        border-right: 1px solid {tokens['panel_border']};
    }}

    .stAlert {{
        border-radius: 10px;
    }}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h2>🎭 StoryWeaver: 动态知识图谱故事生成器</h2>
    <p>混合 NLU + LLM + KG 的多轮互动叙事系统，支持实时世界状态追踪与会话评测。</p>
</div>
""",
    unsafe_allow_html=True,
)


# ── Session State initialisation ─────────────────────────────────────────

_DEFAULTS = {
    "engine": None,
    "history": [],                # list[dict] with role / content
    "consistency_history": [],    # float per turn
    "kg_html": "",
    "options": [],                # list[StoryOption]
    "nlu_debug": {},
    "eval_result": "",
    "eval_auto": {},
    "eval_llm": {},
    "eval_prev_auto": {},
    "eval_prev_llm": {},
    "eval_at": "",
    "chat_fold_mode": False,
    "last_elapsed": 0.0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers ──────────────────────────────────────────────────────────────

def _process_action(action: str) -> None:
    """Run a player action through the engine and update session state."""
    engine: GameEngine | None = st.session_state.engine
    if engine is None:
        st.warning("请先开始新游戏，再输入行动。")
        return

    start = time.time()
    st.session_state.history.append({"role": "user", "content": action})

    result: TurnResult = engine.process_turn(action)

    assistant_msg = result.story_text
    if result.conflicts:
        assistant_msg += "\n\n*⚠ World-consistency notes:*\n" + "\n".join(
            f"- {c}" for c in result.conflicts
        )
    st.session_state.history.append({"role": "assistant", "content": assistant_msg})

    st.session_state.kg_html = result.kg_html
    st.session_state.options = result.options
    if result.nlu_debug:
        st.session_state.nlu_debug = result.nlu_debug

    # Consistency tracking (1.0 = perfect, decreases with conflicts)
    n_conflicts = len(result.conflicts)
    score = 1.0 if n_conflicts == 0 else max(0.0, 1.0 - n_conflicts * 0.2)
    st.session_state.consistency_history.append(score)

    st.session_state.last_elapsed = time.time() - start


def _run_evaluation() -> tuple[str, dict, dict]:
    """Collect session data from the engine and run all evaluations."""
    engine: GameEngine | None = st.session_state.engine
    if engine is None or not engine.all_story_texts:
        return "*暂无可评测内容，请先开始并推进故事。*", {}, {}

    texts = engine.all_story_texts

    # Automatic metrics
    auto = full_evaluation(
        texts=texts,
        entity_names=engine.kg_entity_names,
        turn_conflict_counts=engine.turn_conflict_counts,
    )

    # LLM judge
    transcript = "\n".join(texts)
    llm_scores = llm_judge(transcript)

    # Format as Markdown table (kept for one-click report display)
    lines = [
        "### 自动指标",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Distinct-1 | {auto.get('distinct_1', 0):.4f} |",
        f"| Distinct-2 | {auto.get('distinct_2', 0):.4f} |",
        f"| Distinct-3 | {auto.get('distinct_3', 0):.4f} |",
        f"| Self-BLEU | {auto.get('self_bleu', 0):.4f} |",
        f"| Entity Coverage | {auto.get('entity_coverage', 0):.2%} |",
        f"| Consistency Rate | {auto.get('consistency_rate', 0):.2%} |",
        "",
        "### LLM Judge (1-10)",
        "",
        "| Dimension | Score |",
        "|-----------|-------|",
        f"| Narrative Quality | {llm_scores.get('narrative_quality', 0)} |",
        f"| Consistency | {llm_scores.get('consistency', 0)} |",
        f"| Player Agency | {llm_scores.get('player_agency', 0)} |",
        f"| Creativity | {llm_scores.get('creativity', 0)} |",
        f"| Pacing | {llm_scores.get('pacing', 0)} |",
        f"| **Average** | **{llm_scores.get('average', 0)}** |",
    ]
    return "\n".join(lines), auto, llm_scores


def _story_turn_count() -> int:
    """Return number of user turns in current chat history."""
    return sum(1 for m in st.session_state.history if m["role"] == "user")


def _build_turn_pairs(history: list[dict]) -> tuple[str, list[tuple[int, str, str]]]:
    """Return opening narration and user-assistant turn pairs."""
    opening = ""
    pairs: list[tuple[int, str, str]] = []
    i = 0

    if history and history[0]["role"] == "assistant":
        opening = history[0]["content"]
        i = 1

    turn = 1
    while i < len(history):
        user_text = ""
        ai_text = ""

        if i < len(history) and history[i]["role"] == "user":
            user_text = history[i]["content"]
            i += 1
        if i < len(history) and history[i]["role"] == "assistant":
            ai_text = history[i]["content"]
            i += 1

        if user_text or ai_text:
            pairs.append((turn, user_text, ai_text))
            turn += 1

    return opening, pairs


def _delta_str(current: float, previous: dict, key: str, fmt: str = ".4f") -> str | None:
    """Format metric delta string for st.metric."""
    if not previous or key not in previous:
        return None
    diff = float(current) - float(previous.get(key, 0))
    return f"{diff:+{fmt}}"


def _delta_pct(current: float, previous: dict, key: str) -> str | None:
    """Format percentage-point delta for st.metric."""
    if not previous or key not in previous:
        return None
    diff = float(current) - float(previous.get(key, 0))
    return f"{diff * 100:+.2f}pp"


# ── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<div class='section-title'>📊 故事世界观知识图谱</div>", unsafe_allow_html=True)
    if st.session_state.kg_html:
        components.html(st.session_state.kg_html, height=480, scrolling=True)
    else:
        st.info("开始游戏后将显示知识图谱。")

    st.markdown("<div class='section-title'>📈 一致性趋势</div>", unsafe_allow_html=True)
    if st.session_state.consistency_history:
        recent = st.session_state.consistency_history[-5:]
        offset = max(0, len(st.session_state.consistency_history) - 5)
        for i, score in enumerate(recent):
            st.progress(
                max(0.0, min(1.0, score)),
                text=f"轮次 {offset + i + 1}: {score:.2f}",
            )
        st.line_chart(st.session_state.consistency_history, height=120, use_container_width=True)
    else:
        st.caption("第一轮行动后显示一致性分数")

    with st.expander("🔍 NLU 解析详情", expanded=False):
        nlu = st.session_state.nlu_debug
        if nlu:
            st.markdown(f"**消解后输入:** {nlu.get('resolved_input', '')}")
            st.markdown(
                f"**意图:** {nlu.get('intent', '?')}  "
                f"(置信度 {nlu.get('confidence', 0):.2f})"
            )
            st.markdown(f"**实体:** {nlu.get('entities', [])}")
        else:
            st.caption("行动后自动展示 NLU 解析信息")

    st.markdown("---")

    turn_count = _story_turn_count()
    engine: GameEngine | None = st.session_state.engine
    entity_count = len(engine.kg_entity_names) if engine else 0
    conflict_total = sum(engine.turn_conflict_counts) if engine else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("轮次", turn_count)
    c2.metric("实体", entity_count)
    c3.metric("冲突", conflict_total)

    st.markdown("---")

    if st.session_state.history:
        full_story = "\n\n".join(
            f"{'[Player]' if m['role'] == 'user' else '[Story]'}: {m['content']}"
            for m in st.session_state.history
        )
        st.download_button(
            "📥 下载完整故事",
            full_story,
            file_name="full_story.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ── Main area – Game controls ────────────────────────────────────────────

col_genre, col_btn = st.columns([3, 1])
with col_genre:
    genre = st.text_input(
        "Genre",
        value="fantasy",
        placeholder="类型，如 fantasy / sci-fi / mystery",
        label_visibility="collapsed",
    )
with col_btn:
    new_game_clicked = st.button(
        "🎮 开始新游戏", type="primary", use_container_width=True
    )

if new_game_clicked:
    with st.spinner("正在初始化冒险世界…"):
        engine = GameEngine(genre=genre or "fantasy")
        st.session_state.engine = engine
        result: TurnResult = engine.start_game()
        st.session_state.history = [
            {"role": "assistant", "content": result.story_text}
        ]
        st.session_state.kg_html = result.kg_html
        st.session_state.options = result.options
        st.session_state.consistency_history = []
        st.session_state.nlu_debug = {}
        st.session_state.eval_result = ""
        st.session_state.eval_auto = {}
        st.session_state.eval_llm = {}
        st.session_state.eval_prev_auto = {}
        st.session_state.eval_prev_llm = {}
        st.session_state.eval_at = ""
        st.session_state.last_elapsed = 0.0
    st.rerun()

if st.session_state.engine is None:
    st.info("点击上方“开始新游戏”后，即可进行剧情互动。")


# ── Chat history ─────────────────────────────────────────────────────────

chat_fold_mode = st.toggle(
    "按轮次折叠历史",
    value=st.session_state.chat_fold_mode,
    help="开启后按“用户输入 + 系统回复”折叠显示，适合长会话阅读。",
)
st.session_state.chat_fold_mode = chat_fold_mode

if chat_fold_mode:
    opening, turn_pairs = _build_turn_pairs(st.session_state.history)
    if opening:
        with st.chat_message("assistant"):
            st.markdown(opening)

    for turn, user_text, ai_text in turn_pairs:
        preview = user_text[:28] + ("..." if len(user_text) > 28 else "")
        title = f"第 {turn} 轮 | {preview or '（空输入）'}"
        with st.expander(title, expanded=(turn == len(turn_pairs))):
            if user_text:
                st.markdown("**你：**")
                st.markdown(user_text)
            if ai_text:
                st.markdown("**系统：**")
                st.markdown(ai_text)
else:
    for msg in st.session_state.history:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["content"])


# ── Option buttons ───────────────────────────────────────────────────────

if st.session_state.options:
    st.markdown("<div class='section-title'>🧭 分支选项</div>", unsafe_allow_html=True)
    st.caption("可直接点击选项，也可在下方输入自由行动。")
    opt_cols = st.columns(len(st.session_state.options))
    for idx, opt in enumerate(st.session_state.options):
        with opt_cols[idx]:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.caption(f"意图: {opt.intent_hint} | 风险: {opt.risk_level}")
            btn_key = f"opt_{idx}_{len(st.session_state.history)}"
            if st.button(
                f"{idx + 1}. {opt.text}",
                key=btn_key,
                use_container_width=True,
            ):
                _process_action(opt.text)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


# ── Chat input ───────────────────────────────────────────────────────────

user_input = st.chat_input("输入你的行动（例如：调查遗迹中的符文）…")
if user_input:
    _process_action(user_input)
    st.rerun()


# ── Performance footer ──────────────────────────────────────────────────

if st.session_state.last_elapsed > 0:
    st.caption(f"✅ 本轮生成耗时 {st.session_state.last_elapsed:.2f}s")


# ── Evaluation Dashboard (kept and enhanced) ───────────────────────────

st.markdown("---")
st.markdown("<div class='section-title'>📊 会话评测面板</div>", unsafe_allow_html=True)

col_eval_btn, col_eval_hint = st.columns([1, 2])
with col_eval_btn:
    run_eval = st.button("运行评测", use_container_width=True)
with col_eval_hint:
    st.markdown("<div class='muted-note'>评测将基于当前会话文本计算自动指标，并调用 LLM Judge 打分。</div>", unsafe_allow_html=True)

if run_eval:
    with st.spinner("正在计算评测结果…"):
        report_md, auto_scores, llm_scores = _run_evaluation()

        if st.session_state.eval_auto and st.session_state.eval_llm:
            st.session_state.eval_prev_auto = st.session_state.eval_auto.copy()
            st.session_state.eval_prev_llm = st.session_state.eval_llm.copy()

        st.session_state.eval_result = report_md
        st.session_state.eval_auto = auto_scores
        st.session_state.eval_llm = llm_scores
        st.session_state.eval_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if st.session_state.eval_result:
    if st.session_state.eval_auto and st.session_state.eval_llm:
        st.caption(f"最近评测时间: {st.session_state.eval_at}")

        auto = st.session_state.eval_auto
        llm = st.session_state.eval_llm
        prev_auto = st.session_state.eval_prev_auto
        prev_llm = st.session_state.eval_prev_llm

        st.markdown("**自动指标**")
        a1, a2, a3 = st.columns(3)
        a1.metric(
            "Distinct-1",
            f"{auto.get('distinct_1', 0):.4f}",
            _delta_str(auto.get("distinct_1", 0), prev_auto, "distinct_1"),
        )
        a2.metric(
            "Distinct-2",
            f"{auto.get('distinct_2', 0):.4f}",
            _delta_str(auto.get("distinct_2", 0), prev_auto, "distinct_2"),
        )
        a3.metric(
            "Distinct-3",
            f"{auto.get('distinct_3', 0):.4f}",
            _delta_str(auto.get("distinct_3", 0), prev_auto, "distinct_3"),
        )

        a4, a5, a6 = st.columns(3)
        a4.metric(
            "Self-BLEU",
            f"{auto.get('self_bleu', 0):.4f}",
            _delta_str(auto.get("self_bleu", 0), prev_auto, "self_bleu"),
        )
        a5.metric(
            "Entity Coverage",
            f"{auto.get('entity_coverage', 0):.2%}",
            _delta_pct(auto.get("entity_coverage", 0), prev_auto, "entity_coverage"),
        )
        a6.metric(
            "Consistency Rate",
            f"{auto.get('consistency_rate', 0):.2%}",
            _delta_pct(auto.get("consistency_rate", 0), prev_auto, "consistency_rate"),
        )

        st.markdown("**LLM Judge 维度评分**")
        j1, j2, j3 = st.columns(3)
        j1.metric(
            "Narrative",
            llm.get("narrative_quality", 0),
            _delta_str(llm.get("narrative_quality", 0), prev_llm, "narrative_quality", ".2f"),
        )
        j2.metric(
            "Consistency",
            llm.get("consistency", 0),
            _delta_str(llm.get("consistency", 0), prev_llm, "consistency", ".2f"),
        )
        j3.metric(
            "Agency",
            llm.get("player_agency", 0),
            _delta_str(llm.get("player_agency", 0), prev_llm, "player_agency", ".2f"),
        )

        j4, j5, j6 = st.columns(3)
        j4.metric(
            "Creativity",
            llm.get("creativity", 0),
            _delta_str(llm.get("creativity", 0), prev_llm, "creativity", ".2f"),
        )
        j5.metric(
            "Pacing",
            llm.get("pacing", 0),
            _delta_str(llm.get("pacing", 0), prev_llm, "pacing", ".2f"),
        )
        j6.metric(
            "Average",
            llm.get("average", 0),
            _delta_str(llm.get("average", 0), prev_llm, "average", ".2f"),
        )

        with st.expander("查看原始评测报告 (Markdown)", expanded=False):
            st.markdown(st.session_state.eval_result)
    else:
        st.markdown(st.session_state.eval_result)
