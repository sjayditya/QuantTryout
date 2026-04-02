"""Custom CSS injection for OptiPrice India dark theme."""

import streamlit as st


def inject_custom_css() -> None:
    """Inject custom CSS for the dark theme, cards, typography, and layout."""
    st.markdown(
        """
        <style>
        /* ── Google Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=DM+Sans:wght@400;500;600&display=swap');

        /* ── Root variables ── */
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-elevated: #1a1a28;
            --border: #2a2a3d;
            --text-primary: #e8e8f0;
            --text-secondary: #8888a0;
            --accent-green: #00e676;
            --accent-red: #ff1744;
            --accent-blue: #448aff;
            --accent-amber: #ffab00;
            --accent-purple: #b388ff;
        }

        /* ── Global typography ── */
        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', 'DM Sans', sans-serif;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 600;
        }

        /* ── Hide Streamlit branding ── */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* ── Metric cards ── */
        .model-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 0.8rem;
            transition: border-color 0.2s ease;
        }
        .model-card:hover {
            border-color: var(--text-secondary);
        }
        .model-card .price {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .model-card .model-name {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.3rem;
        }
        .model-card .subtitle {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 0.3rem;
        }

        /* ── Color accents for model cards ── */
        .card-bs { border-left: 3px solid var(--accent-blue); }
        .card-bs .model-name { color: var(--accent-blue); }
        .card-bs .price { color: var(--accent-blue); }

        .card-bayesian { border-left: 3px solid var(--accent-amber); }
        .card-bayesian .model-name { color: var(--accent-amber); }
        .card-bayesian .price { color: var(--accent-amber); }

        .card-nn { border-left: 3px solid var(--accent-purple); }
        .card-nn .model-name { color: var(--accent-purple); }
        .card-nn .price { color: var(--accent-purple); }

        /* ── Stock info card ── */
        .stock-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1rem;
        }
        .stock-card .stock-name {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        .stock-card .stock-price {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.8rem;
            font-weight: 700;
        }
        .stock-card .stock-change.positive {
            color: var(--accent-green);
        }
        .stock-card .stock-change.negative {
            color: var(--accent-red);
        }
        .stock-card .stock-meta {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        /* ── Positive / negative coloring ── */
        .positive { color: var(--accent-green) !important; }
        .negative { color: var(--accent-red) !important; }

        /* ── Sidebar refinements ── */
        section[data-testid="stSidebar"] {
            background-color: var(--bg-secondary);
            border-right: 1px solid var(--border);
        }

        /* ── Data tables ── */
        .stDataFrame {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }

        /* ── Tab styling ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: var(--bg-secondary);
            border-radius: 8px;
            padding: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            font-weight: 500;
            border-radius: 6px;
            padding: 8px 16px;
        }

        /* ── Dividers ── */
        hr {
            border-color: var(--border);
        }

        /* ── Tooltips ── */
        .stTooltipIcon {
            color: var(--text-secondary);
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-secondary);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
