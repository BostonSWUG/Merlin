"""Codename Merlin — Balloon Quantity Analyzer (Streamlit UI)"""

from __future__ import annotations

import tempfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_path
from PIL import Image, ImageDraw

from balloon_quantity_analyzer.analyzer import BalloonAnalyzer
from balloon_quantity_analyzer.models import AnalysisReport, AnalyzerConfig, TallyResult
from balloon_quantity_analyzer.report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# BOM Excel parser
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _parse_bom_excel(file_bytes: bytes) -> dict[str, dict]:
    """Parse a PLM BOM export and return {find_number: {"qty": int, "part": str}} for Level-1 rows.

    The Agile PLM export has columns:
      - Level: hierarchy depth (0 = top assy, 1 = direct children)
      - Number: the part number
      - BOM.Find Num: the Find Number shown on the assembly drawing
      - BOM.Qty: quantity per parent assembly

    We keep only Level-1 rows (direct BOM children) with a non-zero
    Find Number and a positive quantity, since Level-1 rows with
    Find Num 0 or Qty 0 are reference documents (drawings, specs).
    """
    df = pd.read_excel(BytesIO(file_bytes), sheet_name=0)

    level_col = "Level"
    find_col = "BOM.Find Num"
    qty_col = "BOM.Qty"
    number_col = "Number"

    for col in (level_col, find_col, qty_col):
        if col not in df.columns:
            return {}

    has_number = number_col in df.columns

    # Keep only Level-1 rows (direct children of the top-level assembly)
    level1 = df[df[level_col] == 1].copy()

    # Coerce to numeric — Find Num is typically an integer in the PLM export
    level1[find_col] = pd.to_numeric(level1[find_col], errors="coerce")
    level1[qty_col] = pd.to_numeric(level1[qty_col], errors="coerce").fillna(0)

    # Drop rows where Find Num is NaN or 0 (reference docs / drawings)
    level1 = level1[level1[find_col].notna() & (level1[find_col] != 0)]

    bom: dict[str, dict] = {}
    for _, row in level1.iterrows():
        fn = str(int(row[find_col]))  # "1", "2", ... "101"
        qty = int(row[qty_col])
        part = str(row[number_col]).strip() if has_number else ""
        if qty > 0:
            bom[fn] = {"qty": qty, "part": part}
    return bom


st.set_page_config(page_title="Merlin — Balloon Analyzer", page_icon="🧙", layout="wide")

# Compact sidebar spacing
st.markdown(
    """
    <style>
    /* Sidebar top padding — nuke all possible sources */
    section[data-testid="stSidebar"] {padding-top: 0 !important; margin-top: 0 !important; top: 0 !important;}
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div > div,
    section[data-testid="stSidebar"] > div > div > div,
    section[data-testid="stSidebar"] > div > div > div > div {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    [data-testid="stSidebarUserContent"] {padding: 0 !important; margin: 0 !important;}
    [data-testid="stSidebarNav"] {display: none !important;}
    [data-testid="stSidebarHeader"] {display: none !important;}
    /* Main content top padding */
    .stMainBlockContainer {padding-top: 1rem !important;}
    /* Hide collapse button */
    [data-testid="stSidebarCollapseButton"] {display: none !important;}
    /* Dividers */
    [data-testid="stSidebar"] hr {margin: 0.15rem 0;}
    /* Headers */
    [data-testid="stSidebar"] h2 {margin: 0.15rem 0; font-size: 1rem;}
    /* Reduce gap between all sidebar block elements */
    [data-testid="stSidebar"] .stElementContainer {margin-bottom: 0;}
    [data-testid="stSidebar"] .stVerticalBlock > div {gap: 0.35rem;}
    /* File uploader compact */
    [data-testid="stSidebar"] .stFileUploader {margin-bottom: 0;}
    [data-testid="stSidebar"] .stFileUploader section {padding: 0.25rem;}
    [data-testid="stSidebar"] .stFileUploader small {display: none;}
    /* Alert messages (success/warning) — outer and inner */
    [data-testid="stSidebar"] .stAlert {padding: 0 !important; margin: 0 !important;}
    [data-testid="stSidebar"] .stAlert > div {padding: 0.25rem 0.5rem !important; gap: 0.25rem !important;}
    [data-testid="stSidebar"] .stAlert p,
    [data-testid="stSidebar"] .stAlert div {font-size: 0.75rem !important; line-height: 1.2 !important;}
    /* Slider compact */
    [data-testid="stSidebar"] .stSlider {padding-top: 0; margin-top: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧙 Merlin — Balloon Quantity Analyzer")
st.caption("Upload a SOLIDWORKS PDF or image to detect Find Number balloons and tally quantities.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_pdf_pages(file_path: str) -> list[Image.Image]:
    """Render PDF pages as PIL images."""
    if file_path.lower().endswith(".pdf"):
        return convert_from_path(file_path, dpi=150)
    else:
        img = Image.open(file_path)
        return [img]


@st.cache_data(show_spinner=False)
def _cached_analyze(file_bytes: bytes, suffix: str, proximity_radius: float,
                    confidence_threshold: float, custom_phrases_str: str) -> AnalysisReport:
    """Run analysis with caching — only re-runs when inputs change."""
    custom_phrases: dict[str, int] = {}
    for line in custom_phrases_str.strip().splitlines():
        if "=" in line:
            phrase, _, val = line.partition("=")
            try:
                custom_phrases[phrase.strip()] = int(val.strip())
            except ValueError:
                pass

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    config = AnalyzerConfig(
        proximity_radius=proximity_radius,
        confidence_threshold=confidence_threshold,
        custom_multiplier_phrases=custom_phrases,
    )
    analyzer = BalloonAnalyzer(config=config)
    report = analyzer.analyze(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    return report


@st.cache_data(show_spinner=False)
def _cached_render_pages(file_bytes: bytes, suffix: str) -> list[Image.Image]:
    """Render PDF pages with caching — only re-runs when file changes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    pages = _render_pdf_pages(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    return pages


# A palette of distinct highlight colors (RGBA fill, RGBA outline)
_HIGHLIGHT_COLORS: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = [
    ((255, 255, 0, 80), (255, 200, 0, 220)),    # yellow
    ((255, 100, 200, 80), (255, 50, 150, 220)),  # pink
    ((100, 200, 255, 80), (0, 150, 255, 220)),   # blue
    ((100, 255, 100, 80), (0, 200, 0, 220)),     # green
    ((255, 150, 50, 80), (255, 100, 0, 220)),    # orange
    ((200, 100, 255, 80), (150, 50, 255, 220)),  # purple
    ((255, 80, 80, 80), (220, 30, 30, 220)),     # red
    ((0, 255, 200, 80), (0, 200, 150, 220)),     # teal
    ((255, 200, 150, 80), (200, 150, 80, 220)),  # tan
    ((180, 180, 255, 80), (120, 120, 255, 220)), # lavender
]


# Color names matching _HIGHLIGHT_COLORS for display
_COLOR_NAMES = ["🟡", "🩷", "🔵", "🟢", "🟠", "🟣", "🔴", "🩵", "🟤", "💜"]


def _highlight_balloons_on_page(
    page_img: Image.Image,
    report: AnalysisReport,
    page_number: int,
    selected_finds: dict[str, int],
    pdf_to_img_scale: float,
) -> Image.Image:
    """Draw highlight circles on balloons matching the selected Find numbers.

    selected_finds maps Find Number -> color index.
    """
    img = page_img.copy()
    if not selected_finds:
        return img

    draw = ImageDraw.Draw(img, "RGBA")

    for b in report.balloon_breakdown:
        if b.page_number != page_number:
            continue
        if b.find_number not in selected_finds:
            continue

        color_idx = selected_finds[b.find_number] % len(_HIGHLIGHT_COLORS)
        fill_color, outline_color = _HIGHLIGHT_COLORS[color_idx]

        cx = (b.bounding_box.x + b.bounding_box.width / 2) * pdf_to_img_scale
        cy = (b.bounding_box.y + b.bounding_box.height / 2) * pdf_to_img_scale
        r = max(b.bounding_box.width, b.bounding_box.height) / 2 * pdf_to_img_scale
        pad = r * 0.5

        x0 = cx - r - pad
        y0 = cy - r - pad
        x1 = cx + r + pad
        y1 = cy + r + pad
        draw.ellipse([x0, y0, x1, y1], fill=fill_color, outline=outline_color, width=3)

    return img


# ---------------------------------------------------------------------------
# Sidebar: Configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Drawing Import")
    uploaded = st.file_uploader(
        "Upload a drawing (PDF, PNG, JPEG, TIFF)",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
    )
    drawing_status_container = st.container()

    st.divider()
    st.header("BOM Import")
    bom_upload = st.file_uploader(
        "Upload PLM BOM export (Excel)",
        type=["xls", "xlsx"],
        key="bom_upload",
    )
    bom_data: dict[str, dict] = {}
    if bom_upload is not None:
        try:
            bom_data = _parse_bom_excel(bom_upload.getvalue())
            if bom_data:
                st.success(f"Loaded {len(bom_data)} BOM line items.")
            else:
                st.warning("No Level-1 BOM rows found. Check the file format.")
        except Exception as exc:
            st.error(f"Failed to parse BOM: {exc}")

    st.divider()
    proximity_radius = 50.0
    confidence_threshold = 0.5

    # Auto-highlight range slider is rendered after analysis (needs tally max)
    highlight_range_container = st.container()


if uploaded is not None:
    suffix = Path(uploaded.name).suffix
    file_bytes = uploaded.getvalue()

    with st.spinner("Analyzing drawing..."):
        try:
            report = _cached_analyze(
                file_bytes, suffix, proximity_radius,
                confidence_threshold, "",
            )
            page_images = _cached_render_pages(file_bytes, suffix)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    # Determine scale factor: PDF points to rendered image pixels
    # PDF page size is in points (72 dpi), rendered at 150 dpi
    pdf_to_img_scale = 150.0 / 72.0

    with drawing_status_container:
        st.success(f"Found {len(report.balloon_breakdown)} balloons across the drawing.")

    # -----------------------------------------------------------------------
    # Tally table (full width)
    # -----------------------------------------------------------------------

    st.subheader("Tally")

    if report.tally:
        def sort_key(fn: str) -> tuple[int, str]:
            try:
                return (0, fn.zfill(10))
            except ValueError:
                return (1, fn)

        sorted_tally = sorted(report.tally.items(), key=lambda x: sort_key(x[0]))

        max_count = max(count for _, count in sorted_tally)
        with highlight_range_container:
            st.header("Auto-Highlight Range")
            highlight_range: tuple[int, int] = st.slider(
                "Highlight Find Numbers with quantity between:",
                min_value=0,
                max_value=max_count,
                value=(min(10, max_count), max_count),
                step=1,
            )

        # Build tally DataFrame with optional BOM columns
        tally_rows: list[dict] = []
        drawing_fns: set[str] = set()
        for fn, count in sorted_tally:
            drawing_fns.add(fn)
            row: dict = {
                "Highlight": highlight_range[0] <= count <= highlight_range[1],
                "Find Number": fn,
                "Drawing Qty": count,
            }
            if bom_data:
                entry = bom_data.get(fn)
                row["Part Number"] = entry["part"] if entry else None
                bom_val = entry["qty"] if entry else None
                row["BOM Qty"] = bom_val
                if bom_val is not None:
                    delta = count - bom_val
                    row["Delta"] = f"⚠️ {delta}" if delta != 0 else "✅ 0"
                else:
                    row["Delta"] = None
            tally_rows.append(row)

        # Add BOM-only rows (FNs in Excel but not detected in drawing)
        if bom_data:
            for fn, entry in bom_data.items():
                if fn not in drawing_fns and entry["qty"] > 0 and fn != "0":
                    tally_rows.append({
                        "Highlight": False,
                        "Find Number": fn,
                        "Drawing Qty": 0,
                        "Part Number": entry["part"],
                        "BOM Qty": entry["qty"],
                        "Delta": f"⚠️ {-entry['qty']}",
                    })

        # Re-sort after adding BOM-only rows
        tally_rows.sort(key=lambda r: sort_key(r["Find Number"]))

        tally_df = pd.DataFrame(tally_rows)

        # Column config
        col_config: dict = {
            "Highlight": st.column_config.CheckboxColumn("🔍", default=False),
        }
        disabled_cols = ["Find Number", "Drawing Qty"]
        if bom_data:
            col_config["Part Number"] = st.column_config.TextColumn("Part Number")
            col_config["BOM Qty"] = st.column_config.NumberColumn("BOM Qty")
            col_config["Delta"] = st.column_config.TextColumn("Delta Δ")
            disabled_cols += ["Part Number", "BOM Qty", "Delta"]

        edited_df = st.data_editor(
            tally_df,
            hide_index=True,
            use_container_width=True,
            disabled=disabled_cols,
            column_config=col_config,
        )

        # Build a map of selected Find Numbers -> color index
        selected_finds: dict[str, int] = {}
        color_idx = 0
        color_labels: list[str] = []
        for _, row in edited_df.iterrows():
            if row["Highlight"]:
                selected_finds[row["Find Number"]] = color_idx
                color_labels.append(
                    f"{_COLOR_NAMES[color_idx % len(_COLOR_NAMES)]} **{row['Find Number']}**"
                )
                color_idx += 1
        if color_labels:
            st.caption("Highlighting: " + "  ".join(color_labels))
    else:
        st.info("No balloons with Find numbers detected.")
        selected_finds = {}

    if report.excluded_balloon_count > 0:
        st.caption(f"⚠️ {report.excluded_balloon_count} balloon(s) excluded (unreadable)")

    # -----------------------------------------------------------------------
    # Drawing Preview (full width, below tally)
    # -----------------------------------------------------------------------

    st.subheader("Drawing Preview")

    if len(page_images) > 1:
        page_idx = st.slider("Page", 1, len(page_images), 1) - 1
    else:
        page_idx = 0

    page_img = page_images[page_idx]
    highlighted = _highlight_balloons_on_page(
        page_img, report, page_idx + 1, selected_finds, pdf_to_img_scale,
    )
    st.image(highlighted, use_container_width=True)

    # -----------------------------------------------------------------------
    # Balloon Breakdown table
    # -----------------------------------------------------------------------

    with st.expander("Balloon Breakdown", expanded=True):
        rows = []
        for i, b in enumerate(report.balloon_breakdown, 1):
            adj_text = b.adjacent_multiplier_text or ""
            balloon_mult = b.effective_multiplier
            view_mult = 1

            if "(BOTH SIDES)" in adj_text:
                view_mult = 2
                balloon_mult = b.effective_multiplier // view_mult

            rows.append({
                "Item": i,
                "Find Number": b.find_number,
                "Page": b.page_number,
                "Balloon Multiplier": balloon_mult,
                "View Multiplier": view_mult,
                "Total Multiplier": b.effective_multiplier,
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)

    # -----------------------------------------------------------------------
    # Warnings
    # -----------------------------------------------------------------------

    if report.warnings:
        with st.expander(f"Warnings ({len(report.warnings)})", expanded=False):
            for w in report.warnings:
                page_str = f"Page {w.page_number}" if w.page_number else "N/A"
                st.warning(f"[{w.warning_type.value}] {w.message} ({page_str})")

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    st.subheader("Export")
    gen = ReportGenerator()
    tally_result = TallyResult(
        tally=report.tally,
        balloon_breakdown=report.balloon_breakdown,
        excluded_balloon_count=report.excluded_balloon_count,
    )

    col_json, col_tab = st.columns(2)
    with col_json:
        json_output = gen.generate_json(tally_result, report.warnings)
        st.download_button("📥 Download JSON", json_output, file_name="merlin_results.json", mime="application/json")
    with col_tab:
        tabular_output = gen.generate_tabular(tally_result, report.warnings)
        st.download_button("📥 Download Tabular", tabular_output, file_name="merlin_results.txt", mime="text/plain")
