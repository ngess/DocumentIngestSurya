import ctypes
import json
import math
from argparse import ArgumentParser
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as raw

from PIL import ImageDraw, ImageFont
from pydantic import BaseModel
from yaml import safe_load

from surya.inference import SuryaInferenceManager
from surya.layout import LayoutPredictor
from surya.recognition import RecognitionPredictor


def load_yaml(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return safe_load(f)


class PDFOCRConfig(BaseModel):
    file_path: str
    pages: list[tuple[int, int]]
    output_path: str

    @staticmethod
    def from_yaml(file_path: str) -> "PDFOCRConfig":
        return PDFOCRConfig(**load_yaml(file_path))


def load_pdf(
    path: str,
    pages: list[tuple[int, int]],
    scale: float = 2.0,
) -> list[dict]:
    pdf = pdfium.PdfDocument(path)

    images = []

    for start, end in pages:
        for page_index in range(start, end + 1):
            page = pdf[page_index]
            bitmap = page.render(scale=scale)


            page_box = page.get_cropbox()

            words, symbols = extract_text(page)

            images.append({
                "page_index": page_index,
                "image": bitmap.to_pil(),
                "words": words,
                "symbols": symbols,
                "page_box": page_box,
            })
    return images



# PDF font descriptor flag (PDF 32000-1:2008, table 123, bit 7)
FONT_FLAG_ITALIC = 0x40


def extract_symbol(
    textpage: pdfium.PdfTextPage,
    index: int,
    char: str,
) -> dict:
    """Read one character's geometry and font info, in PDF coordinates."""
    origin_x = ctypes.c_double()
    origin_y = ctypes.c_double()
    raw.FPDFText_GetCharOrigin(textpage.raw, index, origin_x, origin_y)

    # Many PDFs set the font size to 1 and scale glyphs with the text
    # matrix, so the effective size is the matrix's vertical scale.
    matrix = raw.FS_MATRIX()
    raw.FPDFText_GetMatrix(textpage.raw, index, matrix)
    font_size = (
        raw.FPDFText_GetFontSize(textpage.raw, index)
        * math.hypot(matrix.c, matrix.d)
    )

    # The loose box spans the font's full ascent/descent, so its height is
    # consistent across glyphs of the same size (unlike the tight glyph box).
    loose = raw.FS_RECTF()
    raw.FPDFText_GetLooseCharBox(textpage.raw, index, loose)

    flags = ctypes.c_int()
    name_length = raw.FPDFText_GetFontInfo(textpage.raw, index, None, 0, flags)
    name_buffer = ctypes.create_string_buffer(name_length)
    raw.FPDFText_GetFontInfo(textpage.raw, index, name_buffer, name_length, flags)

    return {
        "text": char,
        "bbox": textpage.get_charbox(index),
        "loose_bbox": (loose.left, loose.bottom, loose.right, loose.top),
        "origin": (origin_x.value, origin_y.value),
        "font_size": font_size,
        "font_name": name_buffer.value.decode("utf-8", errors="replace"),
        "italic": bool(flags.value & FONT_FLAG_ITALIC),
    }


def extract_text(page: pdfium.PdfPage) -> tuple[list[dict], list[dict]]:
    """Extract words and their individual symbols (non-whitespace characters).

    Words are runs of symbols between whitespace. Each word lists the indices
    of its symbols, and each symbol records the index of its word.
    """
    textpage = page.get_textpage()
    num_chars = textpage.count_chars()

    words = []
    symbols = []
    current = []

    def close_word():
        if not current:
            return

        boxes = [symbols[i]["bbox"] for i in current]

        for i in current:
            symbols[i]["word"] = len(words)

        words.append({
            "text": "".join(symbols[i]["text"] for i in current),
            "bbox": (
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes),
            ),
            "symbols": list(current),
        })
        current.clear()

    for i in range(num_chars):
        char = textpage.get_text_range(index=i, count=1)

        if char.isspace():
            close_word()
        else:
            current.append(len(symbols))
            symbols.append(extract_symbol(textpage, i, char))

    close_word()

    return words, symbols


def write_html_viewer(
    page: dict,
    blocks: list[dict],
    words: list[dict],
    symbols: list[dict],
    output_path: Path,
) -> None:
    image = page["image"]

    width = image.width
    height = image.height

    data = {
        "width": width,
        "height": height,
        "blocks": blocks,
        "words": words,
        "symbols": symbols,
    }

    data_json = json.dumps(data)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">

    <title>Document Layout Debugger</title>

    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: #222;
            color: white;
        }}

        #toolbar {{
            position: sticky;
            top: 0;
            z-index: 1000;

            display: flex;
            gap: 20px;
            align-items: center;

            padding: 10px 16px;

            background: #111;
            border-bottom: 1px solid #555;
        }}

        #viewer {{
            overflow: auto;
            padding: 30px;
        }}

        #page-container {{
            position: relative;

            width: {width}px;
            height: {height}px;

            transform-origin: top left;
        }}

        #page-image {{
            position: absolute;
            left: 0;
            top: 0;

            width: {width}px;
            height: {height}px;
        }}

        #overlay {{
            position: absolute;
            left: 0;
            top: 0;

            width: {width}px;
            height: {height}px;

            pointer-events: none;
        }}

        .surya-box {{
            fill: rgba(255, 0, 0, 0.05);
            stroke: red;
            stroke-width: 2;
            pointer-events: all;
            cursor: pointer;
        }}

        .word-box {{
            fill: rgba(0, 100, 255, 0.05);
            stroke: #0080ff;
            stroke-width: 1;
            pointer-events: all;
            cursor: pointer;
        }}

        .symbol-box {{
            fill: rgba(0, 180, 0, 0.08);
            stroke: #00a000;
            stroke-width: 0.5;
            pointer-events: all;
            cursor: pointer;
        }}

        .surya-label {{
            fill: red;
            font-size: 16px;
            font-weight: bold;

            paint-order: stroke;
            stroke: white;
            stroke-width: 4px;
            stroke-linejoin: round;

            pointer-events: none;
        }}

        .word-label {{
            fill: blue;
            font-size: 9px;

            paint-order: stroke;
            stroke: white;
            stroke-width: 3px;
            stroke-linejoin: round;

            pointer-events: none;
        }}

        #info {{
            margin-left: auto;
            font-family: monospace;
            font-size: 12px;
        }}

        button {{
            cursor: pointer;
        }}
    </style>
</head>

<body>

<div id="toolbar">

    <label>
        <input
            type="checkbox"
            id="toggle-surya"
            checked
        >
        Surya
    </label>

    <label>
        <input
            type="checkbox"
            id="toggle-words"
            checked
        >
        PDF Words
    </label>

    <label>
        <input
            type="checkbox"
            id="toggle-word-labels"
        >
        Word Labels
    </label>

    <label>
        <input
            type="checkbox"
            id="toggle-symbols"
        >
        Symbols
    </label>

    <button id="zoom-in">
        Zoom +
    </button>

    <button id="zoom-out">
        Zoom -
    </button>

    <button id="zoom-reset">
        Reset
    </button>

    <span id="zoom-value">
        100%
    </span>

    <span id="info">
        Click a box
    </span>

</div>

<div id="viewer">

    <div id="page-container">

        <img
            id="page-image"
            src="page.png"
        >

        <svg
            id="overlay"
            viewBox="0 0 {width} {height}"
        >
        </svg>

    </div>

</div>


<script>

const data = {data_json};

const svg = document.getElementById("overlay");

const SVG_NS = "http://www.w3.org/2000/svg";


// ---------------------------------------------------------
// Surya group
// ---------------------------------------------------------

const suryaGroup = document.createElementNS(
    SVG_NS,
    "g"
);

suryaGroup.id = "surya-group";

svg.appendChild(suryaGroup);


// ---------------------------------------------------------
// Word group
// ---------------------------------------------------------

const wordGroup = document.createElementNS(
    SVG_NS,
    "g"
);

wordGroup.id = "word-group";

svg.appendChild(wordGroup);


// ---------------------------------------------------------
// Symbol group (drawn last so symbols sit above words)
// ---------------------------------------------------------

const symbolGroup = document.createElementNS(
    SVG_NS,
    "g"
);

symbolGroup.id = "symbol-group";

symbolGroup.style.display = "none";

svg.appendChild(symbolGroup);


// ---------------------------------------------------------
// Draw Surya blocks
// ---------------------------------------------------------

for (const block of data.blocks) {{

    const [x1, y1, x2, y2] = block.bbox;

    const rect = document.createElementNS(
        SVG_NS,
        "rect"
    );

    rect.setAttribute("x", x1);
    rect.setAttribute("y", y1);

    rect.setAttribute(
        "width",
        x2 - x1
    );

    rect.setAttribute(
        "height",
        y2 - y1
    );

    rect.setAttribute(
        "class",
        "surya-box"
    );

    rect.addEventListener(
        "click",
        () => {{
            document.getElementById(
                "info"
            ).textContent =
                `Surya | ${{block.position}} | ${{block.label}} | ${{JSON.stringify(block.bbox)}}`;
        }}
    );

    suryaGroup.appendChild(rect);


    const text = document.createElementNS(
        SVG_NS,
        "text"
    );

    text.setAttribute("x", x1 + 3);
    text.setAttribute("y", y1 - 4);

    text.setAttribute(
        "class",
        "surya-label"
    );

    text.textContent =
        `${{block.position}}: ${{block.label}}`;

    suryaGroup.appendChild(text);
}}


// ---------------------------------------------------------
// Draw PDFium words
// ---------------------------------------------------------

for (const word of data.words) {{

    const [x1, y1, x2, y2] = word.bbox;

    const rect = document.createElementNS(
        SVG_NS,
        "rect"
    );

    rect.setAttribute("x", x1);
    rect.setAttribute("y", y1);

    rect.setAttribute(
        "width",
        x2 - x1
    );

    rect.setAttribute(
        "height",
        y2 - y1
    );

    rect.setAttribute(
        "class",
        "word-box"
    );

    rect.addEventListener(
        "click",
        () => {{
            document.getElementById(
                "info"
            ).textContent =
                `PDFium | "${{word.text}}" | block: ${{word.block ?? "none"}} | symbols: ${{word.symbols.length}} | image=${{JSON.stringify(word.bbox)}} | pdf=${{JSON.stringify(word.pdf_bbox)}}`;
        }}
    );

    wordGroup.appendChild(rect);


    const text = document.createElementNS(
        SVG_NS,
        "text"
    );

    text.setAttribute("x", x1);
    text.setAttribute("y", y1 - 2);

    text.setAttribute(
        "class",
        "word-label"
    );

    text.textContent = word.text;

    text.style.display = "none";

    wordGroup.appendChild(text);
}}


// ---------------------------------------------------------
// Draw PDFium symbols
// ---------------------------------------------------------

for (const symbol of data.symbols) {{

    const [x1, y1, x2, y2] = symbol.bbox;

    const rect = document.createElementNS(
        SVG_NS,
        "rect"
    );

    rect.setAttribute("x", x1);
    rect.setAttribute("y", y1);

    rect.setAttribute(
        "width",
        x2 - x1
    );

    rect.setAttribute(
        "height",
        y2 - y1
    );

    rect.setAttribute(
        "class",
        "symbol-box"
    );

    rect.addEventListener(
        "click",
        () => {{
            const word = data.words[symbol.word];

            document.getElementById(
                "info"
            ).textContent =
                `Symbol | "${{symbol.text}}" | word: "${{word.text}}" | block: ${{symbol.block ?? "none"}} | size: ${{symbol.font_size.toFixed(2)}}pt | baseline y: ${{symbol.origin[1].toFixed(1)}} | ${{symbol.font_name}}${{symbol.italic ? " (italic)" : ""}}`;
        }}
    );

    symbolGroup.appendChild(rect);
}}


// ---------------------------------------------------------
// Layer controls
// ---------------------------------------------------------

document
    .getElementById("toggle-surya")
    .addEventListener(
        "change",
        event => {{
            suryaGroup.style.display =
                event.target.checked
                    ? ""
                    : "none";
        }}
    );


document
    .getElementById("toggle-words")
    .addEventListener(
        "change",
        event => {{
            wordGroup
                .querySelectorAll(".word-box")
                .forEach(element => {{
                    element.style.display =
                        event.target.checked
                            ? ""
                            : "none";
                }});
        }}
    );


document
    .getElementById("toggle-symbols")
    .addEventListener(
        "change",
        event => {{
            symbolGroup.style.display =
                event.target.checked
                    ? ""
                    : "none";
        }}
    );


document
    .getElementById("toggle-word-labels")
    .addEventListener(
        "change",
        event => {{
            wordGroup
                .querySelectorAll(".word-label")
                .forEach(element => {{
                    element.style.display =
                        event.target.checked
                            ? ""
                            : "none";
                }});
        }}
    );


// ---------------------------------------------------------
// Zoom
// ---------------------------------------------------------

let zoom = 1.0;

const container =
    document.getElementById("page-container");

const zoomValue =
    document.getElementById("zoom-value");


function updateZoom() {{

    container.style.transform =
        `scale(${{zoom}})`;

    zoomValue.textContent =
        `${{Math.round(zoom * 100)}}%`;
}}


document
    .getElementById("zoom-in")
    .addEventListener(
        "click",
        () => {{
            zoom = Math.min(
                zoom + 0.25,
                4.0
            );

            updateZoom();
        }}
    );


document
    .getElementById("zoom-out")
    .addEventListener(
        "click",
        () => {{
            zoom = Math.max(
                zoom - 0.25,
                0.25
            );

            updateZoom();
        }}
    );


document
    .getElementById("zoom-reset")
    .addEventListener(
        "click",
        () => {{
            zoom = 1.0;
            updateZoom();
        }}
    );


// ---------------------------------------------------------
// Page navigation (forwarded to index.html when embedded)
// ---------------------------------------------------------

document.addEventListener(
    "keydown",
    event => {{
        if (window.parent === window) {{
            return;
        }}

        if (event.key === "ArrowLeft") {{
            window.parent.postMessage({{ nav: -1 }}, "*");
        }}

        if (event.key === "ArrowRight") {{
            window.parent.postMessage({{ nav: 1 }}, "*");
        }}
    }}
);

</script>

</body>
</html>
"""

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(html)



def write_layout_results(
    pages: list[dict],
    layouts,
    output_path: str,
) -> None:
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for page, layout in zip(pages, layouts):
        page_index = page["page_index"]
        image = page["image"]

        # ---------------------------------------------------------
        # Create page directory
        # ---------------------------------------------------------
        page_dir = output_dir / f"page_{page_index:04d}"
        page_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------
        # Save original rendered page
        # ---------------------------------------------------------
        image.save(page_dir / "page.png")

        # ---------------------------------------------------------
        # Sort Surya blocks by predicted reading order
        # ---------------------------------------------------------
        ordered_blocks = sorted(
            layout.bboxes,
            key=lambda block: block.position,
        )

        # ---------------------------------------------------------
        # Build structured results
        # ---------------------------------------------------------
        page_words = assign_words_to_blocks(
            ordered_blocks,
            page["words"],
            page["page_box"],
            image.size,
        )

        page_symbols = symbols_to_image(
            page["symbols"],
            page_words,
            page["page_box"],
            image.size,
        )

        blocks = []

        for block in ordered_blocks:
            words = [
                {
                    "id": word["id"],
                    "text": word["text"],
                    "bbox": word["bbox"],
                    "pdf_bbox": word["pdf_bbox"],
                    "symbols": word["symbols"],
                }
                for word in page_words
                if word["block"] == block.position
            ]

            blocks.append({
                "position": block.position,
                "label": block.label,
                "bbox": list(block.bbox),
                "polygon": [
                    list(point)
                    for point in block.polygon
                ],
                "confidence": block.confidence,
                "words": words,
                "text": " ".join(
                    word["text"]
                    for word in words
                ),
            })

        results = {
            "page_index": page_index,
            "width": image.width,
            "height": image.height,
            "page_box": list(page["page_box"]),
            "blocks": blocks,
            "symbols": page_symbols,
        }

        # ---------------------------------------------------------
        # Save JSON
        # ---------------------------------------------------------
        with open(
            page_dir / "results.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(results, f, indent=2)

        # =========================================================
        # Visualization 1:
        # Surya layout + reading order
        # =========================================================

        annotated = image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)

        try:
            block_font = ImageFont.truetype(
                "Arial.ttf",
                24,
            )
        except OSError:
            block_font = ImageFont.load_default()

        for block in ordered_blocks:
            x1, y1, x2, y2 = block.bbox

            # Surya bounding box
            draw.rectangle(
                (x1, y1, x2, y2),
                outline="red",
                width=3,
            )

            label = (
                f"{block.position}: "
                f"{block.label}"
            )

            text_bbox = draw.textbbox(
                (0, 0),
                label,
                font=block_font,
            )

            text_width = (
                text_bbox[2]
                - text_bbox[0]
            )

            text_height = (
                text_bbox[3]
                - text_bbox[1]
            )

            label_y = max(
                0,
                y1 - text_height - 8,
            )

            # Label background
            draw.rectangle(
                (
                    x1,
                    label_y,
                    x1 + text_width + 8,
                    label_y + text_height + 8,
                ),
                fill="white",
            )

            # Label
            draw.text(
                (
                    x1 + 4,
                    label_y + 4,
                ),
                label,
                fill="red",
                font=block_font,
            )

        annotated.save(
            page_dir / "annotated.png"
        )

        # =========================================================
        # Visualization 2:
        # Surya regions + individual PDFium words
        # =========================================================

        detailed = image.copy().convert("RGB")
        draw = ImageDraw.Draw(detailed)

        try:
            block_font = ImageFont.truetype(
                "Arial.ttf",
                24,
            )

            word_font = ImageFont.truetype(
                "Arial.ttf",
                10,
            )

        except OSError:
            block_font = ImageFont.load_default()
            word_font = ImageFont.load_default()

        # ---------------------------------------------------------
        # Draw Surya regions
        # ---------------------------------------------------------
        for block in ordered_blocks:
            x1, y1, x2, y2 = block.bbox

            draw.rectangle(
                (x1, y1, x2, y2),
                outline="red",
                width=3,
            )

            label = (
                f"{block.position}: "
                f"{block.label}"
            )

            text_bbox = draw.textbbox(
                (0, 0),
                label,
                font=block_font,
            )

            text_width = (
                text_bbox[2]
                - text_bbox[0]
            )

            text_height = (
                text_bbox[3]
                - text_bbox[1]
            )

            label_y = max(
                0,
                y1 - text_height - 8,
            )

            draw.rectangle(
                (
                    x1,
                    label_y,
                    x1 + text_width + 8,
                    label_y + text_height + 8,
                ),
                fill="white",
            )

            draw.text(
                (
                    x1 + 4,
                    label_y + 4,
                ),
                label,
                fill="red",
                font=block_font,
            )

        # ---------------------------------------------------------
        # Draw PDFium word boxes
        # ---------------------------------------------------------
        for word in page["words"]:
            word_bbox = pdf_bbox_to_image_bbox(
                word["bbox"],
                page["page_box"],
                image.size,
            )

            x1, y1, x2, y2 = word_bbox

            # Individual word box
            draw.rectangle(
                (x1, y1, x2, y2),
                outline="blue",
                width=1,
            )

            word_text = word["text"]

            text_bbox = draw.textbbox(
                (0, 0),
                word_text,
                font=word_font,
            )

            text_width = (
                text_bbox[2]
                - text_bbox[0]
            )

            text_height = (
                text_bbox[3]
                - text_bbox[1]
            )

            # Put extracted word immediately above
            # the corresponding PDF word.
            text_y = max(
                0,
                y1 - text_height - 2,
            )

            # White background for readability
            draw.rectangle(
                (
                    x1,
                    text_y,
                    x1 + text_width + 2,
                    text_y + text_height + 2,
                ),
                fill="white",
            )

            draw.text(
                (
                    x1 + 1,
                    text_y + 1,
                ),
                word_text,
                fill="blue",
                font=word_font,
            )

        detailed.save(
            page_dir / "annotated_detailed.png"
        )
        write_html_viewer(
            page,
            blocks,
            page_words,
            page_symbols,
            page_dir / "viewer.html",
        )


def write_index_html(output_path: str) -> None:
    output_dir = Path(output_path)

    # Include every page directory, not just this run's, so pages from
    # earlier runs into the same output directory stay navigable.
    page_dirs = sorted(
        path.name
        for path in output_dir.glob("page_*")
        if (path / "viewer.html").exists()
    )

    pages_json = json.dumps(page_dirs)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">

    <title>Document Layout Debugger</title>

    <style>
        html, body {{
            height: 100%;
            margin: 0;
        }}

        body {{
            display: flex;
            flex-direction: column;

            font-family: Arial, sans-serif;
            background: #222;
            color: white;
        }}

        #nav {{
            display: flex;
            gap: 12px;
            align-items: center;

            padding: 8px 16px;

            background: #000;
            border-bottom: 1px solid #555;
        }}

        #frame {{
            flex: 1;
            width: 100%;
            border: none;
        }}

        button, select {{
            cursor: pointer;
        }}
    </style>
</head>

<body>

<div id="nav">

    <button id="prev">
        &larr; Back
    </button>

    <select id="page-select"></select>

    <button id="next">
        Next &rarr;
    </button>

    <span id="page-count"></span>

</div>

<iframe id="frame"></iframe>


<script>

const pages = {pages_json};

const frame = document.getElementById("frame");
const select = document.getElementById("page-select");
const pageCount = document.getElementById("page-count");

let current = Math.max(
    0,
    pages.indexOf(location.hash.slice(1))
);


for (const [i, name] of pages.entries()) {{
    const option = document.createElement("option");

    option.value = i;
    option.textContent = name;

    select.appendChild(option);
}}


function show(index) {{
    current = Math.min(
        Math.max(index, 0),
        pages.length - 1
    );

    frame.src = `${{pages[current]}}/viewer.html`;
    select.value = current;
    pageCount.textContent = `${{current + 1}} / ${{pages.length}}`;

    history.replaceState(null, "", `#${{pages[current]}}`);

    document.getElementById("prev").disabled = current === 0;
    document.getElementById("next").disabled =
        current === pages.length - 1;
}}


document
    .getElementById("prev")
    .addEventListener("click", () => show(current - 1));

document
    .getElementById("next")
    .addEventListener("click", () => show(current + 1));

select.addEventListener(
    "change",
    () => show(Number(select.value))
);

document.addEventListener(
    "keydown",
    event => {{
        if (event.key === "ArrowLeft") show(current - 1);
        if (event.key === "ArrowRight") show(current + 1);
    }}
);

// Arrow keys pressed while the viewer iframe has focus
window.addEventListener(
    "message",
    event => {{
        if (event.data && event.data.nav) {{
            show(current + event.data.nav);
        }}
    }}
);

show(current);

</script>

</body>
</html>
"""

    with open(
        output_dir / "index.html",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(html)


def pdf_bbox_to_image_bbox(
    bbox: tuple[float, float, float, float],
    page_box: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    left, bottom, right, top = bbox

    # The rendered image covers the crop box, whose origin may not be (0, 0).
    box_left, box_bottom, box_right, box_top = page_box
    image_width, image_height = image_size

    scale_x = image_width / (box_right - box_left)
    scale_y = image_height / (box_top - box_bottom)

    x1 = (left - box_left) * scale_x
    x2 = (right - box_left) * scale_x

    # PDF y-axis points upward; image y-axis points downward.
    y1 = (box_top - top) * scale_y
    y2 = (box_top - bottom) * scale_y

    return x1, y1, x2, y2

# Surya boxes sometimes clip the last line of a block, so words just
# outside a box (in image pixels) still count as belonging to it.
BLOCK_PADDING = 10


def bbox_contains_word(
    block_bbox,
    word_bbox,
    padding: float = 0,
) -> bool:
    x1, y1, x2, y2 = block_bbox
    wx1, wy1, wx2, wy2 = word_bbox

    center_x = (wx1 + wx2) / 2
    center_y = (wy1 + wy2) / 2

    return (
        x1 - padding <= center_x <= x2 + padding
        and
        y1 - padding <= center_y <= y2 + padding
    )


def bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def assign_words_to_blocks(
    blocks,
    words: list[dict],
    page_box,
    image_size,
    padding: float = BLOCK_PADDING,
) -> list[dict]:
    """Convert words to image coordinates and assign each to at most one block.

    A block that strictly contains the word's center wins over one that only
    contains it after padding. Ties go to the smallest block, so an equation
    nested inside a text block keeps its own words.
    """
    assigned = []

    for word_id, word in enumerate(words):
        image_bbox = pdf_bbox_to_image_bbox(
            word["bbox"],
            page_box,
            image_size,
        )

        candidates = [
            block
            for block in blocks
            if bbox_contains_word(block.bbox, image_bbox)
        ] or [
            block
            for block in blocks
            if bbox_contains_word(block.bbox, image_bbox, padding)
        ]

        owner = min(
            candidates,
            key=lambda block: bbox_area(block.bbox),
            default=None,
        )

        assigned.append({
            "id": word_id,
            "text": word["text"],
            "bbox": list(image_bbox),
            "pdf_bbox": list(word["bbox"]),
            "block": owner.position if owner else None,
            "symbols": word["symbols"],
        })

    return assigned


def pdf_point_to_image_point(
    point: tuple[float, float],
    page_box: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[float, float]:
    x, y, _, _ = pdf_bbox_to_image_bbox(
        (point[0], point[1], point[0], point[1]),
        page_box,
        image_size,
    )
    return x, y


def symbols_to_image(
    symbols: list[dict],
    words: list[dict],
    page_box,
    image_size,
) -> list[dict]:
    """Convert symbols to image coordinates; each inherits its word's block."""
    converted = []

    for symbol_id, symbol in enumerate(symbols):
        origin = pdf_point_to_image_point(
            symbol["origin"],
            page_box,
            image_size,
        )

        converted.append({
            "id": symbol_id,
            "text": symbol["text"],
            "bbox": list(pdf_bbox_to_image_bbox(
                symbol["bbox"],
                page_box,
                image_size,
            )),
            "pdf_bbox": list(symbol["bbox"]),
            "loose_bbox": list(pdf_bbox_to_image_bbox(
                symbol["loose_bbox"],
                page_box,
                image_size,
            )),
            # Glyph origin: the pen position on the baseline.
            "origin": list(origin),
            "pdf_origin": list(symbol["origin"]),
            "font_size": symbol["font_size"],
            "font_name": symbol["font_name"],
            "italic": symbol["italic"],
            "word": symbol["word"],
            "block": words[symbol["word"]]["block"],
        })

    return converted

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    config = PDFOCRConfig.from_yaml(args.config)

    pages = load_pdf(
        config.file_path,
        config.pages,
    )

    images = [
        page["image"]
        for page in pages
    ]

    manager = SuryaInferenceManager()

    layout_predictor = LayoutPredictor(manager)
    recognition_predictor = RecognitionPredictor(manager)

    layouts = layout_predictor(images)

    write_layout_results(
        pages,
        layouts,
        config.output_path,
    )

    write_index_html(config.output_path)

    for page, layout in zip(pages, layouts):
        print(
            f"Processed PDF page index: "
            f"{page['page_index']}"
        )