#!/usr/bin/env python
"""Generate an animated, terminal-style GIF demoing data-wrangler.

This script actually *runs* real ``datawrangler`` calls (no faked or
hand-typed outputs) and renders the typed commands plus their real
results as a dark, monospace, terminal-style animation using Pillow.

The resulting GIF is meant to live at ``docs/images/demo.gif`` and be
embedded near the top of ``README.rst`` -- suitable for the README and
for sharing on social media.

Usage::

    python scripts/make_demo_gif.py
"""
import os
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import datawrangler as dw

# ---------------------------------------------------------------------------
# Layout / style constants
# ---------------------------------------------------------------------------
WIDTH = 900
HEIGHT = 560
TITLEBAR_HEIGHT = 36
PADDING = 16
FONT_SIZE = 15
LINE_HEIGHT = 21
MAX_VISIBLE_LINES = (HEIGHT - TITLEBAR_HEIGHT - 2 * PADDING) // LINE_HEIGHT

BG_COLOR = (30, 30, 36)
TITLEBAR_COLOR = (48, 48, 56)
TITLE_TEXT_COLOR = (170, 170, 180)
PROMPT_COLOR = (98, 222, 137)
TEXT_COLOR = (225, 225, 230)
OUTPUT_COLOR = (150, 200, 255)
CURSOR_COLOR = PROMPT_COLOR
DOT_RED = (255, 95, 86)
DOT_YELLOW = (255, 189, 46)
DOT_GREEN = (39, 201, 63)

TYPE_MS = 35
PAUSE_MS = 350
REVEAL_MS = 350
HOLD_MS = 2200
INTRO_MS = 900
OUTRO_MS = 2500

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "images", "demo.gif"
)


def find_monospace_font(size):
    """Locate a monospace TTF, falling back to Pillow's default font."""
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
        "/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def run_demos():
    """Actually execute real datawrangler calls and capture their real results."""
    demos = []

    # 1. Arrays become DataFrames automatically.
    array_input = np.array([[1, 2, 3], [4, 5, 6]])
    df = dw.wrangle(array_input)
    demos.append(
        {
            "command": ">>> dw.wrangle(np.array([[1, 2, 3], [4, 5, 6]]))",
            "output": str(df),
        }
    )

    # 2. High-performance Polars backend for large arrays.
    large_array = np.random.rand(50000, 20)
    start = time.time()
    polars_df = dw.wrangle(large_array, backend="polars")
    elapsed_ms = (time.time() - start) * 1000
    demos.append(
        {
            "command": ">>> dw.wrangle(np.random.rand(50000, 20), backend='polars')",
            "output": (
                "{}.{}  shape={}  ({:.1f} ms)".format(
                    type(polars_df).__module__.split(".")[0],
                    type(polars_df).__name__,
                    polars_df.shape,
                    elapsed_ms,
                )
            ),
        }
    )

    # 3. Text -> sentence embeddings.
    sentences = ["hi there", "data wrangler rocks"]
    embeddings = dw.wrangle(sentences, text_kwargs={"model": "all-MiniLM-L6-v2"})
    demos.append(
        {
            "command": (
                ">>> dw.wrangle(['hi there', 'data wrangler rocks'],\n"
                "...           text_kwargs={'model': 'all-MiniLM-L6-v2'})"
            ),
            "output": "{} of shape {}  # sentence embeddings".format(
                type(embeddings).__name__, embeddings.shape
            ),
        }
    )

    # 4. @dw.decorate.funnel: write functions as if inputs are DataFrames.
    @dw.decorate.funnel
    def n_rows(data):
        return data.shape[0]

    funnel_result = n_rows(np.array([[1, 2], [3, 4], [5, 6]]))
    demos.append(
        {
            "command": (
                ">>> @dw.decorate.funnel\n"
                "... def n_rows(data):\n"
                "...     return data.shape[0]\n"
                ">>> n_rows(np.array([[1, 2], [3, 4], [5, 6]]))"
            ),
            "output": repr(funnel_result),
        }
    )

    # 5. Stack a list of DataFrames into one, then unstack it back.
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
    stacked = dw.stack([df1, df2])
    unstacked = dw.unstack(stacked)
    demos.append(
        {
            "command": ">>> stacked = dw.stack([df1, df2])\n>>> dw.unstack(stacked)",
            "output": "stacked.shape={}  ->  unstacked into {} DataFrame(s)".format(
                stacked.shape, len(unstacked)
            ),
        }
    )

    return demos


def draw_terminal_line(draw, font, x, y, text, kind):
    """Draw a single terminal line and return the x position after it."""
    prefixes = (">>> ", "... ")
    if kind == "prompt" and text[:4] in prefixes:
        prefix, rest = text[:4], text[4:]
        draw.text((x, y), prefix, font=font, fill=PROMPT_COLOR)
        prefix_w = draw.textlength(prefix, font=font)
        draw.text((x + prefix_w, y), rest, font=font, fill=TEXT_COLOR)
        return x + prefix_w + draw.textlength(rest, font=font)
    color = OUTPUT_COLOR if kind == "output" else TEXT_COLOR
    draw.text((x, y), text, font=font, fill=color)
    return x + draw.textlength(text, font=font)


def render_frame(font, title_font, history, current_line=None, cursor=False):
    """Render one terminal-window frame from committed history plus an optional in-progress line."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, WIDTH, TITLEBAR_HEIGHT], fill=TITLEBAR_COLOR)
    for i, dot_color in enumerate((DOT_RED, DOT_YELLOW, DOT_GREEN)):
        cx = 22 + i * 22
        cy = TITLEBAR_HEIGHT // 2
        draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=dot_color)
    title = "python3 -- data-wrangler demo"
    title_w = draw.textlength(title, font=title_font)
    draw.text(((WIDTH - title_w) / 2, (TITLEBAR_HEIGHT - FONT_SIZE) / 2), title, font=title_font, fill=TITLE_TEXT_COLOR)

    lines = list(history)
    if current_line is not None:
        lines.append(current_line)
    visible = lines[-MAX_VISIBLE_LINES:]

    y = TITLEBAR_HEIGHT + PADDING
    end_x = PADDING
    for text, kind in visible:
        end_x = draw_terminal_line(draw, font, PADDING, y, text, kind)
        y += LINE_HEIGHT

    if cursor and current_line is not None:
        cursor_y = y - LINE_HEIGHT
        draw.rectangle([end_x + 3, cursor_y + 2, end_x + 11, cursor_y + FONT_SIZE + 2], fill=CURSOR_COLOR)

    return img


def build_frames(demos, font, title_font):
    frames = []
    durations = []
    history = []

    def add_frame(current_line=None, cursor=False, duration=TYPE_MS):
        frames.append(render_frame(font, title_font, history, current_line, cursor))
        durations.append(duration)

    add_frame(duration=INTRO_MS)

    for demo in demos:
        for command_line in demo["command"].split("\n"):
            step = max(1, len(command_line) // 10)
            for end in range(step, len(command_line), step):
                add_frame(current_line=(command_line[:end], "prompt"), cursor=True, duration=TYPE_MS)
            add_frame(current_line=(command_line, "prompt"), cursor=True, duration=TYPE_MS)
            history.append((command_line, "prompt"))

        add_frame(duration=PAUSE_MS)

        output_lines = demo["output"].split("\n")
        for i, output_line in enumerate(output_lines):
            history.append((output_line, "output"))
            is_last = i == len(output_lines) - 1
            add_frame(duration=HOLD_MS if is_last else REVEAL_MS)

    history.append(("", "output"))
    history.append(("# pip install pydata-wrangler", "output"))
    add_frame(duration=OUTRO_MS)

    return frames, durations


def main():
    demos = run_demos()

    font = find_monospace_font(FONT_SIZE)
    title_font = find_monospace_font(FONT_SIZE)

    frames, durations = build_frames(demos, font, title_font)

    palette_frames = [frame.convert("P", palette=Image.ADAPTIVE, colors=96) for frame in frames]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    palette_frames[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=palette_frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print("Wrote {} frames ({:.0f} KB) to {}".format(len(palette_frames), size_kb, OUTPUT_PATH))


if __name__ == "__main__":
    main()
