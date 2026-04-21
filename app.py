"""
app.py - ScribblBot inference application.
"""

import sys
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from config import CLASSES, CLASS_EMOJIS, MODELS_DIR, NUM_CLASSES
from scripts.model import ScribblNet


def _load_model() -> tuple[ScribblNet, torch.device]:
    """Load trained ScribblNet weights from disk."""
    device = torch.device("cpu")
    model_path = MODELS_DIR / "deep_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Weights not found at {model_path}. Run python setup.py first.")
    model = ScribblNet(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


MODEL, DEVICE = _load_model()


def predict(sketch: Optional[dict], _counter: int) -> tuple[str, int]:
    """Run inference on an ImageEditor drawing.

    Args:
        sketch: Dict from gr.ImageEditor with 'composite' key.
        _counter: Click counter used to bust Gradio output caching.

    Returns:
        Tuple of (HTML results string, incremented counter).
    """
    _counter += 1
    if sketch is None:
        return _empty_state_html(), _counter

    img_array = sketch.get("composite") if isinstance(sketch, dict) else sketch
    if img_array is None:
        return _empty_state_html(), _counter

    try:
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        if img_pil.mode == "RGBA":
            white = Image.new("RGBA", img_pil.size, (248, 247, 242, 255))
            img_pil = Image.alpha_composite(white, img_pil).convert("L")
        else:
            img_pil = img_pil.convert("L")
        img_pil = img_pil.resize((28, 28), Image.LANCZOS)
        arr = np.array(img_pil, dtype=np.float32)
        arr = (255.0 - arr) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)
    except Exception as exc:
        return _error_html(str(exc)), _counter

    with torch.no_grad():
        probs = F.softmax(MODEL(tensor), dim=1)[0].cpu().numpy()

    top = [(CLASSES[i], float(probs[i])) for i in np.argsort(probs)[::-1][:5]]
    return _results_html(top), _counter


def _results_html(top: list[tuple[str, float]]) -> str:
    best_cls, best_prob = top[0]
    conf_pct = best_prob * 100
    label = "CONFIDENT" if best_prob > 0.7 else "LIKELY" if best_prob > 0.4 else "UNSURE"
    bars = ""
    for i, (cls, prob) in enumerate(top):
        pct = prob * 100
        bars += f"""
        <div class="bar-row" style="animation-delay:{i*0.08}s">
            <span class="bar-emoji">{CLASS_EMOJIS.get(cls,'')}</span>
            <span class="bar-label">{cls.upper()}</span>
            <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;animation-delay:{i*0.08+0.1}s"></div></div>
            <span class="bar-pct">{pct:.1f}%</span>
        </div>"""
    return f"""
    <div class="results-panel fade-in">
        <div class="result-tag">[ PREDICTION ]</div>
        <div class="top-result">
            <span class="top-emoji">{CLASS_EMOJIS.get(best_cls,'?')}</span>
            <div class="top-text">
                <div class="top-label">{best_cls.upper()}</div>
                <div class="top-conf">{conf_pct:.1f}% &nbsp;·&nbsp; {label}</div>
            </div>
        </div>
        <div class="divider"></div>
        <div class="section-label">TOP 5 PROBABILITIES</div>
        {bars}
    </div>"""


def _empty_state_html() -> str:
    return """
    <div class="results-panel empty-state">
        <div class="empty-icon">✏️</div>
        <div class="empty-title">DRAW SOMETHING</div>
        <div class="empty-sub">then hit ANALYZE</div>
        <div class="class-pills">
            <span class="pill">🐱 cat</span><span class="pill">🐶 dog</span>
            <span class="pill">🍕 pizza</span><span class="pill">🚲 bicycle</span>
            <span class="pill">🏠 house</span><span class="pill">☀️ sun</span>
            <span class="pill">🌳 tree</span><span class="pill">🚗 car</span>
            <span class="pill">🐟 fish</span><span class="pill">🦋 butterfly</span>
            <span class="pill">🎸 guitar</span><span class="pill">🍔 hamburger</span>
            <span class="pill">✈️ airplane</span><span class="pill">🍌 banana</span>
            <span class="pill">⭐ star</span>
        </div>
    </div>"""


def _error_html(msg: str) -> str:
    return f'<div class="results-panel error-state"><p class="err-msg">⚠ {msg}</p></div>'


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=VT323&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg:        #080808;
    --surface:   #111111;
    --surface2:  #1a1a1a;
    --border:    #2a2a2a;
    --accent:    #b8ff57;
    --text:      #e8e8e0;
    --text-muted:#888880;
    --red:       #ff5f57;
    --mono:      'IBM Plex Mono', monospace;
    --display:   'VT323', monospace;
}
body, .gradio-container, #root {
    background: var(--bg) !important;
    font-family: var(--mono) !important;
    color: var(--text) !important;
}
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
footer { display: none !important; }
.block, .gr-box { background: transparent !important; border: none !important; box-shadow: none !important; }

.app-header { text-align: center; padding: 36px 20px 20px; border-bottom: 1px solid var(--border); margin-bottom: 28px; }
.app-title  { font-family: var(--display); font-size: 72px; line-height: 1; color: var(--accent); letter-spacing: 6px; text-shadow: 0 0 30px rgba(184,255,87,0.3); margin: 0; }
.app-subtitle { font-size: 12px; color: var(--text-muted); letter-spacing: 4px; margin-top: 6px; }

/* ImageEditor styling */
/* Override Gradio's orange accent with our green */
.sketch-col { --color-accent: #b8ff57 !important; --color-accent-soft: rgba(184,255,87,0.15) !important; }
.sketch-col .image-editor { border: 1px solid var(--border) !important; border-radius: 4px !important; background: var(--surface) !important; }
/* Hide color picker and swatch - we only need pen and eraser */
.sketch-col [aria-label="Color"],
.sketch-col [title="Color"],
.sketch-col .image-editor .toolbar > button:nth-child(3),
.sketch-col .image-editor .toolbar > button:nth-child(4) { display: none !important; }
/* Toolbar background */
.sketch-col .image-editor > div { background: var(--surface2) !important; }
/* All buttons */
.sketch-col .image-editor button {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    margin: 2px !important;
    color: var(--text) !important;
}
.sketch-col .image-editor button:hover {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #000 !important;
}
/* Active tool */
.sketch-col .image-editor button[aria-pressed="true"] {
    border: 2px solid var(--accent) !important;
    background: rgba(184,255,87,0.15) !important;
    color: var(--accent) !important;
}
/* Force all SVG icons to white/text color */
.sketch-col .image-editor svg *  { color: inherit !important; stroke: currentColor !important; }
.sketch-col [data-testid="layer-wrap"] { display: none !important; }
.sketch-col .layers-panel { display: none !important; }
/* White canvas */
.sketch-col .konvajs-content,
.sketch-col .konvajs-content canvas,
.sketch-col canvas { background: #f8f7f2 !important; background-color: #f8f7f2 !important; }
.sketch-col canvas { cursor: crosshair !important; }
.sketch-col * { cursor: auto; }
.sketch-col canvas { cursor: crosshair !important; }

.results-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 20px; min-height: 420px; font-family: var(--mono); }
.result-tag { font-size: 11px; color: var(--accent); letter-spacing: 3px; margin-bottom: 16px; }
.top-result { display: flex; align-items: center; gap: 18px; margin-bottom: 18px; }
.top-emoji  { font-size: 56px; line-height: 1; }
.top-label  { font-family: var(--display); font-size: 52px; color: var(--text); line-height: 1; letter-spacing: 3px; }
.top-conf   { font-size: 13px; color: var(--accent); margin-top: 4px; }
.divider    { height: 1px; background: var(--border); margin: 16px 0; }
.section-label { font-size: 10px; color: var(--text-muted); letter-spacing: 3px; margin-bottom: 12px; }
.bar-row { display: grid; grid-template-columns: 28px 90px 1fr 50px; align-items: center; gap: 8px; margin-bottom: 10px; opacity: 0; animation: slideIn 0.3s ease forwards; }
.bar-emoji { font-size: 16px; text-align: center; }
.bar-label { font-size: 11px; color: var(--text-muted); letter-spacing: 1px; }
.bar-track { height: 6px; background: var(--surface2); border-radius: 3px; overflow: hidden; }
.bar-fill  { height: 100%; background: var(--accent); border-radius: 3px; width: 0; animation: barGrow 0.4s ease forwards; }
.bar-pct   { font-size: 11px; color: var(--text); text-align: right; }

.empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 360px; }
.empty-icon  { font-size: 48px; margin-bottom: 12px; }
.empty-title { font-family: var(--display); font-size: 36px; color: var(--accent); letter-spacing: 3px; }
.empty-sub   { font-size: 12px; color: var(--text-muted); margin: 4px 0 24px; letter-spacing: 2px; }
.class-pills { display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; max-width: 340px; }
.pill { background: var(--surface2); border: 1px solid var(--border); padding: 3px 10px; border-radius: 20px; font-size: 11px; color: var(--text-muted); }
.error-state { display: flex; align-items: center; justify-content: center; min-height: 200px; }
.err-msg { font-size: 13px; color: var(--red); }

.analyze-row { padding: 12px 0 0 !important; }
.analyze-row button { width: 100% !important; background: rgba(184,255,87,0.06) !important; border: 2px solid var(--accent) !important; color: var(--accent) !important; font-family: var(--mono) !important; font-size: 15px !important; letter-spacing: 4px !important; padding: 14px !important; border-radius: 2px !important; cursor: pointer !important; transition: background 0.15s !important; }
.analyze-row button:hover { background: var(--accent) !important; color: #000 !important; }

.app-footer { text-align: center; padding: 18px; font-size: 11px; color: var(--text-muted); letter-spacing: 1px; border-top: 1px solid var(--border); margin-top: 16px; }

@keyframes slideIn { from { opacity: 0; transform: translateX(-8px); } to { opacity: 1; transform: translateX(0); } }
@keyframes barGrow  { from { width: 0; } }
.fade-in { animation: fadeIn 0.25s ease; }
@keyframes fadeIn   { from { opacity: 0; } to { opacity: 1; } }
"""


def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""
    with gr.Blocks(css=CUSTOM_CSS, title="ScribblBot") as app:

        gr.HTML("""
        <div class="app-header">
            <h1 class="app-title">SCRIBBLBOT</h1>
            <p class="app-subtitle">NEURAL SKETCH CLASSIFIER · 15 CATEGORIES · QUICK DRAW DATASET</p>
        </div>
        """)

        click_counter = gr.State(0)

        with gr.Row():
            with gr.Column(elem_classes=["sketch-col"]):
                sketch_input = gr.ImageEditor(
                    type="numpy",
                    image_mode="RGBA",
                    canvas_size=(480, 480),
                    layers=False,
                    sources=[],
                    brush=gr.Brush(
                        colors=["#111111"],
                        default_size=14,
                        color_mode="fixed",
                    ),
                    eraser=gr.Eraser(default_size=20),
                    show_label=False,
                )

            with gr.Column():
                result_html = gr.HTML(_empty_state_html())

        with gr.Row(elem_classes=["analyze-row"]):
            analyze_btn = gr.Button("ANALYZE")

        gr.HTML('<div class="app-footer">ScribblBot · built with Quick Draw · PyTorch · Gradio</div>')

        analyze_btn.click(
            fn=predict,
            inputs=[sketch_input, click_counter],
            outputs=[result_html, click_counter],
        )

    return app


if __name__ == "__main__":
    demo = build_app()
    demo.launch()