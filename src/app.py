"""Gradio Web Demo for Chinese poetry generation.

Displays three models side-by-side for comparison:
  - Decoder-Only Transformer (from scratch)
  - Encoder-Decoder Transformer (from scratch)
  - Pretrained GPT-2 Chinese Poem (fine-tuned)

Usage:
    python app.py
"""

import os
import sys
import traceback
import torch
import gradio as gr

# Write debug logs to a file
_DEBUG_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_debug.log")

def debug_log(*args):
    msg = " ".join(str(a) for a in args)
    with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()
    try:
        print(msg, flush=True)
    except:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate import (
    load_decoder_only,
    load_encoder_decoder,
    load_pretrained,
    encode_decoder_only,
    encode_encoder_decoder,
    encode_pretrained,
    decode_output,
    decode_pretrained,
    generate_greedy,
    generate_beam,
    generate_sample,
)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(_BASE_DIR, "checkpoints")
VOCAB_PATH = os.path.join(_BASE_DIR, "data", "vocab.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DECODER_CKPT = os.path.join(CHECKPOINT_DIR, "decoder_only_best.pt")
ENCODER_DECODER_CKPT = os.path.join(CHECKPOINT_DIR, "encoder_decoder_best.pt")
PRETRAINED_CKPT = os.path.join(CHECKPOINT_DIR, "pretrained_best.pt")

# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------

_models = {}

def get_model(name):
    if name in _models:
        return _models[name]

    debug_log(f"Loading {name} model...")
    try:
        if name == "decoder-only":
            if not os.path.exists(DECODER_CKPT):
                debug_log(f"  Checkpoint not found: {DECODER_CKPT}")
                return None
            model, char2id, id2char = load_decoder_only(DECODER_CKPT, VOCAB_PATH, DEVICE)
            _models[name] = (model, char2id, id2char)
        elif name == "encoder-decoder":
            if not os.path.exists(ENCODER_DECODER_CKPT):
                debug_log(f"  Checkpoint not found: {ENCODER_DECODER_CKPT}")
                return None
            model, char2id, id2char = load_encoder_decoder(ENCODER_DECODER_CKPT, VOCAB_PATH, DEVICE)
            _models[name] = (model, char2id, id2char)
        elif name == "pretrained":
            if not os.path.exists(PRETRAINED_CKPT):
                debug_log(f"  Checkpoint not found: {PRETRAINED_CKPT}")
                return None
            pretrained = load_pretrained(PRETRAINED_CKPT, DEVICE)
            _models[name] = pretrained
        debug_log(f"  {name} loaded successfully on {DEVICE}")
    except Exception as e:
        debug_log(f"  Failed to load {name}: {e}")
        traceback.print_exc()
        return None

    return _models.get(name)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    "Greedy": "greedy",
    "Beam Search (beam=5)": "beam",
    "Temperature Sampling": "sample",
}

@torch.no_grad()
def _generate_single(upper_text, model_key, strategy, temperature):
    """Generate with a single model."""
    max_new = 32

    if model_key == "pretrained":
        pretrained = get_model("pretrained")
        if pretrained is None:
            return "模型未加载"
        tokenizer = pretrained.tokenizer
        input_ids, attention_mask = encode_pretrained(upper_text, tokenizer, DEVICE)

        if strategy == "beam":
            output_ids = pretrained.generate_beam(
                input_ids, attention_mask=attention_mask,
                beam_size=5, max_new_tokens=max_new,
                eos_id=tokenizer.sep_token_id, pad_id=tokenizer.pad_token_id,
            )
        elif strategy == "sample":
            output_ids = pretrained.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new, temperature=temperature,
                do_sample=True, eos_id=tokenizer.sep_token_id,
                pad_id=tokenizer.pad_token_id,
            )
            output_ids = output_ids[0]
        else:
            output_ids = pretrained.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new, temperature=temperature,
                do_sample=False, eos_id=tokenizer.sep_token_id,
                pad_id=tokenizer.pad_token_id,
            )
            output_ids = output_ids[0]

        return decode_pretrained(output_ids, tokenizer)

    else:
        if model_key == "decoder-only":
            model_info = get_model("decoder-only")
            if model_info is None:
                return "模型未加载"
            model, char2id, id2char = model_info
            input_ids = encode_decoder_only(upper_text, char2id)
        else:
            model_info = get_model("encoder-decoder")
            if model_info is None:
                return "模型未加载"
            model, char2id, id2char = model_info
            input_ids = encode_encoder_decoder(upper_text, char2id)

        eos_id = 3

        if strategy == "greedy":
            seq, in_len = generate_greedy(model, input_ids, max_new_tokens=max_new,
                                          device=DEVICE, model_type=model_key)
        elif strategy == "beam":
            seq, in_len = generate_beam(model, input_ids, beam_size=5,
                                        max_new_tokens=max_new, device=DEVICE,
                                        model_type=model_key)
        else:
            seq, in_len = generate_sample(model, input_ids, temperature=temperature,
                                          max_new_tokens=max_new, device=DEVICE,
                                          model_type=model_key)

        return decode_output(seq, id2char, eos_id=eos_id, skip_first_n=in_len)


# ---------------------------------------------------------------------------
# Check available models
# ---------------------------------------------------------------------------

AVAILABLE = []
CHECKPOINTS = {
    "decoder-only": DECODER_CKPT,
    "encoder-decoder": ENCODER_DECODER_CKPT,
    "pretrained": PRETRAINED_CKPT,
}
for name, path in CHECKPOINTS.items():
    if os.path.exists(path):
        AVAILABLE.append(name)

MODEL_DISPLAY_NAMES = {
    "decoder-only": "Decoder-Only",
    "encoder-decoder": "Encoder-Decoder",
    "pretrained": "Pretrained (GPT-2)",
}

debug_log(f"Using device: {DEVICE}")
debug_log(f"Available models: {[MODEL_DISPLAY_NAMES.get(m, m) for m in AVAILABLE]}")
debug_log("Preloading models...")
for name in AVAILABLE:
    get_model(name)

# ---------------------------------------------------------------------------
# Gradio Blocks Interface
# ---------------------------------------------------------------------------

def generate_with_selection(upper_text, strategy, temperature, use_decoder, use_encoder, use_pretrained):
    """Generate with selected models."""
    debug_log(f"generate: upper={upper_text!r}, strategy={strategy!r}, temp={temperature!r}")

    if not upper_text or not upper_text.strip():
        return "请输入上句。", "请输入上句。", "请输入上句。"
    upper_text = upper_text.strip()

    strategy_key = STRATEGY_MAP.get(strategy, "greedy")

    selected = []
    if use_decoder:
        selected.append("decoder-only")
    if use_encoder:
        selected.append("encoder-decoder")
    if use_pretrained:
        selected.append("pretrained")

    debug_log(f"  selected models: {selected}")

    d_result, e_result, p_result = "", "", ""

    for key in selected:
        try:
            result = _generate_single(upper_text, key, strategy_key, temperature)
            if key == "decoder-only":
                d_result = result
            elif key == "encoder-decoder":
                e_result = result
            elif key == "pretrained":
                p_result = result
        except Exception as e:
            debug_log(f"  Error with {key}: {e}")
            traceback.print_exc()

    debug_log(f"  results: d={d_result!r}, e={e_result!r}, p={p_result!r}")
    return d_result, e_result, p_result

# Build the interface
if not AVAILABLE:
    demo = gr.Interface(
        fn=lambda: "未找到模型 checkpoint，请先运行 python train.py",
        inputs=[],
        outputs="text",
        title="古诗生成系统",
        description="未找到任何模型 checkpoint，请先训练模型。",
    )
else:
    with gr.Blocks(title="古诗生成系统") as demo:
        gr.Markdown("# 古诗生成系统")
        gr.Markdown("输入古诗上句，选择解码策略，生成对应的下句。")

        with gr.Row():
            with gr.Column(scale=1):
                upper_input = gr.Textbox(
                    label="上句输入",
                    placeholder="例如：床前明月光",
                    lines=2,
                )
                strategy = gr.Radio(
                    choices=["Greedy", "Beam Search (beam=5)", "Temperature Sampling"],
                    label="解码策略",
                    value="Beam Search (beam=5)",
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.8, step=0.05,
                    label="Temperature（仅采样有效）",
                )

                gr.Markdown("### 选择模型")
                use_decoder = gr.Checkbox(label="Decoder-Only", value=True)
                use_encoder = gr.Checkbox(label="Encoder-Decoder", value=True)
                use_pretrained = gr.Checkbox(label="Pretrained (GPT-2)", value=True)

                submit_btn = gr.Button("生成下句", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    d_output = gr.Textbox(label="Decoder-Only", lines=3, interactive=False)
                with gr.Row():
                    e_output = gr.Textbox(label="Encoder-Decoder", lines=3, interactive=False)
                with gr.Row():
                    p_output = gr.Textbox(label="Pretrained (GPT-2)", lines=3, interactive=False)

        gr.Examples(
            examples=[
                ["床前明月光", "Beam Search (beam=5)", 0.8],
                ["举头望明月", "Beam Search (beam=5)", 0.8],
                ["白日依山尽", "Greedy", 0.8],
                ["春眠不觉晓", "Temperature Sampling", 0.8],
                ["海内存知己", "Beam Search (beam=5)", 0.8],
            ],
            inputs=[upper_input, strategy, temperature],
        )

        submit_btn.click(
            fn=generate_with_selection,
            inputs=[upper_input, strategy, temperature, use_decoder, use_encoder, use_pretrained],
            outputs=[d_output, e_output, p_output],
        )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    debug_log("Starting server on 127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, css="footer {visibility: hidden}")
