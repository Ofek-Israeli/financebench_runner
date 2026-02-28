#!/usr/bin/env python3
"""
Tk GUI or curses TUI to edit financebench_runner YAML .config.
Run from financebench_runner: python3 menuconfig_gui.py [.config]
Or: make menuconfig
Uses Tk GUI if python3-tk is available, otherwise curses TUI.
"""

import sys
from pathlib import Path

import yaml

try:
    from tkinter import (
        Tk,
        Frame,
        Label,
        Entry,
        Button,
        Text,
        Scrollbar,
        StringVar,
        DoubleVar,
        IntVar,
        messagebox,
        N,
        S,
        E,
        W,
        BOTH,
        END,
    )
    HAS_TK = True
except ImportError:
    HAS_TK = False

from config_io import load_yaml, save_yaml

CONFIG_DEFAULT = ".config"


def run_gui(config_path: Path) -> None:
    cfg = load_yaml(config_path)
    sglang = cfg.get("sglang") or {}
    if not isinstance(sglang, dict):
        sglang = {}

    root = Tk()
    root.title("financebench_runner — Config")
    root.minsize(500, 520)

    main = Frame(root, padx=10, pady=10)
    main.pack(fill=BOTH, expand=True)

    row = 0

    def add_row(label_text: str, var=None, widget=None):
        nonlocal row
        lbl = Label(main, text=label_text, anchor=W)
        lbl.grid(row=row, column=0, sticky=W, pady=2)
        if widget is not None:
            widget.grid(row=row, column=1, sticky=W + E, pady=2)
        else:
            ent = Entry(main, textvariable=var, width=50)
            ent.grid(row=row, column=1, sticky=W + E, pady=2)
        row += 1

    model_id = StringVar(value=cfg.get("model_id") or "llama3.2:3b")
    temperature = DoubleVar(value=float(cfg.get("temperature", 0.0)))
    indices_raw = cfg.get("example_indices")
    indices_str = ", ".join(str(i) for i in indices_raw) if isinstance(indices_raw, list) else ""
    example_indices_var = StringVar(value=indices_str)
    sglang_url_var = StringVar(value=sglang.get("base_url") or "http://localhost:11434/v1")
    sglang_timeout_var = DoubleVar(value=float(sglang.get("timeout_s", 120.0)))
    sglang_retries_var = IntVar(value=int(sglang.get("max_retries", 3)))
    correctness_model_var = StringVar(value=cfg.get("correctness_model") or "gpt-4o")
    correctness_tol_var = DoubleVar(value=float(cfg.get("correctness_tolerance", 0.10)))
    max_new_tokens_var = IntVar(value=int(cfg.get("max_new_tokens", 512)))
    top_p_var = DoubleVar(value=float(cfg.get("top_p", 1.0)))
    seed_var = IntVar(value=int(cfg.get("seed", 42)))

    add_row("model_id:", model_id)
    add_row("temperature:", temperature)
    add_row("example_indices (comma-sep, empty=all):", example_indices_var)
    add_row("sglang.base_url:", sglang_url_var)
    add_row("sglang.timeout_s:", sglang_timeout_var)
    add_row("sglang.max_retries:", sglang_retries_var)
    add_row("correctness_model:", correctness_model_var)
    add_row("correctness_tolerance:", correctness_tol_var)
    add_row("max_new_tokens:", max_new_tokens_var)
    add_row("top_p:", top_p_var)
    add_row("seed:", seed_var)

    Label(main, text="prompt_template:", anchor=W).grid(row=row, column=0, sticky=N + W, pady=4)
    prompt_text = Text(main, width=50, height=8, wrap="word")
    prompt_text.grid(row=row, column=1, sticky=W + E + N + S, pady=2)
    sb = Scrollbar(main, command=prompt_text.yview)
    sb.grid(row=row, column=2, sticky=N + S)
    prompt_text.config(yscrollcommand=sb.set)
    prompt_text.insert(END, cfg.get("prompt_template") or "  Context:\n  {context}\n\n  Question:\n  {query}\n\n  Answer:\n")
    row += 1

    main.columnconfigure(1, weight=1)
    main.rowconfigure(row - 1, weight=1)

    def save():
        try:
            indices_str = example_indices_var.get().strip()
            if indices_str:
                example_indices = [int(x.strip()) for x in indices_str.split(",") if x.strip()]
            else:
                example_indices = None
            out = {
                "model_id": model_id.get().strip() or "llama3.2:3b",
                "temperature": float(temperature.get()),
                "sglang": {
                    "base_url": sglang_url_var.get().strip() or "http://localhost:11434/v1",
                    "timeout_s": float(sglang_timeout_var.get()),
                    "max_retries": int(sglang_retries_var.get()),
                },
                "correctness_model": correctness_model_var.get().strip() or "gpt-4o",
                "correctness_tolerance": float(correctness_tol_var.get()),
                "max_new_tokens": int(max_new_tokens_var.get()),
                "top_p": float(top_p_var.get()),
                "seed": int(seed_var.get()),
                "prompt_template": prompt_text.get("1.0", END).rstrip(),
            }
            if example_indices is not None:
                out["example_indices"] = example_indices
            save_yaml(config_path, out)
            messagebox.showinfo("Saved", f"Configuration saved to {config_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    btn_frame = Frame(main)
    btn_frame.grid(row=row, column=0, columnspan=3, pady=10)
    Button(btn_frame, text="Save", command=save).pack(side="left", padx=4)
    Button(btn_frame, text="Quit", command=root.quit).pack(side="left", padx=4)

    root.mainloop()


def main():
    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else CONFIG_DEFAULT)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if HAS_TK:
        run_gui(config_path)
    else:
        import menuconfig_tui
        menuconfig_tui.main(config_path)


if __name__ == "__main__":
    main()
