#!/usr/bin/env python3
"""Curses TUI to edit financebench_runner YAML .config (fallback when tkinter is not available)."""
import sys
from pathlib import Path

from config_io import load_yaml, save_yaml

CONFIG_DEFAULT = ".config"


def _cfg_str(cfg: dict, sglang: dict, key: str, default: str = "") -> str:
    if key.startswith("sglang."):
        return str(sglang.get(key.split(".", 1)[1], default))
    return str(cfg.get(key, default))


def _cfg_float(cfg: dict, sglang: dict, key: str, default: float = 0.0) -> float:
    try:
        if key.startswith("sglang."):
            return float(sglang.get(key.split(".", 1)[1], default))
        return float(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def _cfg_int(cfg: dict, sglang: dict, key: str, default: int = 0) -> int:
    try:
        if key.startswith("sglang."):
            return int(sglang.get(key.split(".", 1)[1], default))
        return int(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def run_tui(config_path: Path) -> None:
    if not sys.stdin.isatty():
        print(
            "No TTY. Run 'make menuconfig' from an interactive terminal,\n"
            "or install python3-tk for the graphical config (apt install python3-tk).",
            file=sys.stderr,
        )
        sys.exit(1)
    import curses

    cfg = load_yaml(config_path)
    sglang = cfg.get("sglang") or {}
    if not isinstance(sglang, dict):
        sglang = {}

    indices_raw = cfg.get("example_indices")
    if isinstance(indices_raw, list):
        indices_str = ", ".join(str(i) for i in indices_raw)
    else:
        indices_str = ""

    fields = [
        ("model_id", _cfg_str(cfg, sglang, "model_id", "llama3.2:3b"), "str"),
        ("temperature", str(_cfg_float(cfg, sglang, "temperature", 0.0)), "float"),
        ("example_indices (comma-sep, empty=all)", indices_str, "str"),
        ("sglang.base_url", _cfg_str(cfg, sglang, "sglang.base_url", "http://localhost:11434/v1"), "str"),
        ("sglang.timeout_s", str(_cfg_float(cfg, sglang, "sglang.timeout_s", 120.0)), "float"),
        ("sglang.max_retries", str(_cfg_int(cfg, sglang, "sglang.max_retries", 3)), "int"),
        ("correctness_model", _cfg_str(cfg, sglang, "correctness_model", "gpt-4o"), "str"),
        ("correctness_tolerance", str(_cfg_float(cfg, sglang, "correctness_tolerance", 0.10)), "float"),
        ("max_new_tokens", str(_cfg_int(cfg, sglang, "max_new_tokens", 512)), "int"),
        ("top_p", str(_cfg_float(cfg, sglang, "top_p", 1.0)), "float"),
        ("seed", str(_cfg_int(cfg, sglang, "seed", 42)), "int"),
    ]
    prompt_template = cfg.get("prompt_template") or "  Context:\n  {context}\n\n  Question:\n  {query}\n\n  Answer:\n"

    current = 0
    edit_buf = ""

    def draw(scr, editing: bool = False):
        scr.clear()
        h, w = scr.getmaxyx()
        curses.curs_set(1 if editing else 0)
        title = " financebench_runner config — Up/Down: select, Enter: edit, S: save, Q: quit "
        scr.addstr(0, 0, title[: w - 1], curses.A_BOLD)
        for i, (label, value, _) in enumerate(fields):
            line = f"  {label}: {value}"
            if len(line) > w - 1:
                line = line[: w - 2]
            attr = curses.A_REVERSE if i == current and not editing else 0
            scr.addstr(i + 2, 0, line, attr)
        scr.addstr(len(fields) + 3, 0, "  [S] Save  [Q] Quit", curses.A_DIM)
        scr.addstr(len(fields) + 5, 0, "  prompt_template (edit in GUI or file): (fixed in TUI)", curses.A_DIM)
        if editing:
            scr.addstr(len(fields) + 7, 0, "  New value: " + edit_buf[: w - 20], curses.A_NORMAL)
        scr.refresh()

    def edit_field(scr, idx: int) -> str:
        nonlocal edit_buf
        _, _, typ = fields[idx]
        label, value, _ = fields[idx]
        edit_buf = value
        h, w = scr.getmaxyx()
        while True:
            draw(scr, editing=True)
            scr.move(len(fields) + 7, 13 + len(edit_buf))
            key = scr.getch()
            if key in (curses.KEY_ENTER, 10, 13):
                return edit_buf
            if key == 27:  # ESC
                return value
            if key == curses.KEY_BACKSPACE or key == 127:
                edit_buf = edit_buf[:-1]
            elif key >= 32 and key < 127:
                edit_buf += chr(key)
        return edit_buf

    def save():
        values = []
        for i, (_, _, typ) in enumerate(fields):
            raw = fields[i][1]
            if typ == "float":
                values.append(float(raw))
            elif typ == "int":
                values.append(int(raw))
            else:
                values.append(raw)

        indices_str = values[2].strip()
        example_indices = None
        if indices_str:
            example_indices = [int(x.strip()) for x in indices_str.split(",") if x.strip()]

        out = {
            "model_id": values[0].strip() or "llama3.2:3b",
            "temperature": values[1],
            "sglang": {
                "base_url": values[3].strip() or "http://localhost:11434/v1",
                "timeout_s": values[4],
                "max_retries": int(values[5]),
            },
            "correctness_model": values[6].strip() or "gpt-4o",
            "correctness_tolerance": values[7],
            "max_new_tokens": int(values[8]),
            "top_p": values[9],
            "seed": int(values[10]),
            "prompt_template": prompt_template,
        }
        if example_indices is not None:
            out["example_indices"] = example_indices
        save_yaml(config_path, out)

    def main_loop(scr):
        nonlocal current, fields
        curses.cbreak()
        scr.keypad(True)
        curses.noecho()
        while True:
            draw(scr)
            key = scr.getch()
            if key == ord("q") or key == ord("Q"):
                return
            if key == ord("s") or key == ord("S"):
                try:
                    save()
                    h, w = scr.getmaxyx()
                    scr.addstr(len(fields) + 8, 0, f"  Saved to {config_path}".ljust(w - 1), curses.A_BOLD)
                    scr.refresh()
                    scr.getch()
                except Exception as e:
                    scr.addstr(len(fields) + 8, 0, f"  Error: {e}".ljust(80))
                    scr.refresh()
                    scr.getch()
                continue
            if key == curses.KEY_UP:
                current = (current - 1) % len(fields)
            elif key == curses.KEY_DOWN:
                current = (current + 1) % len(fields)
            elif key in (curses.KEY_ENTER, 10, 13):
                new_val = edit_field(scr, current)
                fields[current] = (fields[current][0], new_val, fields[current][2])

    try:
        curses.wrapper(main_loop)
    except KeyboardInterrupt:
        pass


def main(config_path: Path | None = None):
    if config_path is None:
        config_path = Path(sys.argv[1] if len(sys.argv) > 1 else CONFIG_DEFAULT)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    run_tui(config_path)


if __name__ == "__main__":
    main()
