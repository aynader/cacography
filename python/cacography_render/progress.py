"""How far the drawing has got: a bar in a terminal, a small window from an editor.

An editor (TeXstudio, VS Code, ...) shows what the renderer prints only in its message pane,
often only at the end, and cannot redraw a bar. So when nobody is watching a terminal, a
small window shows the progress instead, with a Cancel button. The window is its own process
(`python -m cacography_render.progress window`), fed one JSON line per update; that keeps
Tk out of the renderer, which works on every system and never blocks the drawing.

progress=auto (default) picks the bar in a terminal and the window otherwise; window or bar
forces one; none shows neither. CACOGRAPHY_PROGRESS overrides the document's setting.
"""
import json
import os
import subprocess
import sys
import time

MODES = ("auto", "window", "bar", "none")
CANCELLED = 2  # the window's exit code when Cancel was pressed

# A page's share of the work, by stage: writing the words takes most of it on the CPU.
WRITING, SHEET = 0.7, 0.8


def _terminal():
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


class Progress:
    def __init__(self, mode, title, pages):
        """pages: the document's page numbers that will be drawn, in order."""
        mode = os.environ.get("CACOGRAPHY_PROGRESS", mode or "auto").strip().lower()
        if mode not in MODES:
            mode = "auto"
        if mode == "auto":
            mode = "bar" if _terminal() else "window"
        self.pages, self.title = list(pages), title
        self.index, self.within, self.stage = 0, 0.0, "starting"
        self.t0 = time.time()
        self.bar = self.window = None
        if not self.pages or mode == "none":
            return
        if mode == "bar":
            try:
                from tqdm import tqdm
                self.bar = tqdm(total=1000, unit="", bar_format="{l_bar}{bar}| {elapsed}<{remaining}",
                                dynamic_ncols=True, file=sys.stdout)
            except ImportError:
                pass
        else:
            self.window = _open_window()
        self._show()

    # -- what the renderer reports ----------------------------------------------

    def page(self, n):
        """Starting the document's page n."""
        self.index = self.pages.index(n) if n in self.pages else self.index
        self.within, self.stage = 0.0, "writing"
        self._show(words=None)

    def words(self, done, total):
        self.within = WRITING * (done / total if total else 1.0)
        self.stage = "writing"
        self._show(words=(done, total))

    def sheet(self):
        self.within, self.stage = WRITING, "inking the sheet"
        self._show()

    def photo(self):
        self.within, self.stage = SHEET, "photographing"
        self._show()

    def page_done(self):
        self.within = 1.0
        self._show()

    def cancelled(self):
        """True once Cancel was pressed in the window."""
        return self.window is not None and self.window.poll() == CANCELLED

    def close(self):
        if self.bar is not None:
            self.bar.n = 1000
            self.bar.refresh()
            self.bar.close()
            self.bar = None
        if self.window is not None:
            self._send({"done": True})
            try:
                self.window.stdin.close()
                self.window.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.window.kill()
            self.window = None

    # -- showing it -------------------------------------------------------------

    def _fraction(self):
        return (self.index + self.within) / max(len(self.pages), 1)

    def _show(self, words=None):
        n = self.pages[self.index] if self.pages else 0
        label = f"page {self.index + 1} of {len(self.pages)}"
        if len(self.pages) and n != self.index + 1:
            label += f" (page {n})"
        detail = self.stage
        if words is not None:
            detail = f"writing word {words[0]} of {words[1]}"
        if self.bar is not None:
            self.bar.set_description_str(f"cacography: {label}, {detail}"[:70])
            self.bar.n = int(1000 * self._fraction())
            self.bar.refresh()
        if self.window is not None:
            self._send({"title": self.title, "label": f"Drawing {label}", "detail": detail,
                        "fraction": self._fraction(), "elapsed": time.time() - self.t0})

    def _send(self, message):
        try:
            self.window.stdin.write(json.dumps(message) + "\n")
            self.window.stdin.flush()
        except (OSError, ValueError):
            pass  # the window was closed: carry on drawing without it


def _open_window():
    """Start the window process, or None where no window can be shown."""
    flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW: no console flashes up
    try:
        child = subprocess.Popen([sys.executable, "-m", "cacography_render.progress", "window"],
                                 stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                                 creationflags=flags)
    except OSError:
        return None
    time.sleep(0.3)
    return child if child.poll() is None else None  # exits at once without a display


# -- the window process ---------------------------------------------------------

def window():
    import queue
    import threading
    import tkinter as tk
    from tkinter import ttk

    try:
        root = tk.Tk()
    except tk.TclError:
        sys.exit(1)  # no display
    root.title("cacography")
    root.resizable(False, False)
    root.attributes("-topmost", True)
    frame = ttk.Frame(root, padding=14)
    frame.grid()
    title = ttk.Label(frame, text="cacography", font=("TkDefaultFont", 11, "bold"))
    title.grid(sticky="w")
    label = ttk.Label(frame, text="Getting ready ...", width=46)
    label.grid(sticky="w", pady=(6, 0))
    bar = ttk.Progressbar(frame, length=340, maximum=1000)
    bar.grid(pady=8)
    detail = ttk.Label(frame, text="", foreground="#555")
    detail.grid(sticky="w")
    cancel = ttk.Button(frame, text="Cancel", command=lambda: (root.destroy(), sys.exit(CANCELLED)))
    cancel.grid(sticky="e", pady=(8, 0))
    root.protocol("WM_DELETE_WINDOW", root.destroy)  # closing it only hides the progress
    root.update_idletasks()
    x = root.winfo_screenwidth() - root.winfo_reqwidth() - 40
    root.geometry(f"+{x}+60")

    updates = queue.Queue()

    def read():
        for line in sys.stdin:
            updates.put(line)
        updates.put(None)

    threading.Thread(target=read, daemon=True).start()

    def poll():
        try:
            while True:
                line = updates.get_nowait()
                if line is None:
                    root.destroy()
                    return
                m = json.loads(line)
                if m.get("done"):
                    root.destroy()
                    return
                title.config(text=f"cacography: {m.get('title', '')}")
                label.config(text=m.get("label", ""))
                f = float(m.get("fraction", 0))
                bar["value"] = int(1000 * f)
                left = ""
                if f > 0.03:
                    secs = m.get("elapsed", 0) * (1 - f) / f
                    left = f", about {secs / 60:.0f} min left" if secs >= 90 else f", about {secs:.0f} s left"
                detail.config(text=m.get("detail", "") + left)
        except queue.Empty:
            pass
        except (ValueError, tk.TclError):
            pass
        root.after(100, poll)

    root.after(100, poll)
    root.mainloop()


if __name__ == "__main__" and sys.argv[1:] == ["window"]:
    window()
