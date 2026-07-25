"""Entry point for the ON1 plugin: take selected photographs, analyse, offer metadata.

Launched by PhotoCuller.app with the selected files as arguments. The bundle is what
ON1's "Send to Other Application" (and Finder's Open With, and dragging onto the Dock
icon) can actually reach; everything past that point happens here.

Run it straight from a terminal too, for testing:

    python3.11 on1_plugin/app.py ~/Pictures/Shoot/DSC_0001.NEF
    python3.11 on1_plugin/app.py --fast ~/Pictures/Shoot
"""

from pathlib import Path
from typing import List, Optional
import os
import subprocess
import sys
import threading

# Running as a plain script puts on1_plugin/ on the path but not the project root, and
# the analysis lives in the root. Nothing else in this package needs to care.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tkinter as tk
from tkinter import filedialog, messagebox

from batch import BatchCuller
from config import Config
from decision import FAILED
from grouping import annotate_results
from vision import ModelCannotSee, VisionUnavailable, detect_vision_model

from on1_plugin.resolve import resolve_paths
from on1_plugin.review import Entry, ReviewWindow

# A selection this big is a folder-sized job and belongs on the command line, where it
# can run overnight and write a CSV. The popup would also need 900 thumbnails.
SELECTION_WARNING_THRESHOLD = 200


def bring_to_front():
    """Take focus from ON1.

    The bundle launches us as a detached process rather than as the bundle's own
    executable, so macOS does not activate the window for us and the popup would open
    behind ON1 -- looking, from the photographer's side, like nothing happened.
    """
    try:
        subprocess.run(
            ["/usr/bin/osascript", "-e",
             'tell application "System Events" to set frontmost of the first process '
             f"whose unix id is {os.getpid()} to true"],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass


class PluginRun:
    def __init__(self, entries: List[Entry], config: Config):
        self.entries = entries
        self.config = config

        self.root = tk.Tk()
        self.window = ReviewWindow(self.root, entries, self._load_thumbnail)
        self.extractor = None

    # ------------------------------------------------------------------ analysis

    def start(self, fast: bool = False):
        threading.Thread(target=self._analyse, args=(fast,), daemon=True).start()

    def _build_culler(self, fast: bool) -> BatchCuller:
        config = self.config
        model = None if fast else (config.model_name or detect_vision_model(config.host))
        return BatchCuller(
            cache_dir=config.resolved_cache_dir(),
            mode="fast" if fast else "accurate",
            # One at a time: results stream into the list in the order they are shown,
            # and a handful of photographs gains nothing from parallelism anyway.
            max_workers=1,
            use_ollama=not fast,
            ollama_model=model,
            ollama_host=config.host,
            timeout=config.timeout_seconds,
            context_tokens=config.context_tokens,
            burst_gap_seconds=config.burst_gap_seconds,
            sharp_evidence_vetoes_delete=config.sharp_evidence_vetoes_delete,
            with_tags=config.tagging and not fast,
            verify_vision=config.verify_vision and not fast,
        )

    def _analyse(self, fast: bool):
        try:
            culler = self._build_culler(fast)
        except (VisionUnavailable, ModelCannotSee) as e:
            retry = None if fast else (lambda: self.start(fast=True))
            self._on_main(self.window.on_analysis_failed, str(e), retry)
            return
        except Exception as e:
            self._on_main(self.window.on_analysis_failed, f"Could not start the culler: {e}")
            return

        self.extractor = culler.extractor

        results = []
        for index, entry in enumerate(self.entries):
            result = culler.process_image(entry.photo)
            results.append(result)
            error = "; ".join(result.issues) if result.decision == FAILED else ""
            self._on_main(self.window.on_result, index, result, error)

        if self.config.grouping_enabled and len(results) > 1:
            # Grouping needs every frame before it can say which one is the best of a
            # burst, so it runs once at the end and the rows are redrawn afterwards.
            annotate_results(results, burst_gap_seconds=self.config.burst_gap_seconds)

        message = ""
        if fast:
            ready = sum(1 for e in self.entries if e.ready)
            message = f"Analysed {ready} photographs in fast mode - no model, no deletions"
        self._on_main(self.window.on_analysis_finished, message)

    def _on_main(self, callback, *args):
        """Hop back to the thread that owns the window; Tk tolerates nothing else.

        Closing the window part-way through a run is a normal thing to do, and it leaves
        this worker holding a destroyed root. Tk says so by raising, which is the one
        case here that is not worth reporting.
        """
        try:
            self.root.after(0, callback, *args)
        except tk.TclError:
            pass

    def _load_thumbnail(self, photo: Path):
        if self.extractor is None:
            try:
                from extractor import RawThumbnailExtractor
            except ImportError:  # pragma: no cover - project layout guarantees this
                return None
            self.extractor = RawThumbnailExtractor(self.config.resolved_cache_dir())
        return self.extractor.extract(photo)

    def run(self):
        bring_to_front()
        self.root.mainloop()


def pick_folder() -> List[str]:
    """No arguments means somebody opened the app directly, so ask what to cull."""
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Choose a folder of photographs to cull")
    root.destroy()
    return [folder] if folder else []


def fatal(message: str):
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("PhotoCuller", message)
    root.destroy()
    sys.exit(1)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    fast = "--fast" in argv
    paths = [a for a in argv if not a.startswith("--")]

    if not paths:
        paths = pick_folder()
        if not paths:
            return 0

    config = Config.load()
    resolved = resolve_paths(paths, config.extensions)

    if not resolved:
        fatal(
            "Nothing here that PhotoCuller can read.\n\n"
            "It handles RAW, HEIC, JPEG, TIFF and PSD. If you sent these from ON1, "
            "check the Send To dialog was set to a file type rather than cancelled."
        )

    if len(resolved) > SELECTION_WARNING_THRESHOLD:
        root = tk.Tk()
        root.withdraw()
        proceed = messagebox.askokcancel(
            "PhotoCuller",
            f"{len(resolved)} photographs selected. At a few seconds each this will take "
            f"a while, and the review list will be long.\n\nFor a whole shoot the command "
            f"line is the better tool - it writes a CSV and an HTML contact sheet.\n\n"
            f"Carry on anyway?",
        )
        root.destroy()
        if not proceed:
            return 0

    run = PluginRun([Entry(resolved=r) for r in resolved], config)
    run.start(fast=fast)
    run.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
