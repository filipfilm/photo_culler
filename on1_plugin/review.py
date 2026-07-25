"""The metadata popup: what the culler found, and which parts of it you want kept.

The command line writes everything it decides. That is right for a thousand-frame
overnight run and wrong for the handful of photographs you select in ON1, where the
point is to look at the suggestion before it touches the catalogue. So every field is a
checkbox, keywords and descriptions are editable, and nothing is written until the
Write button is pressed.

Analysis runs on a worker thread and streams into the list as it finishes, because the
first photograph should be reviewable while the twentieth is still being looked at.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from queue import Empty, Queue
from typing import Callable, Dict, List, Optional
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk

try:
    from ..decision import DELETE, FAILED, KEEP, REVIEW, suggested_rating
    from ..models import CullResult
    from ..sidecars import WriteOptions, write_sidecar
except ImportError:
    from decision import DELETE, FAILED, KEEP, REVIEW, suggested_rating
    from models import CullResult
    from sidecars import WriteOptions, write_sidecar

try:
    from .resolve import ResolvedPhoto
except ImportError:
    from resolve import ResolvedPhoto

THUMBNAIL_SIZE = (460, 460)

DECISION_ICONS = {KEEP: "🟢", REVIEW: "🟡", DELETE: "🔴", FAILED: "⚫"}


@dataclass
class Entry:
    """One photograph in the popup, plus what the photographer has decided about it."""

    resolved: ResolvedPhoto
    result: Optional[CullResult] = None
    error: str = ""
    thumbnail: Optional[ImageTk.PhotoImage] = None
    thumbnail_requested: bool = False

    # Edited text, held here rather than in the widgets so switching photographs does
    # not throw away what you typed.
    keywords_text: str = ""
    description_text: str = ""
    choices: Dict[str, bool] = field(default_factory=dict)
    rating: int = 0

    @property
    def photo(self) -> Path:
        return self.resolved.photo

    @property
    def ready(self) -> bool:
        return self.result is not None and self.result.decision != FAILED


# The tickable fields, in the order they appear. The keys match WriteOptions.
FIELDS = [
    ("culler_keywords", "Culling keywords (PhotoCuller:Keep, confidence, issues)"),
    ("keywords", "Descriptive keywords"),
    ("description", "Description"),
    ("analysis", "Full analysis block"),
    ("rating", "Star rating (overwrites any existing rating)"),
]

# Ratings stay off by default here for the same reason they are opt-in on the command
# line: a star rating is the photographer's own shorthand and the culler does not get to
# invent one unasked. Everything else is on, because you asked for it by running this.
DEFAULT_CHOICES = {key: key != "rating" for key, _ in FIELDS}


class ReviewWindow:
    def __init__(
        self,
        root: tk.Tk,
        entries: List[Entry],
        thumbnail_loader: Callable[[Path], Optional[Image.Image]],
    ):
        self.root = root
        self.entries = entries
        self.thumbnail_loader = thumbnail_loader
        self.current: Optional[Entry] = None
        self.analysis_done = False
        self._thumbnail_queue: "Queue[tuple]" = Queue()

        for entry in entries:
            entry.choices = dict(DEFAULT_CHOICES)

        root.title("PhotoCuller - review metadata")
        root.geometry("1120x720")
        root.minsize(900, 560)

        self._build_widgets()
        self._populate_list()
        self.root.after(100, self._drain_thumbnail_queue)

    # ------------------------------------------------------------------ layout

    def _build_widgets(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        self.status = ttk.Label(outer, text="Starting analysis...")
        self.status.pack(anchor="w")

        self.progress = ttk.Progressbar(outer, mode="determinate", maximum=len(self.entries))
        self.progress.pack(fill="x", pady=(4, 10))

        panes = ttk.PanedWindow(outer, orient="horizontal")
        panes.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(panes, columns=("write",), show="tree headings", height=20)
        self.tree.heading("#0", text="Photograph")
        self.tree.heading("write", text="Write")
        self.tree.column("#0", width=300, stretch=True)
        self.tree.column("write", width=60, anchor="center", stretch=False)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        panes.add(self.tree, weight=1)

        detail = ttk.Frame(panes, padding=(12, 0, 0, 0))
        panes.add(detail, weight=2)
        self._build_detail(detail)

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(10, 0))

        self.write_on1 = tk.BooleanVar(value=False)
        self.write_xmp = tk.BooleanVar(value=False)
        ttk.Label(footer, text="Write to:").pack(side="left")
        ttk.Checkbutton(footer, text=".on1 sidecar", variable=self.write_on1).pack(side="left", padx=6)
        ttk.Checkbutton(footer, text=".xmp sidecar", variable=self.write_xmp).pack(side="left", padx=6)

        self.write_button = ttk.Button(footer, text="Write metadata", command=self._write)
        self.write_button.pack(side="right")
        self.write_button.state(["disabled"])
        ttk.Button(footer, text="Cancel", command=self.root.destroy).pack(side="right", padx=6)
        ttk.Button(footer, text="Apply these ticks to all", command=self._apply_to_all).pack(
            side="right", padx=6
        )

    def _build_detail(self, parent: ttk.Frame):
        self.headline = ttk.Label(parent, text="", font=("", 15, "bold"))
        self.headline.pack(anchor="w")
        self.subline = ttk.Label(parent, text="", foreground="#666")
        self.subline.pack(anchor="w", pady=(0, 8))

        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True)

        self.thumbnail_label = ttk.Label(body, anchor="center")
        self.thumbnail_label.pack(side="left", padx=(0, 12), anchor="n")

        fields = ttk.Frame(body)
        fields.pack(side="left", fill="both", expand=True)

        self.include = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            fields, text="Write metadata for this photograph",
            variable=self.include, command=self._on_include_toggled,
        ).pack(anchor="w", pady=(0, 6))

        self.verdict = ttk.Label(fields, text="", wraplength=460, justify="left")
        self.verdict.pack(anchor="w", pady=(0, 10))

        self.choice_vars: Dict[str, tk.BooleanVar] = {}
        for key, label in FIELDS:
            var = tk.BooleanVar(value=DEFAULT_CHOICES[key])
            self.choice_vars[key] = var
            row = ttk.Frame(fields)
            row.pack(fill="x", anchor="w")
            ttk.Checkbutton(
                row, text=label, variable=var,
                command=lambda k=key: self._on_choice_toggled(k),
            ).pack(side="left")
            if key == "rating":
                self.rating_var = tk.IntVar(value=0)
                ttk.Spinbox(
                    row, from_=0, to=5, width=3, textvariable=self.rating_var,
                    command=self._commit_current,
                ).pack(side="left", padx=6)

        ttk.Label(fields, text="Keywords (comma separated)").pack(anchor="w", pady=(10, 2))
        self.keywords_box = tk.Text(fields, height=3, wrap="word")
        self.keywords_box.pack(fill="x")

        ttk.Label(fields, text="Description").pack(anchor="w", pady=(10, 2))
        self.description_box = tk.Text(fields, height=4, wrap="word")
        self.description_box.pack(fill="both", expand=True)

    # ------------------------------------------------------------------ list

    def _row_text(self, entry: Entry) -> str:
        if entry.error:
            return f"⚫ {entry.photo.name}"
        if entry.result is None:
            return f"   {entry.photo.name}  ..."
        icon = DECISION_ICONS.get(entry.result.decision, "  ")
        return f"{icon} {entry.result.decision:<7} {entry.photo.name}"

    def _populate_list(self):
        for index, entry in enumerate(self.entries):
            self.tree.insert("", "end", iid=str(index), text=self._row_text(entry), values=("",))
        if self.entries:
            self.tree.selection_set("0")

    def _refresh_row(self, index: int):
        entry = self.entries[index]
        tick = "✓" if entry.ready and entry.choices.get("_include", True) else ""
        self.tree.item(str(index), text=self._row_text(entry), values=(tick,))

    # ------------------------------------------------------------------ analysis feed

    def on_result(self, index: int, result: Optional[CullResult], error: str = ""):
        """Called on the main thread as each photograph finishes."""
        entry = self.entries[index]
        entry.result = result
        entry.error = error
        entry.choices["_include"] = entry.ready

        if result is not None:
            entry.keywords_text = ", ".join(result.metrics.keywords or [])
            entry.description_text = result.metrics.description or ""
            entry.rating = suggested_rating(result.metrics)

        self._refresh_row(index)
        done = sum(1 for e in self.entries if e.result is not None or e.error)
        self.progress["value"] = done
        self.status.config(text=f"Analysed {done} of {len(self.entries)}")

        if self.current is entry or (self.current is None and index == 0):
            self._show(entry)

    def on_analysis_finished(self, message: str = ""):
        self.analysis_done = True
        ready = sum(1 for e in self.entries if e.ready)
        self.status.config(
            text=message or f"Analysed {len(self.entries)} photographs, {ready} ready to write"
        )
        if ready:
            self.write_button.state(["!disabled"])
            self._choose_default_sidecars()
        # Bursts are only grouped once every frame is in, so the keywords for any
        # photograph in a burst change at the end of the run.
        for index in range(len(self.entries)):
            self._refresh_row(index)
        if self.current:
            self._show(self.current)

    def on_analysis_failed(self, message: str, retry_in_fast_mode: Optional[Callable] = None):
        self.analysis_done = True
        self.status.config(text="Analysis failed")

        # Ollama being down is the common failure and it has a useful answer: the
        # measurement-only path needs no model at all. Offering it here saves closing
        # the window, starting Ollama and re-selecting everything in ON1.
        if retry_in_fast_mode is not None and messagebox.askyesno(
            "PhotoCuller",
            f"{message}\n\nRun measurement-only fast mode instead? It can tell sharp "
            "from soft and judge exposure, but it never proposes a deletion and writes "
            "no descriptions or keywords.",
            parent=self.root,
        ):
            self.status.config(text="Retrying in fast mode...")
            self.progress["value"] = 0
            retry_in_fast_mode()
            return

        messagebox.showerror("PhotoCuller", message, parent=self.root)

    def _choose_default_sidecars(self):
        """Tick whichever sidecar this folder already uses."""
        photos = [e.photo for e in self.entries if e.ready]
        has_on1 = any(p.with_suffix(".on1").exists() for p in photos)
        self.write_on1.set(has_on1)
        self.write_xmp.set(not has_on1)

    # ------------------------------------------------------------------ selection

    def _on_select(self, _event=None):
        selection = self.tree.selection()
        if not selection:
            return
        self._commit_current()
        self._show(self.entries[int(selection[0])])

    def _commit_current(self):
        """Copy the widgets back into the entry before we show a different photograph."""
        entry = self.current
        if entry is None:
            return
        entry.keywords_text = self.keywords_box.get("1.0", "end").strip()
        entry.description_text = self.description_box.get("1.0", "end").strip()
        entry.rating = self.rating_var.get()
        entry.choices["_include"] = self.include.get()
        for key, var in self.choice_vars.items():
            entry.choices[key] = var.get()

    def _on_choice_toggled(self, key: str):
        if self.current is not None:
            self.current.choices[key] = self.choice_vars[key].get()

    def _on_include_toggled(self):
        if self.current is not None:
            self.current.choices["_include"] = self.include.get()
            self._refresh_row(self.entries.index(self.current))

    def _show(self, entry: Entry):
        self.current = entry
        photo = entry.photo

        if entry.error:
            self.headline.config(text=photo.name)
            self.subline.config(text=f"could not be analysed: {entry.error}")
            self.verdict.config(text="")
        elif entry.result is None:
            self.headline.config(text=photo.name)
            self.subline.config(text="analysing...")
            self.verdict.config(text="")
        else:
            result = entry.result
            self.headline.config(
                text=f"{DECISION_ICONS.get(result.decision, '')} {result.decision}"
                     f"  ({result.confidence:.2f})"
            )
            note = entry.resolved.note
            self.subline.config(text=f"{photo.name}" + (f"   -  {note}" if note else ""))
            self.verdict.config(text=self._verdict_text(result))

        self.include.set(entry.choices.get("_include", entry.ready))
        for key, var in self.choice_vars.items():
            var.set(entry.choices.get(key, DEFAULT_CHOICES[key]))
        self.rating_var.set(entry.rating)

        self.keywords_box.delete("1.0", "end")
        self.keywords_box.insert("1.0", entry.keywords_text)
        self.description_box.delete("1.0", "end")
        self.description_box.insert("1.0", entry.description_text)

        self._show_thumbnail(entry)

    def _verdict_text(self, result: CullResult) -> str:
        metrics = result.metrics
        lines = [
            f"sharpness {metrics.subject_sharpness or '-'}    "
            f"exposure {metrics.exposure or '-'}    framing {metrics.framing or '-'}",
        ]
        if metrics.subject:
            lines.append(f"subject: {metrics.subject}")
        if result.issues:
            lines.append("issues: " + ", ".join(result.issues))
        if metrics.verdict_reason:
            lines.append(metrics.verdict_reason)
        if result.group_size > 1:
            role = "best" if result.is_best_of_group else "alternate"
            lines.append(f"burst: {role} of {result.group_size}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ thumbnails

    def _show_thumbnail(self, entry: Entry):
        if entry.thumbnail is not None:
            self.thumbnail_label.config(image=entry.thumbnail, text="")
            return

        self.thumbnail_label.config(image="", text="loading preview...", width=40)
        if not entry.thumbnail_requested:
            entry.thumbnail_requested = True
            # Decoding a RAW can take a second or two, and the window must stay usable.
            threading.Thread(target=self._load_thumbnail, args=(entry,), daemon=True).start()

    def _load_thumbnail(self, entry: Entry):
        try:
            image = self.thumbnail_loader(entry.photo)
            if image is not None:
                image = image.convert("RGB")
                image.thumbnail(THUMBNAIL_SIZE)
        except Exception:
            image = None
        self._thumbnail_queue.put((entry, image))

    def _drain_thumbnail_queue(self):
        """PhotoImage objects have to be built on the thread that owns the window."""
        while True:
            try:
                entry, image = self._thumbnail_queue.get_nowait()
            except Empty:
                break
            if image is not None:
                entry.thumbnail = ImageTk.PhotoImage(image)
                if self.current is entry:
                    self.thumbnail_label.config(image=entry.thumbnail, text="")
            elif self.current is entry:
                self.thumbnail_label.config(image="", text="(no preview)")
        self.root.after(150, self._drain_thumbnail_queue)

    # ------------------------------------------------------------------ writing

    def _apply_to_all(self):
        """Copy this photograph's ticks onto every other one, text left alone."""
        self._commit_current()
        if self.current is None:
            return
        ticks = {key: self.current.choices.get(key, DEFAULT_CHOICES[key]) for key, _ in FIELDS}
        for index, entry in enumerate(self.entries):
            if entry is not self.current and entry.ready:
                entry.choices.update(ticks)
                self._refresh_row(index)
        self.status.config(text="Applied the current ticks to every photograph")

    def _write(self):
        self._commit_current()

        styles = []
        if self.write_on1.get():
            styles.append("on1")
        if self.write_xmp.get():
            styles.append("xmp")
        if not styles:
            messagebox.showwarning(
                "PhotoCuller", "Tick .on1, .xmp or both first.", parent=self.root
            )
            return

        chosen = [e for e in self.entries if e.ready and e.choices.get("_include", True)]
        if not chosen:
            messagebox.showwarning(
                "PhotoCuller", "No photographs are ticked for writing.", parent=self.root
            )
            return

        written, missing_on1, failed = 0, [], []
        for entry in chosen:
            result = self._result_with_edits(entry)
            options = WriteOptions(
                keywords=entry.choices.get("keywords", True),
                culler_keywords=entry.choices.get("culler_keywords", True),
                description=entry.choices.get("description", True),
                force_description=entry.choices.get("description", True),
                analysis=entry.choices.get("analysis", True),
                # An explicitly ticked rating is a deliberate act, so unlike the command
                # line's suggestion it is allowed to replace one that is already there.
                rating=entry.rating if entry.choices.get("rating") else None,
                force_rating=bool(entry.choices.get("rating")),
            )
            for style in styles:
                try:
                    if write_sidecar(result, style, options=options):
                        written += 1
                    elif style == "on1":
                        missing_on1.append(entry.photo.name)
                except Exception as e:
                    failed.append(f"{entry.photo.name}: {e}")

        self._report(written, len(chosen), styles, missing_on1, failed)

    def _result_with_edits(self, entry: Entry) -> CullResult:
        keywords = [k.strip() for k in entry.keywords_text.split(",") if k.strip()]
        metrics = replace(
            entry.result.metrics, keywords=keywords, description=entry.description_text
        )
        return replace(entry.result, metrics=metrics)

    def _report(
        self, written: int, chosen: int, styles: List[str], missing_on1: List[str],
        failed: List[str],
    ):
        lines = [f"Wrote {written} sidecar(s) for {chosen} photograph(s) ({', '.join(styles)})."]
        if missing_on1:
            lines.append(
                f"\n{len(missing_on1)} photograph(s) had no .on1 file yet. ON1 has to have "
                "browsed a photo before its sidecar can be updated:\n  "
                + "\n  ".join(missing_on1[:8])
            )
        if failed:
            lines.append("\nFailed:\n  " + "\n  ".join(failed[:8]))
        if written and "on1" in styles:
            lines.append("\nRestart ON1 to see the keywords in the catalogue.")

        self.status.config(text=lines[0])
        if failed:
            messagebox.showerror("PhotoCuller", "\n".join(lines), parent=self.root)
        else:
            messagebox.showinfo("PhotoCuller", "\n".join(lines), parent=self.root)
            self.root.destroy()
