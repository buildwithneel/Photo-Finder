from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import sqlite3
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageOps, ImageTk, UnidentifiedImageError

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None

try:
    from insightface.app import FaceAnalysis
except Exception as exc:
    FaceAnalysis = None
    _INSIGHTFACE_IMPORT_ERROR = exc
else:
    _INSIGHTFACE_IMPORT_ERROR = None


APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "facefinder_config.json"
DB_PATH = APP_DIR / "facefinder_index.sqlite3"
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"
}
MODEL_NAME = "buffalo_l"
DETECTION_SIZE = (640, 640)
DEFAULT_THRESHOLD = 0.40
PREVIEW_MAX_SIZE = (1000, 700)
THUMB_SIZE = (150, 115)


def normalized_embedding(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid face embedding")
    return arr / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = normalized_embedding(a)
    bb = normalized_embedding(b)
    if aa.shape != bb.shape:
        return -1.0
    return float(np.clip(np.dot(aa, bb), -1.0, 1.0))


def load_config() -> dict:
    defaults = {"source_dir": "", "destination_dir": "", "threshold": DEFAULT_THRESHOLD}
    try:
        if CONFIG_PATH.exists():
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                defaults.update(saved)
    except Exception:
        pass
    return defaults


def save_config(config: dict) -> None:
    try:
        temp = CONFIG_PATH.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        temp.replace(CONFIG_PATH)
    except OSError as exc:
        raise RuntimeError(f"Could not save settings: {exc}") from exc


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            path TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            size INTEGER NOT NULL,
            width INTEGER,
            height INTEGER,
            taken_at TEXT,
            file_hash TEXT,
            indexed_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            face_index INTEGER NOT NULL,
            bbox_json TEXT NOT NULL,
            embedding BLOB NOT NULL,
            embedding_dim INTEGER NOT NULL,
            det_score REAL,
            face_quality REAL,
            UNIQUE(image_path, face_index),
            FOREIGN KEY(image_path) REFERENCES images(path) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_image ON faces(image_path)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            name TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            reference_paths TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def image_taken_time(path: Path) -> str:
    # File modification time is a reliable fallback; EXIF DateTimeOriginal is preferred.
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            for key in (36867, 306):  # DateTimeOriginal, DateTime
                val = exif.get(key)
                if val:
                    try:
                        return datetime.strptime(str(val), "%Y:%m:%d %H:%M:%S").isoformat(sep=" ")
                    except ValueError:
                        pass
    except Exception:
        pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ", timespec="seconds")
    except OSError:
        return ""


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def image_file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def quality_score(face, rgb: np.ndarray) -> float:
    """Heuristic face quality score, used for ranking only."""
    try:
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(rgb.shape[1], x2), min(rgb.shape[0], y2)
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        area_ratio = crop.shape[0] * crop.shape[1] / max(1, rgb.shape[0] * rgb.shape[1])
        # Log compression prevents giant faces dominating; normalized 0..1-ish.
        sharp = min(1.0, sharpness / 180.0)
        size = min(1.0, np.sqrt(max(0.0, area_ratio)) * 5.0)
        det = float(getattr(face, "det_score", 0.0) or 0.0)
        return float(0.45 * sharp + 0.30 * size + 0.25 * det)
    except Exception:
        return float(getattr(face, "det_score", 0.0) or 0.0)


@dataclass
class FaceRecord:
    image_path: str
    face_index: int
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray
    det_score: float
    quality: float


@dataclass
class MatchRecord:
    path: str
    score: float
    quality: float
    taken_at: str
    face_index: int
    bbox: tuple[int, int, int, int]
    duplicate_group: str = ""


class FaceEngine:
    def __init__(self, status: Optional[Callable[[str], None]] = None):
        if FaceAnalysis is None:
            raise RuntimeError(
                "InsightFace could not be imported. Install dependencies with:\n"
                "python -m pip install insightface onnxruntime opencv-python pillow pillow-heif numpy\n\n"
                f"Import error: {_INSIGHTFACE_IMPORT_ERROR}"
            )
        self.status = status or (lambda _msg: None)
        self.app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
        self.status("Loading face model (first launch may take a while)...")
        self.app.prepare(ctx_id=-1, det_size=DETECTION_SIZE)

    def detect(self, path: Path) -> list[FaceRecord]:
        try:
            pil = load_rgb(path)
            rgb = np.asarray(pil)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            faces = self.app.get(bgr)
            out = []
            for i, face in enumerate(faces):
                emb = normalized_embedding(face.embedding)
                bbox = tuple(int(round(v)) for v in face.bbox)
                out.append(FaceRecord(
                    image_path=str(path.resolve()),
                    face_index=i,
                    bbox=bbox,
                    embedding=emb,
                    det_score=float(getattr(face, "det_score", 0.0) or 0.0),
                    quality=quality_score(face, rgb),
                ))
            return out
        except (OSError, UnidentifiedImageError, ValueError, cv2.error):
            return []
        except Exception:
            # One damaged/unusual image must not abort the entire album scan.
            return []

    def detect_pil(self, pil: Image.Image) -> list:
        rgb = np.asarray(pil.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return self.app.get(bgr)


def get_image_paths(folder: Path) -> list[Path]:
    found = []
    for root, dirs, files in os.walk(folder):
        # Don't recurse into FaceFinder's own export folder if it sits inside source.
        dirs[:] = [d for d in dirs if d.lower() not in {"facefinder_export", ".facefinder_cache"}]
        for name in files:
            p = Path(root) / name
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                found.append(p)
    return sorted(found, key=lambda p: str(p).lower())


def index_image(conn: sqlite3.Connection, engine: FaceEngine, path: Path) -> tuple[int, bool]:
    try:
        stat = path.stat()
        resolved = str(path.resolve())
        row = conn.execute("SELECT mtime, size FROM images WHERE path=?", (resolved,)).fetchone()
        if row and abs(float(row[0]) - stat.st_mtime) < 0.001 and int(row[1]) == stat.st_size:
            return conn.execute("SELECT COUNT(*) FROM faces WHERE image_path=?", (resolved,)).fetchone()[0], False

        faces = engine.detect(path)
        width = height = None
        try:
            with Image.open(path) as im:
                width, height = im.size
        except Exception:
            pass
        taken_at = image_taken_time(path)
        # Hash only changed/new files; useful for exact duplicate detection.
        try:
            digest = image_file_hash(path)
        except OSError:
            digest = ""
        conn.execute("DELETE FROM faces WHERE image_path=?", (resolved,))
        conn.execute("""
            INSERT OR REPLACE INTO images(path,mtime,size,width,height,taken_at,file_hash,indexed_at)
            VALUES(?,?,?,?,?,?,?,?)
        """, (resolved, stat.st_mtime, stat.st_size, width, height, taken_at, digest, time.time()))
        for f in faces:
            conn.execute("""
                INSERT OR REPLACE INTO faces(image_path,face_index,bbox_json,embedding,embedding_dim,det_score,face_quality)
                VALUES(?,?,?,?,?,?,?)
            """, (
                resolved, f.face_index, json.dumps(f.bbox),
                f.embedding.astype(np.float32).tobytes(), int(f.embedding.size),
                f.det_score, f.quality
            ))
        return len(faces), True
    except (OSError, sqlite3.Error):
        return 0, False


def load_indexed_embeddings(conn: sqlite3.Connection) -> list[tuple[str, str, np.ndarray, tuple, float, float]]:
    rows = conn.execute("""
        SELECT f.image_path, i.taken_at, f.embedding, f.embedding_dim, f.bbox_json, f.face_quality, f.det_score
        FROM faces f JOIN images i ON i.path=f.image_path
    """).fetchall()
    result = []
    for path, taken_at, blob, dim, bbox_json, quality, det_score in rows:
        try:
            emb = np.frombuffer(blob, dtype=np.float32, count=int(dim)).copy()
            result.append((path, taken_at or "", normalized_embedding(emb),
                           tuple(json.loads(bbox_json)), float(quality or 0), float(det_score or 0)))
        except Exception:
            continue
    return result


class FaceChooser(tk.Toplevel):
    """Tk-only face picker. Avoids mixing cv2.imshow with Tk dialogs."""
    def __init__(self, parent, pil: Image.Image, boxes: list[tuple[int, int, int, int]], title="Choose a face"):
        super().__init__(parent)
        self.title(title)
        self.result = None
        self.transient(parent)
        self.grab_set()
        self.geometry("1000x760")
        self.pil = pil.copy()
        self.boxes = boxes
        self.scale = min(1.0, 900 / max(1, pil.width), 620 / max(1, pil.height))
        display = pil.resize((int(pil.width*self.scale), int(pil.height*self.scale)), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display)
        ttk.Label(self, text="Click the face you want to use as the reference.").pack(padx=8, pady=6)
        self.canvas = tk.Canvas(self, width=display.width, height=display.height, highlightthickness=0)
        self.canvas.pack(padx=8, pady=8)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        for i, (x1,y1,x2,y2) in enumerate(boxes):
            self.canvas.create_rectangle(x1*self.scale, y1*self.scale, x2*self.scale, y2*self.scale,
                                         outline="#00d084", width=3)
            self.canvas.create_text(x1*self.scale+5, y1*self.scale+5, anchor="nw",
                                    text=str(i+1), fill="#ffeb3b", font=("Arial", 16, "bold"))
        self.canvas.bind("<Button-1>", self._click)
        row = ttk.Frame(self); row.pack(fill="x", padx=8, pady=6)
        ttk.Button(row, text="Cancel", command=self.destroy).pack(side="right")

    def _click(self, event):
        x, y = event.x / self.scale, event.y / self.scale
        for i, (x1,y1,x2,y2) in enumerate(self.boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.result = i
                self.destroy()
                return


class ImagePreview(tk.Toplevel):
    def __init__(self, parent, path: Path, title="Photo preview"):
        super().__init__(parent)
        self.title(title)
        self.geometry("900x700")
        try:
            im = load_rgb(path)
            im.thumbnail(PREVIEW_MAX_SIZE, Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(im)
            ttk.Label(self, image=self.photo).pack(expand=True, padx=8, pady=8)
            ttk.Label(self, text=str(path), wraplength=850).pack(padx=8, pady=5)
        except Exception as exc:
            ttk.Label(self, text=f"Cannot preview image:\n{exc}").pack(padx=20, pady=20)


class FaceFinder(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FaceFinder — Wedding Album Search")
        self.geometry("1120x820")
        self.minsize(900, 650)
        self.config_data = load_config()
        self.source_dir = self.config_data.get("source_dir", "")
        self.destination_dir = self.config_data.get("destination_dir", "")
        self.threshold = tk.DoubleVar(value=float(self.config_data.get("threshold", DEFAULT_THRESHOLD)))
        self.status_var = tk.StringVar(value="Ready. Choose a person and search your album.")
        self.progress_var = tk.DoubleVar(value=0)
        self.reference_paths: list[Path] = []
        self.reference_embeddings: list[np.ndarray] = []
        self.reference_name = tk.StringVar(value="")
        self.matches: list[MatchRecord] = []
        self.result_widgets = []
        self.thumbnail_refs = []
        self.busy = False
        self.msg_queue = queue.Queue()
        self.engine = None
        self._build_ui()
        self.after(150, self._poll_queue)
        self.after(250, self._load_profiles_into_combo)

    def _build_ui(self):
        top = ttk.Frame(self, padding=10); top.pack(fill="x")
        ttk.Label(top, text="FaceFinder", font=("Segoe UI", 19, "bold")).pack(anchor="w")
        ttk.Label(top, text="Search a large wedding album for one person. Results are suggestions—review before exporting.",
                  wraplength=1000).pack(anchor="w", pady=(2, 8))

        folders = ttk.LabelFrame(self, text="Saved folder locations", padding=8); folders.pack(fill="x", padx=10, pady=4)
        self.source_label = ttk.Label(folders, text=self.source_dir or "Source folder: not set", wraplength=800)
        self.source_label.grid(row=0, column=0, sticky="w")
        self.dest_label = ttk.Label(folders, text=self.destination_dir or "Destination folder: not set", wraplength=800)
        self.dest_label.grid(row=1, column=0, sticky="w", pady=(4,0))
        self.change_folders_btn = ttk.Button(folders, text="Change Folder Locations", command=self.change_folders)
        self.change_folders_btn.grid(row=0, column=1, rowspan=2, padx=10)

        person = ttk.LabelFrame(self, text="Person to find", padding=8); person.pack(fill="x", padx=10, pady=4)
        ttk.Label(person, text="Saved person:").grid(row=0,column=0,sticky="w")
        self.profile_combo = ttk.Combobox(person, width=25, state="readonly")
        self.profile_combo.grid(row=0,column=1,padx=5,sticky="w")
        ttk.Button(person,text="Load profile",command=self.load_profile).grid(row=0,column=2,padx=3)
        ttk.Label(person,text="Profile name:").grid(row=0,column=3,padx=(12,2),sticky="e")
        ttk.Entry(person,textvariable=self.reference_name,width=22).grid(row=0,column=4,padx=3)
        ttk.Button(person,text="Save profile",command=self.save_profile).grid(row=0,column=5,padx=3)
        refrow = ttk.Frame(person); refrow.grid(row=1,column=0,columnspan=6,sticky="w",pady=(9,2))
        ttk.Button(refrow,text="Add reference photo(s)",command=self.add_reference_photos).pack(side="left",padx=(0,5))
        ttk.Button(refrow,text="Capture webcam photo",command=self.capture_webcam).pack(side="left",padx=5)
        ttk.Button(refrow,text="Remove all references",command=self.clear_references).pack(side="left",padx=5)
        self.refs_label = ttk.Label(person,text="No reference photos selected",wraplength=900)
        self.refs_label.grid(row=2,column=0,columnspan=6,sticky="w",pady=(5,0))

        settings = ttk.LabelFrame(self, text="Search accuracy and indexing", padding=8); settings.pack(fill="x", padx=10, pady=4)
        ttk.Label(settings,text="Similarity threshold:").pack(side="left")
        ttk.Scale(settings,from_=0.25,to=0.65,variable=self.threshold,orient="horizontal",length=220).pack(side="left",padx=8)
        self.threshold_label = ttk.Label(settings,text=f"{self.threshold.get():.2f}")
        self.threshold_label.pack(side="left")
        self.threshold.trace_add("write", self._threshold_changed)
        ttk.Label(settings,text="Higher = fewer, stricter matches; lower = more matches and more false positives.",
                  wraplength=600).pack(side="left",padx=10)
        self.index_btn = ttk.Button(settings,text="Rebuild index",command=self.rebuild_index)
        self.index_btn.pack(side="right",padx=4)

        actions = ttk.Frame(self,padding=(10,6)); actions.pack(fill="x")
        self.search_btn = ttk.Button(actions,text="Find Person in Album",command=self.start_search)
        self.search_btn.pack(side="left")
        self.export_btn = ttk.Button(actions,text="Export checked matches",command=self.export_matches,state="disabled")
        self.export_btn.pack(side="left",padx=6)
        ttk.Button(actions,text="Select all",command=lambda:self._set_all_checks(True)).pack(side="left",padx=3)
        ttk.Button(actions,text="Select none",command=lambda:self._set_all_checks(False)).pack(side="left",padx=3)
        ttk.Button(actions,text="Show duplicate files",command=self.show_duplicates).pack(side="left",padx=3)

        progress = ttk.Frame(self,padding=(10,0)); progress.pack(fill="x")
        self.progress = ttk.Progressbar(progress,variable=self.progress_var,maximum=100)
        self.progress.pack(fill="x")
        ttk.Label(progress,textvariable=self.status_var).pack(anchor="w",pady=4)

        results = ttk.LabelFrame(self,text="Results — tick the photos you want to export",padding=6)
        results.pack(fill="both",expand=True,padx=10,pady=(2,10))
        controls=ttk.Frame(results); controls.pack(fill="x")
        ttk.Label(controls,text="Sort:").pack(side="left")
        self.sort_var=tk.StringVar(value="Best match")
        sort=ttk.Combobox(controls,textvariable=self.sort_var,values=["Best match","Best quality","Newest first","Oldest first","File name"],state="readonly",width=16)
        sort.pack(side="left",padx=4); sort.bind("<<ComboboxSelected>>",lambda _e:self.render_results())
        ttk.Label(controls,text="Filter:").pack(side="left",padx=(12,2))
        self.filter_var=tk.StringVar(value="All matches")
        filt=ttk.Combobox(controls,textvariable=self.filter_var,values=["All matches","High confidence (≥ 0.50)","Very high (≥ 0.60)"],state="readonly",width=23)
        filt.pack(side="left",padx=4); filt.bind("<<ComboboxSelected>>",lambda _e:self.render_results())
        self.count_label=ttk.Label(controls,text="0 matches"); self.count_label.pack(side="right")
        self.canvas=tk.Canvas(results,highlightthickness=0)
        scrollbar=ttk.Scrollbar(results,orient="vertical",command=self.canvas.yview)
        self.result_frame=ttk.Frame(self.canvas)
        self.result_frame.bind("<Configure>",lambda _e:self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_window=self.canvas.create_window((0,0),window=self.result_frame,anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind("<Configure>",lambda e:self.canvas.itemconfigure(self.canvas_window,width=e.width))
        self.canvas.pack(side="left",fill="both",expand=True); scrollbar.pack(side="right",fill="y")
        self.canvas.bind_all("<MouseWheel>",lambda e:self.canvas.yview_scroll(int(-1*(e.delta/120)),"units"))

    def _threshold_changed(self, *_):
        try: self.threshold_label.configure(text=f"{self.threshold.get():.2f}")
        except Exception: pass

    def _poll_queue(self):
        try:
            while True:
                item=self.msg_queue.get_nowait()
                kind=item[0]
                if kind=="status": self.status_var.set(item[1])
                elif kind=="progress": self.progress_var.set(item[1])
                elif kind=="done":
                    self.busy=False
                    self.search_btn.configure(state="normal"); self.index_btn.configure(state="normal")
                    self.change_folders_btn.configure(state="normal")
                    self.status_var.set(item[1])
                    if item[2] is not None:
                        self.matches=item[2]; self.render_results()
                        self.export_btn.configure(state="normal" if self.matches else "disabled")
                    if item[3]:
                        messagebox.showerror("FaceFinder",item[3],parent=self)
                elif kind=="profiles": self.profile_combo["values"]=item[1]
        except queue.Empty: pass
        self.after(150,self._poll_queue)

    def _worker_status(self,text): self.msg_queue.put(("status",text))
    def _worker_progress(self,value): self.msg_queue.put(("progress",value))

    def change_folders(self):
        if self.busy: return
        src=filedialog.askdirectory(title="Choose wedding album/source folder",initialdir=self.source_dir or str(Path.home()))
        if not src: return
        dst=filedialog.askdirectory(title="Choose destination for exported matches",initialdir=self.destination_dir or str(Path.home()))
        if not dst: return
        if Path(src).resolve()==Path(dst).resolve():
            messagebox.showwarning("Choose separate folders","Please choose a destination different from the source folder.",parent=self); return
        self.source_dir,self.destination_dir=src,dst
        self.config_data.update(source_dir=src,destination_dir=dst,threshold=float(self.threshold.get()))
        try: save_config(self.config_data)
        except Exception as exc: messagebox.showerror("Settings",str(exc),parent=self)
        self.source_label.configure(text=f"Source: {src}"); self.dest_label.configure(text=f"Destination: {dst}")
        self.status_var.set("Folder locations saved.")

    def add_reference_photos(self):
        paths=filedialog.askopenfilenames(title="Choose one or more clear photos of the person",
            filetypes=[("Supported photos","*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.heic *.heif"),("All files","*.*")])
        if not paths: return
        added=0; failures=[]
        for p in map(Path,paths):
            if p not in self.reference_paths:
                try:
                    pil=load_rgb(p)
                    faces=self._get_engine().detect_pil(pil)
                    if not faces:
                        failures.append(f"{p.name}: no face detected"); continue
                    if len(faces)==1: chosen=0
                    else:
                        boxes=[tuple(int(round(v)) for v in f.bbox) for f in faces]
                        picker=FaceChooser(self,pil,boxes,title=f"Choose face — {p.name}")
                        self.wait_window(picker)
                        if picker.result is None: continue
                        chosen=picker.result
                    self.reference_paths.append(p)
                    self.reference_embeddings.append(normalized_embedding(faces[chosen].embedding))
                    added+=1
                except Exception as exc: failures.append(f"{p.name}: {exc}")
        self._update_refs_label()
        if failures: messagebox.showwarning("Some references skipped","\n".join(failures[:12]),parent=self)
        if added: self.status_var.set(f"Added {added} reference photo(s). Multiple references can improve robustness.")

    def capture_webcam(self):
        cap=cv2.VideoCapture(0,cv2.CAP_DSHOW if os.name=="nt" else 0)
        if not cap.isOpened():
            cap.release(); messagebox.showerror("Webcam","Could not open webcam. Check camera permissions.",parent=self); return
        win=tk.Toplevel(self); win.title("Capture reference face"); win.geometry("800x650")
        ttk.Label(win,text="Position the person clearly, then click Capture.").pack(pady=5)
        preview=ttk.Label(win); preview.pack(expand=True,padx=8,pady=8)
        buttons=ttk.Frame(win); buttons.pack(pady=8)
        state={"running":True,"frame":None,"photo":None}
        def update():
            if not win.winfo_exists() or not state["running"]: return
            ok,frame=cap.read()
            if ok:
                frame=cv2.flip(frame,1)
                state["frame"]=frame.copy()
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                im=Image.fromarray(rgb); im.thumbnail((760,520),Image.Resampling.LANCZOS)
                state["photo"]=ImageTk.PhotoImage(im); preview.configure(image=state["photo"])
            win.after(30,update)
        def cleanup():
            state["running"]=False
            try: cap.release()
            except Exception: pass
            try: win.destroy()
            except Exception: pass
        def capture():
            frame=state["frame"]
            if frame is None: return
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            pil=Image.fromarray(rgb)
            faces=self._get_engine().detect_pil(pil)
            if not faces:
                messagebox.showwarning("No face","No face detected. Try better lighting and move closer.",parent=win); return
            if len(faces)>1:
                boxes=[tuple(int(round(v)) for v in f.bbox) for f in faces]
                picker=FaceChooser(win,pil,boxes,title="Choose webcam face")
                win.wait_window(picker)
                if picker.result is None: return
                idx=picker.result
            else: idx=0
            self.reference_embeddings.append(normalized_embedding(faces[idx].embedding))
            # Save captured image for provenance and future profile loading.
            ref_dir=APP_DIR/"facefinder_references"; ref_dir.mkdir(exist_ok=True)
            target=ref_dir/f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            pil.save(target,"JPEG",quality=95)
            self.reference_paths.append(target)
            self._update_refs_label()
            cleanup()
            self.status_var.set("Webcam reference captured.")
        ttk.Button(buttons,text="Capture",command=capture).pack(side="left",padx=5)
        ttk.Button(buttons,text="Cancel",command=cleanup).pack(side="left",padx=5)
        win.protocol("WM_DELETE_WINDOW",cleanup)
        update()

    def clear_references(self):
        self.reference_paths=[]; self.reference_embeddings=[]
        self._update_refs_label()

    def _update_refs_label(self):
        names=[p.name for p in self.reference_paths]
        self.refs_label.configure(text=(f"{len(names)} reference(s): "+", ".join(names[:5])+(" …" if len(names)>5 else "")) if names else "No reference photos selected")

    def _get_engine(self):
        if self.engine is None:
            self.engine=FaceEngine(status=self.status_var.set)
        return self.engine

    def save_profile(self):
        name=self.reference_name.get().strip()
        if not name:
            messagebox.showwarning("Profile name","Enter a name for this person.",parent=self); return
        if not self.reference_embeddings:
            messagebox.showwarning("No references","Add at least one reference photo first.",parent=self); return
        try:
            conn=connect_db()
            conn.execute("INSERT OR REPLACE INTO profiles(name,created_at,reference_paths) VALUES(?,?,?)",
                         (name,time.time(),json.dumps([str(p) for p in self.reference_paths])))
            conn.commit(); conn.close()
            self._load_profiles_into_combo()
            self.profile_combo.set(name)
            self.status_var.set(f"Saved profile: {name}")
        except Exception as exc: messagebox.showerror("Profile",str(exc),parent=self)

    def _load_profiles_into_combo(self):
        try:
            conn=connect_db()
            names=[r[0] for r in conn.execute("SELECT name FROM profiles ORDER BY name COLLATE NOCASE")]
            conn.close()
            self.profile_combo["values"]=names
        except Exception: pass

    def load_profile(self):
        name=self.profile_combo.get().strip()
        if not name: return
        try:
            conn=connect_db()
            row=conn.execute("SELECT reference_paths FROM profiles WHERE name=?",(name,)).fetchone()
            conn.close()
            if not row: return
            self.clear_references()
            paths=[]
            for s in json.loads(row[0]):
                p=Path(s)
                if p.exists(): paths.append(p)
            for p in paths:
                pil=load_rgb(p); faces=self._get_engine().detect_pil(pil)
                if not faces: continue
                if len(faces)==1: idx=0
                else:
                    picker=FaceChooser(self,pil,[tuple(int(round(v)) for v in f.bbox) for f in faces],title=f"Choose face — {p.name}")
                    self.wait_window(picker)
                    if picker.result is None: continue
                    idx=picker.result
                self.reference_paths.append(p)
                self.reference_embeddings.append(normalized_embedding(faces[idx].embedding))
            self.reference_name.set(name); self._update_refs_label()
            self.status_var.set(f"Loaded profile {name} with {len(self.reference_embeddings)} usable reference(s).")
        except Exception as exc: messagebox.showerror("Profile",str(exc),parent=self)

    def rebuild_index(self):
        self._start_background("index",rebuild=True)

    def start_search(self):
        if not self.reference_embeddings:
            messagebox.showwarning("Reference needed","Add one or more reference photos, or capture a webcam photo first.",parent=self); return
        if not self.source_dir or not Path(self.source_dir).is_dir():
            messagebox.showinfo("Choose folders","Please set your source and destination folders using Change Folder Locations.",parent=self)
            self.change_folders()
            if not self.source_dir or not Path(self.source_dir).is_dir(): return
        if not self.destination_dir or not Path(self.destination_dir).is_dir():
            messagebox.showinfo("Choose folders","Please set your source and destination folders using Change Folder Locations.",parent=self)
            self.change_folders()
            if not self.destination_dir or not Path(self.destination_dir).is_dir(): return
        self.config_data.update(source_dir=self.source_dir,destination_dir=self.destination_dir,threshold=float(self.threshold.get()))
        try: save_config(self.config_data)
        except Exception: pass
        self._start_background("search",rebuild=False)

    def _start_background(self,mode,rebuild=False):
        if self.busy: return
        if not self.source_dir or not Path(self.source_dir).is_dir():
            messagebox.showwarning("Source folder","Set a valid source folder first.",parent=self); return
        self.busy=True
        self.search_btn.configure(state="disabled"); self.index_btn.configure(state="disabled"); self.change_folders_btn.configure(state="disabled")
        self.progress_var.set(0)
        refs=[e.copy() for e in self.reference_embeddings]
        threshold=float(self.threshold.get())
        src=Path(self.source_dir)
        def worker():
            error=None; results=None
            conn=None
            try:
                engine=FaceEngine(status=self._worker_status)
                conn=connect_db()
                paths=get_image_paths(src)
                if not paths: raise RuntimeError("No supported images were found in the source folder.")
                if rebuild:
                    conn.execute("DELETE FROM faces"); conn.execute("DELETE FROM images"); conn.commit()
                existing={r[0] for r in conn.execute("SELECT path FROM images")}
               
                live={str(p.resolve()) for p in paths}
                stale=existing-live
                if stale:
                    conn.executemany("DELETE FROM faces WHERE image_path=?",[(p,) for p in stale])
                    conn.executemany("DELETE FROM images WHERE path=?",[(p,) for p in stale])
                    conn.commit()
                total=len(paths); changed=0
                for i,p in enumerate(paths,1):
                    _,did=index_image(conn,engine,p)
                    changed+=int(did)
                    if i%10==0 or i==total:
                        conn.commit()
                        self._worker_progress(100*i/total)
                        self._worker_status(f"Indexing {i:,}/{total:,} photos · {changed:,} new/changed")
                conn.commit()
                if mode=="index":
                    results=None
                    msg=f"Index ready. {total:,} photos checked; {changed:,} new or changed."
                else:
                    rows=load_indexed_embeddings(conn)
                    if not rows: raise RuntimeError("No faces were detected in the indexed photos.")
                 
                    ref_matrix=np.stack([normalized_embedding(e) for e in refs])
                    ref_mean=normalized_embedding(np.mean(ref_matrix,axis=0))
                    best_by_image={}
                    for j,(path,taken,emb,bbox,quality,det) in enumerate(rows,1):
                        scores=[cosine_similarity(emb,r) for r in ref_matrix]
                        score=max(scores+[cosine_similarity(emb,ref_mean)])
                        old=best_by_image.get(path)
                        if score>=threshold and (old is None or score>old.score):
                            best_by_image[path]=MatchRecord(path,score,quality,taken,0,bbox)
                        if j%100==0 or j==len(rows):
                            self._worker_progress(100*j/len(rows))
                            self._worker_status(f"Comparing face {j:,}/{len(rows):,}")
                    # Preserve the best matched face bbox/index for result review.
                    results=list(best_by_image.values())
                    # Exact duplicate grouping is computed from the index hash.
                    hash_rows=conn.execute("SELECT path,file_hash FROM images WHERE file_hash IS NOT NULL AND file_hash!=''").fetchall()
                    hash_to_paths={}
                    for p,h in hash_rows: hash_to_paths.setdefault(h,[]).append(p)
                    for m in results:
                        for h,ps in hash_to_paths.items():
                            if m.path in ps and len(ps)>1:
                                m.duplicate_group=h[:10]; break
                    results.sort(key=lambda m:m.score,reverse=True)
                    msg=f"Search complete: {len(results):,} matching photos from {total:,} images. Review before exporting."
            except Exception as exc:
                error=f"{exc}\n\n{traceback.format_exc()}"
                msg="Operation failed."
            finally:
                if conn:
                    try: conn.close()
                    except Exception: pass
                self.msg_queue.put(("done",msg,results,error))
        threading.Thread(target=worker,name="FaceFinderWorker",daemon=True).start()

    def _filtered_sorted_matches(self):
        matches=list(self.matches)
        f=self.filter_var.get()
        if f=="High confidence (≥ 0.50)": matches=[m for m in matches if m.score>=0.50]
        elif f=="Very high (≥ 0.60)": matches=[m for m in matches if m.score>=0.60]
        sort=self.sort_var.get()
        if sort=="Best quality": matches.sort(key=lambda m:(m.quality,m.score),reverse=True)
        elif sort=="Newest first": matches.sort(key=lambda m:m.taken_at,reverse=True)
        elif sort=="Oldest first": matches.sort(key=lambda m:m.taken_at)
        elif sort=="File name": matches.sort(key=lambda m:Path(m.path).name.lower())
        else: matches.sort(key=lambda m:m.score,reverse=True)
        return matches

    def render_results(self):
        for child in self.result_frame.winfo_children(): child.destroy()
        self.result_widgets=[]; self.thumbnail_refs=[]
        matches=self._filtered_sorted_matches()
        self.count_label.configure(text=f"{len(matches):,} shown / {len(self.matches):,} matches")
        for idx,m in enumerate(matches):
            card=ttk.Frame(self.result_frame,padding=5,relief="groove")
            card.grid(row=idx//3,column=idx%3,sticky="nsew",padx=4,pady=4)
            self.result_frame.grid_columnconfigure(idx%3,weight=1)
            selected=tk.BooleanVar(value=True)
            self.result_widgets.append((m,selected))
            try:
                pil=load_rgb(Path(m.path))
                pil.thumbnail(THUMB_SIZE,Image.Resampling.LANCZOS)

                from PIL import ImageDraw
                draw=ImageDraw.Draw(pil)
                x1,y1,x2,y2=m.bbox
                try:
                    with Image.open(m.path) as orig: ow,oh=orig.size
                    sx=pil.width/max(1,ow); sy=pil.height/max(1,oh)
                    draw.rectangle((x1*sx,y1*sy,x2*sx,y2*sy),outline="lime",width=2)
                except Exception: pass
                photo=ImageTk.PhotoImage(pil)
                self.thumbnail_refs.append(photo)
                btn=ttk.Button(card,image=photo,command=lambda mm=m:ImagePreview(self,Path(mm.path)))
                btn.pack()
            except Exception:
                ttk.Label(card,text="[Preview unavailable]").pack()
            ttk.Checkbutton(card,text="Export",variable=selected).pack(anchor="w")
            conf="Very high" if m.score>=0.60 else ("High" if m.score>=0.50 else "Review")
            ttk.Label(card,text=f"{conf} · similarity {m.score:.3f}").pack(anchor="w")
            ttk.Label(card,text=f"Quality {m.quality:.2f} · {m.taken_at[:19]}").pack(anchor="w")
            ttk.Label(card,text=Path(m.path).name,wraplength=300).pack(anchor="w")
            if m.duplicate_group: ttk.Label(card,text=f"Exact duplicate group {m.duplicate_group}").pack(anchor="w")
        for col in range(3): self.result_frame.grid_columnconfigure(col,weight=1)

    def _set_all_checks(self,value):
        for _,var in self.result_widgets: var.set(value)

    def export_matches(self):
        selected=[m for m,var in self.result_widgets if var.get()]
        if not selected:
            messagebox.showinfo("Nothing selected","Tick at least one result to export.",parent=self); return
        dst=Path(self.destination_dir)
        try:
            dst.mkdir(parents=True,exist_ok=True)
            person=(self.reference_name.get().strip() or "matched_person")
            safe="".join(c for c in person if c.isalnum() or c in " -_").strip() or "matched_person"
            out=dst/safe
            out.mkdir(parents=True,exist_ok=True)
            manifest=out/"facefinder_matches.csv"
            import csv
            copied=0; skipped=0
            with manifest.open("w",newline="",encoding="utf-8-sig") as f:
                writer=csv.writer(f)
                writer.writerow(["source_path","exported_path","similarity","quality","taken_at"])
                for m in selected:
                    src=Path(m.path)
                    try:
                        target=out/src.name
                        if target.exists():
                            target=out/f"{src.stem}_{hashlib.sha1(str(src).encode()).hexdigest()[:8]}{src.suffix}"
                        shutil.copy2(src,target)
                        writer.writerow([str(src),str(target),f"{m.score:.5f}",f"{m.quality:.4f}",m.taken_at])
                        copied+=1
                    except OSError: skipped+=1
            messagebox.showinfo("Export complete",f"Copied {copied} photos to:\n{out}\n\nManifest: {manifest}\nSkipped: {skipped}",parent=self)
            self.status_var.set(f"Exported {copied} photos.")
        except Exception as exc: messagebox.showerror("Export failed",str(exc),parent=self)

    def show_duplicates(self):
        try:
            conn=connect_db()
            groups=conn.execute("""
                SELECT file_hash,COUNT(*),GROUP_CONCAT(path,'\n')
                FROM images WHERE file_hash IS NOT NULL AND file_hash!=''
                GROUP BY file_hash HAVING COUNT(*)>1 ORDER BY COUNT(*) DESC
            """).fetchall()
            conn.close()
            win=tk.Toplevel(self); win.title("Exact duplicate files"); win.geometry("850x550")
            text=tk.Text(win,wrap="word"); text.pack(fill="both",expand=True)
            if not groups: text.insert("end","No exact duplicate files found in the current index.")
            else:
                for digest,count,paths in groups:
                    text.insert("end",f"{count} identical files · SHA-256 {digest[:16]}\n{paths}\n\n")
            text.configure(state="disabled")
        except Exception as exc: messagebox.showerror("Duplicates",str(exc),parent=self)


def main():
    app=FaceFinder()
    app.mainloop()


if __name__ == "__main__":
    main()
