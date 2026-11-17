
"""Browser-based manual correction of flagged droplet masks.

Run:
    python src/roi_annotation_server.py

Then open:
    http://localhost:8765

Actions:
    Accept existing automatic mask
    Draw and save a corrected polygon
    Skip an image
    Include previously skipped images

Only flagged train/validation images are shown.
Test images are never opened.
Original photographs are never modified.
"""

from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs, quote
import json
import mimetypes
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from skimage.draw import polygon
from skimage.measure import find_contours, label, regionprops


ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = ROOT / "data/interim/roi_mask_manifest.csv"

POLYGON_DIR = ROOT / "data/interim/manual_roi_polygons"

HOST = "127.0.0.1"
PORT = 8765

DEVELOPMENT_ROLES = {"train", "validation"}
CROP_MARGIN_FRACTION = 0.06

manifest = None


# =========================================================
# HTML APPLICATION
# =========================================================

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Droplet ROI correction</title>

<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    padding: 18px;
    background: #f3f4f6;
    color: #202124;
    font-family: Arial, sans-serif;
}

header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 15px;
    flex-wrap: wrap;
    margin-bottom: 16px;
}

h2 { margin: 0 0 6px; }

.small {
    color: #626b78;
    font-size: 13px;
}

.layout {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 290px;
    gap: 18px;
}

.panel {
    background: white;
    border-radius: 10px;
    padding: 14px;
    box-shadow: 0 2px 8px #00000012;
}

.canvas-area {
    min-height: 300px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: auto;
    background: #242424;
    border-radius: 8px;
    padding: 10px;
}

canvas {
    display: block;
    max-width: 100%;
    max-height: 75vh;
    width: auto;
    height: auto;
    cursor: crosshair;
}

button {
    padding: 10px 12px;
    border: 0;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
    margin: 4px 2px;
}

button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}

.primary { background: #1769aa; color: white; }
.accept { background: #16834a; color: white; }
.skip { background: #e5e7eb; color: #202124; }
.danger { background: #b42318; color: white; }
.secondary { background: #e8eef5; color: #263445; }

button:hover:not(:disabled) { filter: brightness(0.94); }

.meta {
    overflow-wrap: anywhere;
    line-height: 1.7;
    font-size: 13px;
}

.controls {
    margin-top: 12px;
}

label {
    display: block;
    margin: 12px 0;
    font-size: 13px;
}

.status {
    min-height: 25px;
    margin-top: 10px;
    color: #46515f;
    font-size: 13px;
}

.instructions {
    line-height: 1.55;
    font-size: 13px;
}

.instructions li { margin-bottom: 7px; }

#empty {
    display: none;
    background: white;
    padding: 30px;
    border-radius: 10px;
}

@media (max-width: 850px) {
    .layout { grid-template-columns: 1fr; }
    canvas { max-height: 65vh; }
}
</style>
</head>

<body>

<header>
    <div>
        <h2>Droplet ROI correction</h2>
        <div class="small">
            Development images only · Original photographs remain unchanged
        </div>
    </div>
    <div id="counter" class="small">Loading...</div>
</header>

<div id="empty">
    <h3>No pending flagged images</h3>
    <p>All currently available flagged images have been reviewed or skipped.</p>
</div>

<div id="app" class="layout">

    <section class="panel">
        <div id="imageTitle" class="meta"></div>

        <div class="canvas-area">
            <canvas id="canvas"></canvas>
        </div>

        <div class="controls">
            <button class="secondary" id="undo">Undo last point</button>
            <button class="secondary" id="clear">Clear polygon</button>
            <button class="secondary" id="overlayToggle">
                Hide existing mask overlay
            </button>
        </div>

        <div class="controls">
            <button class="accept" id="accept">
                Accept existing mask
            </button>

            <button class="primary" id="save" disabled>
                Save manual contour
            </button>

            <button class="skip" id="skip">
                Skip
            </button>
        </div>

        <div class="status" id="status"></div>
    </section>

    <aside class="panel">
        <h3>Image information</h3>
        <div id="info" class="meta"></div>

        <hr>

        <h3>How to correct</h3>

        <ol class="instructions">
            <li>Inspect the existing green mask and red contour.</li>
            <li>Press <b>Accept existing mask</b> if the boundary is adequate.</li>
            <li>Otherwise click around the actual outer droplet boundary.</li>
            <li>Use at least three points, following the perimeter.</li>
            <li>Press <b>Save manual contour</b>.</li>
            <li>Use Skip when the contour is difficult to determine.</li>
        </ol>

        <hr>

        <label>
            <input type="checkbox" id="includeSkipped">
            Include previously skipped images
        </label>

        <button class="secondary" id="reload">
            Reload pending images
        </button>

        <p class="small">
            Green indicates the existing mask. Red indicates its existing
            contour. Yellow indicates your new polygon.
        </p>

        <p class="small">
            A small boundary error is acceptable. Try to include the
            entire droplet without including substantial background.
        </p>
    </aside>

</div>

<script>
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const statusEl = document.getElementById("status");
const counterEl = document.getElementById("counter");

let items = [];
let index = 0;
let current = null;

let vertices = [];
let showOverlay = true;

const originalImage = new Image();
const overlayImage = new Image();


function setStatus(message) {
    statusEl.textContent = message;
}


function fileURL(path) {
    return "/file?path=" + encodeURIComponent(path);
}


async function requestJSON(url, options = {}) {
    const response = await fetch(url, options);

    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || "Request failed");
    }

    return data;
}


async function loadItems() {

    const includeSkipped = document.getElementById(
        "includeSkipped"
    ).checked;

    const url = "/api/items?include_skipped=" +
        (includeSkipped ? "1" : "0");

    try {
        const data = await requestJSON(url);

        items = data.items;
        index = 0;

        await loadCurrent();

    } catch (error) {
        setStatus(error.message);
    }
}


async function loadCurrent() {

    vertices = [];
    current = null;

    if (items.length === 0) {
        document.getElementById("app").style.display = "none";
        document.getElementById("empty").style.display = "block";

        counterEl.textContent = "0 pending images";
        return;
    }

    document.getElementById("app").style.display = "grid";
    document.getElementById("empty").style.display = "none";

    if (index < 0) index = 0;
    if (index >= items.length) index = items.length - 1;

    const id = items[index].image_id;

    try {
        current = await requestJSON(
            "/api/item?image_id=" + encodeURIComponent(id)
        );

        originalImage.onload = redraw;
        originalImage.onerror = () => {
            setStatus("Could not load original image");
        };

        overlayImage.onload = redraw;

        originalImage.src = fileURL(current.image_path);
        overlayImage.src = fileURL(current.overlay_path);

        document.getElementById("imageTitle").textContent =
            current.image_id;

        document.getElementById("info").innerHTML = `
            <b>Instance:</b> ${current.instance_id}<br>
            <b>Tube:</b> ${current.tube_id}<br>
            <b>Concentration:</b> ${current.conc}%<br>
            <b>Device:</b> ${current.device}<br>
            <b>Lamp:</b> ${current.lamp}<br>
            <b>Role:</b> ${current.role}<br>
            <b>Original flag:</b> ${current.failure_reason || "None"}<br>
            <b>Mask area:</b> ${
                current.area_fraction === null
                ? "Unavailable"
                : (100 * current.area_fraction).toFixed(2) + "%"
            }
        `;

        counterEl.textContent =
            `Image ${index + 1} of ${items.length}`;

        document.getElementById("save").disabled = true;

        setStatus("Click the perimeter to draw a replacement polygon.");

    } catch (error) {
        setStatus(error.message);
    }
}


function redraw() {

    if (!current || !originalImage.complete ||
        !originalImage.naturalWidth) return;

    canvas.width = originalImage.naturalWidth;
    canvas.height = originalImage.naturalHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(originalImage, 0, 0);

    if (showOverlay && overlayImage.complete &&
        overlayImage.naturalWidth) {

        ctx.save();
        ctx.globalAlpha = 0.55;
        ctx.drawImage(overlayImage, 0, 0);
        ctx.restore();
    }

    if (vertices.length > 0) {

        ctx.beginPath();
        ctx.moveTo(vertices[0].x, vertices[0].y);

        for (let i = 1; i < vertices.length; i++) {
            ctx.lineTo(vertices[i].x, vertices[i].y);
        }

        ctx.strokeStyle = "#ffff00";
        ctx.lineWidth = Math.max(
            2, Math.round(canvas.width / 600)
        );

        ctx.stroke();

        for (const p of vertices) {
            ctx.beginPath();
            ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
            ctx.fillStyle = "#ffff00";
            ctx.fill();
        }
    }

    document.getElementById("save").disabled =
        vertices.length < 3;
}


canvas.addEventListener("click", event => {

    if (!current) return;

    const rect = canvas.getBoundingClientRect();

    const x = (
        event.clientX - rect.left
    ) * canvas.width / rect.width;

    const y = (
        event.clientY - rect.top
    ) * canvas.height / rect.height;

    vertices.push({x, y});

    redraw();

    setStatus(`${vertices.length} polygon points added.`);
});


document.getElementById("undo").onclick = () => {
    vertices.pop();
    redraw();
};

document.getElementById("clear").onclick = () => {
    vertices = [];
    redraw();
};

document.getElementById("overlayToggle").onclick = () => {

    showOverlay = !showOverlay;

    document.getElementById("overlayToggle").textContent =
        showOverlay
        ? "Hide existing mask overlay"
        : "Show existing mask overlay";

    redraw();
};


async function performAction(action) {

    if (!current) return;

    if (action === "manual" && vertices.length < 3) {
        setStatus("Draw at least three polygon points first.");
        return;
    }

    const payload = {
        image_id: current.image_id,
        action: action,
        vertices: action === "manual" ? vertices : []
    };

    try {
        setStatus("Saving...");

        await requestJSON("/api/action", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const completedId = current.image_id;

        items = items.filter(
            item => item.image_id !== completedId
        );

        if (index >= items.length) {
            index = items.length - 1;
        }

        await loadCurrent();

    } catch (error) {
        setStatus("Error: " + error.message);
    }
}


document.getElementById("accept").onclick = () => {
    performAction("accept");
};

document.getElementById("save").onclick = () => {
    performAction("manual");
};

document.getElementById("skip").onclick = () => {
    performAction("skip");
};

document.getElementById("reload").onclick = loadItems;

document.getElementById("includeSkipped").onchange = loadItems;


document.addEventListener("keydown", event => {

    if (event.target.tagName === "INPUT") return;

    const key = event.key.toLowerCase();

    if (key === "a") performAction("accept");

    if (key === "s") performAction("skip");

    if (key === "u") {
        vertices.pop();
        redraw();
    }

    if (key === "escape") {
        vertices = [];
        redraw();
    }
});


loadItems();
</script>

</body>
</html>
"""


# =========================================================
# FILE UTILITIES
# =========================================================

def safe_filename(value):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    )


def safe_project_path(value):
    path = (ROOT / str(value)).resolve()

    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Path escapes project root")

    allowed_roots = [
        ROOT / "data/raw",
        ROOT / "data/interim",
        ROOT / "results/figures",
    ]

    allowed = any(
        path.is_relative_to(p.resolve())
        for p in allowed_roots
    )

    if not allowed:
        raise ValueError("File is outside allowed image directories")

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def get_row(image_id):
    matches = manifest[
        manifest["image_id"].astype(str) == str(image_id)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely identify image_id: {image_id}"
        )

    return matches.index[0], matches.iloc[0]


def relative_path(value):
    return str(value).replace("\\", "/")


def row_item(row):
    return {
        "image_id": str(row["image_id"]),
        "instance_id": int(row["instance_id"]),
        "tube_id": str(row["tube_id"]),
        "conc": float(row["conc"]),
        "device": str(row["device"]),
        "lamp": str(row["lamp"]),
        "role": str(row["role"]),
        "failure_reason": str(row.get("failure_reason", "") or ""),
        "area_fraction": (
            float(row["area_fraction"])
            if pd.notna(row.get("area_fraction", np.nan))
            else None
        ),
    }


# =========================================================
# MASK WRITING
# =========================================================

def save_manual_polygon(row, vertices):
    """Replace one image's mask with the browser-drawn polygon."""

    image_path = safe_project_path(row["image_path"])

    with Image.open(image_path) as im:
        im.load()
        rgb = np.asarray(
            im.convert("RGB"),
            dtype=np.uint8,
        )

    height, width = rgb.shape[:2]

    points = np.asarray(vertices, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Polygon coordinates must be pairs")

    if len(points) < 3:
        raise ValueError("At least three polygon points are required")

    if not np.isfinite(points).all():
        raise ValueError("Polygon contains invalid coordinates")

    xs = np.clip(points[:, 0], 0, width - 1)
    ys = np.clip(points[:, 1], 0, height - 1)

    rr, cc = polygon(
        ys,
        xs,
        shape=(height, width),
    )

    mask = np.zeros(
        (height, width),
        dtype=bool,
    )

    mask[rr, cc] = True

    if not mask.any():
        raise ValueError("The drawn polygon is empty")

    ys_mask, xs_mask = np.where(mask)

    area_fraction = float(mask.mean())

    if area_fraction < 0.001:
        raise ValueError("Polygon is too small")

    if area_fraction > 0.90:
        raise ValueError(
            "Polygon covers over 90% of the image. Check the boundary."
        )

    # Save using the existing manifest paths.
    mask_path = safe_project_path(row["mask_path"])
    crop_path = (
        ROOT / str(row["crop_path"])
    ).resolve()

    overlay_path = (
        ROOT / str(row["overlay_path"])
    ).resolve()

    for path in [crop_path, overlay_path]:
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError("Output path escapes project root")

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    alpha = mask.astype(np.uint8) * 255

    rgba = np.dstack([rgb, alpha])
    rgba_image = Image.fromarray(rgba)

    rgba_image.save(mask_path)

    # Crop around the polygon. Original pixel values are preserved.
    margin = int(
        0.06 * max(width, height)
    )

    left = max(0, int(xs_mask.min()) - margin)
    right = min(width, int(xs_mask.max()) + 1 + margin)

    top = max(0, int(ys_mask.min()) - margin)
    bottom = min(height, int(ys_mask.max()) + 1 + margin)

    rgba_image.crop(
        (left, top, right, bottom)
    ).save(crop_path)

    # Create green overlay with red contour.
    overlay = rgb.astype(np.float32).copy()

    green = np.array(
        [0, 255, 0],
        dtype=np.float32,
    )

    overlay[mask] = (
        0.65 * overlay[mask]
        + 0.35 * green
    )

    overlay = np.clip(
        overlay, 0, 255
    ).astype(np.uint8)

    overlay_image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(overlay_image)

    for contour in find_contours(
        mask.astype(float),
        level=0.5,
    ):
        line = [
            (int(x), int(y))
            for y, x in contour
        ]

        if len(line) >= 2:
            draw.line(
                line,
                fill=(255, 0, 0),
                width=max(2, round(max(width, height) / 700)),
            )

    overlay_image.save(
        overlay_path,
        quality=90,
    )

    # Save exact polygon points for reproducibility.
    POLYGON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    polygon_path = (
        POLYGON_DIR
        / f"{safe_filename(row['image_id'])}.json"
    )

    polygon_path.write_text(
        json.dumps(
            {
                "image_id": str(row["image_id"]),
                "method": "manual_browser_polygon",
                "vertices_xy": [
                    [float(x), float(y)]
                    for x, y in points
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    region = regionprops(label(mask))[0]

    cy, cx = region.centroid

    centre_y = (height - 1) / 2
    centre_x = (width - 1) / 2

    centre_distance = float(
        np.sqrt(
            ((cx - centre_x) / width) ** 2
            + ((cy - centre_y) / height) ** 2
        )
    )

    manifest_row = manifest.index[
        manifest["image_id"].astype(str)
        == str(row["image_id"])
    ][0]

    updates = {
        "mask_status": "manual_corrected",
        "failure_reason": "",
        "review_action": "manual_polygon",
        "manual_polygon_path": relative_path(
            polygon_path.relative_to(ROOT)
        ),
        "area_fraction": area_fraction,
        "centre_distance_normalised": centre_distance,
        "solidity": float(region.solidity),
        "bbox_aspect_ratio": float(
            (region.bbox[3] - region.bbox[1])
            / max(1, region.bbox[2] - region.bbox[0])
        ),
        "touches_image_edge": bool(
            left == 0 or top == 0
            or right == width or bottom == height
        ),
        "crop_left": left,
        "crop_top": top,
        "crop_right": right,
        "crop_bottom": bottom,
    }

    for column, value in updates.items():
        if column not in manifest.columns:
            manifest[column] = ""

        manifest.at[manifest_row, column] = value

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    return area_fraction


# =========================================================
# HTTP SERVER
# =========================================================

class AnnotationHandler(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")

        self.send_response(status)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header(
            "Content-Length",
            str(len(body)),
        )
        self.send_header(
            "Cache-Control",
            "no-store",
        )
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):

        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":

            body = HTML.encode("utf-8")

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8",
            )
            self.send_header(
                "Content-Length",
                str(len(body)),
            )
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/items":

            include_skipped = (
                params.get("include_skipped", ["0"])[0] == "1"
            )

            pending = manifest[
                (manifest["mask_status"] == "flagged")
                & manifest["role"].isin(DEVELOPMENT_ROLES)
            ].copy()

            if not include_skipped and "review_action" in pending:
                pending = pending[
                    pending["review_action"].fillna("")
                    != "skipped"
                ]

            items = [
                row_item(row)
                for _, row in pending.iterrows()
            ]

            self.send_json({
                "items": items,
                "count": len(items),
            })
            return

        if path == "/api/item":

            image_id = params.get("image_id", [""])[0]

            try:
                _, row = get_row(image_id)

                self.send_json({
                    "image_id": str(row["image_id"]),
                    "image_path": relative_path(row["image_path"]),
                    "overlay_path": relative_path(row["overlay_path"]),
                    "mask_path": relative_path(row["mask_path"]),
                    "instance_id": int(row["instance_id"]),
                    "tube_id": str(row["tube_id"]),
                    "conc": float(row["conc"]),
                    "device": str(row["device"]),
                    "lamp": str(row["lamp"]),
                    "role": str(row["role"]),
                    "failure_reason": str(
                        row.get("failure_reason", "") or ""
                    ),
                    "area_fraction": (
                        float(row["area_fraction"])
                        if pd.notna(row.get("area_fraction", np.nan))
                        else None
                    ),
                })

            except Exception as exc:
                self.send_json(
                    {"error": str(exc)},
                    status=400,
                )

            return

        if path == "/file":

            try:
                file_value = params.get("path", [""])[0]
                file_path = safe_project_path(file_value)

                content_type, _ = mimetypes.guess_type(
                    str(file_path)
                )

                content_type = (
                    content_type or "application/octet-stream"
                )

                body = file_path.read_bytes()

                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    content_type,
                )
                self.send_header(
                    "Content-Length",
                    str(len(body)),
                )
                self.send_header(
                    "Cache-Control",
                    "no-store",
                )
                self.end_headers()
                self.wfile.write(body)

            except Exception as exc:
                self.send_json(
                    {"error": str(exc)},
                    status=404,
                )

            return

        self.send_json(
            {"error": "Not found"},
            status=404,
        )

    def do_POST(self):

        if urlparse(self.path).path != "/api/action":
            self.send_json(
                {"error": "Not found"},
                status=404,
            )
            return

        try:
            length = int(
                self.headers.get("Content-Length", "0")
            )

            if length <= 0 or length > 5_000_000:
                raise ValueError("Invalid request size")

            payload = json.loads(
                self.rfile.read(length).decode("utf-8")
            )

            image_id = str(payload.get("image_id", ""))
            action = payload.get("action")

            manifest_index, row = get_row(image_id)

            # Never accept actions on test images.
            if row["role"] not in DEVELOPMENT_ROLES:
                raise ValueError(
                    "Test images cannot be modified in this tool"
                )

            if row["mask_status"] != "flagged":
                raise ValueError(
                    "This image is no longer flagged"
                )

            if action == "accept":

                manifest.at[
                    manifest_index, "mask_status"
                ] = "manual_accepted"

                manifest.at[
                    manifest_index, "failure_reason"
                ] = ""

                manifest.at[
                    manifest_index, "review_action"
                ] = "accepted_auto"

                area = float(row["area_fraction"])

            elif action == "manual":

                vertices = payload.get("vertices", [])

                area = save_manual_polygon(
                    row,
                    vertices,
                )

            elif action == "skip":

                if "review_action" not in manifest.columns:
                    manifest["review_action"] = ""

                manifest.at[
                    manifest_index, "review_action"
                ] = "skipped"

                area = (
                    float(row["area_fraction"])
                    if pd.notna(row["area_fraction"])
                    else None
                )

            else:
                raise ValueError(
                    "Action must be accept, manual or skip"
                )

            manifest.to_csv(
                MANIFEST_PATH,
                index=False,
            )

            self.send_json({
                "success": True,
                "image_id": image_id,
                "action": action,
                "area_fraction": area,
            })

        except Exception as exc:
            self.send_json(
                {"error": str(exc)},
                status=400,
            )

    def log_message(self, format, *args):
        print(
            "[annotation server] " + (format % args)
        )


# =========================================================
# START
# =========================================================

def main():

    global manifest

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    required = {
        "image_id",
        "image_path",
        "mask_path",
        "crop_path",
        "overlay_path",
        "mask_status",
        "failure_reason",
        "role",
        "instance_id",
        "tube_id",
        "conc",
        "device",
        "lamp",
    }

    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing)}"
        )

    if manifest["image_id"].duplicated().any():
        raise ValueError(
            "Duplicate image_id values in manifest"
        )

    for col in [
        "review_action",
        "manual_polygon_path",
    ]:
        if col not in manifest.columns:
            manifest[col] = ""

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    pending = manifest[
        (manifest["mask_status"] == "flagged")
        & manifest["role"].isin(DEVELOPMENT_ROLES)
    ]

    print("\nDROPLET ROI BROWSER ANNOTATION")
    print("=" * 55)
    print("Flagged development images:", len(pending))
    print("Manifest:", MANIFEST_PATH)
    print()
    print("Open this address in your Windows browser:")
    print(f"    http://localhost:{PORT}")
    print()
    print("Keep this terminal open while using the browser.")
    print("Press Ctrl+C here when finished.")
    print()

    server = HTTPServer(
        (HOST, PORT),
        AnnotationHandler,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAnnotation server stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
