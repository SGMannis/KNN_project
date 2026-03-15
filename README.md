# KNN_project

KNN magnum opus

## visualize_matches.py

This script provides visual validation of the matching process by drawing results and annotations onto a dual-pane canvas.
It allows you to compare the original document (left) with a clean structural reconstruction (right).

### Features

* **Dual-Pane View:** Generates an extended image showing the source document alongside a white "extension" page.
* **Hierarchical Shading:** Automatically adjusts color contrast and line thickness based on subchapter depth (Level 0, Level 1, etc.).
* **Granular Control:** Toggle bounding boxes, text, and alignment lines independently for each side of the image.

### Prerequisites

You need **OpenCV**, **NumPy**, and **Pillow** installed in your virtual environment:

```bash
pip install opencv-python numpy pillow

```

### Usage

**1. Basic execution (using defaults):**

```bash
python visualize_matches.py

```

**2. Customizing the output (example):**
To show only bounding boxes on the original page and only text/lines on the extension:

```bash
python visualize_matches.py --p_bbox --no-p_text --no-p_line --no-e_bbox --e_text --e_line

```

**3. Command Line Arguments:**

| Argument | Description | Default |
| --- | --- | --- |
| `-j`, `--json_dir` | Path to matched JSON directory | `out/` |
| `-i`, `--image_dir` | Path to source images | `data/.../images/` |
| `--p_bbox`, `--p_text`, `--p_line` | Toggle elements on the **Original Page** | BBox/Line: ON |
| `--e_bbox`, `--e_text`, `--e_line` | Toggle elements on the **Extension** | Text: ON |

### Visualization Legend (Level 0)

| Element | Primary Color | RGB (Pillow) |
| --- | --- | --- |
| **Name** | Red | `(255, 0, 0)` |
| **Chapter Number** | Blue | `(0, 0, 255)` |
| **Page Number** | Green | `(0, 255, 0)` |
| **Description** | Yellow | `(255, 255, 0)` |
