# KNN_project - Scanned table of contents data extraction with geometrical positions


### Directories
`examples_data`- Examples of used data..

`openai` - Scripts and results for GPT models.

`training` - Scripts and results for florence and QWEN models (used on MetaCentrum).

`prompts` - Prompts used for GPT and QWEN models.


## full_match.py

This script processes manual annotations and aligns them with raw OCR output. It creates ground truth dataset in JSON format for further usage. Results can be visualised using the `visualise_matches.py` script.

To parse the annotations and OCR output, this script uses `pero_ocr.py` and `detector_parser.py` scripts, which were provided by project supervisor.


### Usage

**1. Basic execution (using defaults):**

```bash
python full_match.py
```

**2. Customizing the output (example):**
To run the script with a specific annotation file, a custom OCR directory, and save the results to a new folder:

```bash
python full_match.py -j data/my_custom_annotations.json -c data/my_ocr_files/ -o custom_out/
```

**3. Command Line Arguments:**

| Argument | Description |
| --- | --- |
| `-j`, `--json_annotations` | Path to the exported JSON annotations file |
| `-c`, `--ocr_dir` | Path to the OCR output directory |
| `-o`, `--output_dir` | Path to the output directory |




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



## evaluation.py

This script evaluates the performance of the model by comparing its output against ground truth data. 
It processes directories of JSON files, calculates evaluation metrics, and exports the results into a file (either as a CSV or a human-readable text format).


### Evaluated Metrics

The script calculates a comprehensive set of metrics divided into three main categories:

* **Text Quality:** Calculates Character Error Rate (CER) for extracted chapter names, chapter numbers, page numbers, and descriptions.
* **Bounding Box Localization:** Computes the F1-Score and Mean Intersection over Union (mIoU) for the detected bounding boxes of names, chapter numbers, page numbers, and descriptions.
* **JSON Structure & Parsing:** Evaluates the overall hierarchical structure using F1-Score for parsed chapters, and calculates strict accuracy for the assignment of page numbers, chapter numbers, and descriptions within the structure. These metrics are heavily dependent on the previous two.


### Usage

**1. Basic execution (using defaults, outputs as CSV into `eval.txt`):**

```bash
python evaluation.py
```

**2. Customizing the output with Pretty Print:**
To run the evaluation on custom directories and output the results in a human-readable format:

```bash
python evaluation.py -g custom_gt/ -m custom_model_outputs/ -e results_summary.txt --pretty
```

**3. Command Line Arguments:**

| Argument | Description | Default |
| --- | --- | --- |
| `-g`, `--gt_data` | Directory with ground truth JSONs | `data_gt/` |
| `-m`, `--model_data` | Directory with model output JSONs | `data_model/` |
| `-e`, `--eval_file` | Path to the output file (CSV or text) | `eval.txt` |
| `-p`, `--pretty` | Enable pretty print. Output is CSV if not inserted. | `False` (Flag) |
