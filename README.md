# KNN_project

KNN magnum opus

## visualize_matches.py

This script provides a visual validation of the matching process between OCR results and annotations.
It reads the structured JSON output and draws bounding boxes (polygons) directly onto the original source images to verify that text fields like chapter names, numbers, and page references are correctly localized.

### Prerequisites

Ensure you have OpenCV installed in your virtual environment:

```bash
pip install opencv-python

```

### Usage

You can run the script using default paths defined in the code or specify your own directories.

**1. Using defaults:**

```bash
python visualize_results.py

```

**2. Specifying custom directories:**

```bash
python visualize_results.py --json_dir ./my_json_out --image_dir ./source_images --output_dir ./visualized_results

```

**3. Displaying Help:**
To see all available options and default values, use the `-h` or `--help` flag:

```bash
python visualize_results.py --help

```

### Visualization Legend

The script uses the following BGR color scheme to distinguish between different elements:

| Element | Color | BGR Value |
| --- | --- | --- |
| **Name** | Red | `(0, 0, 255)` |
| **Chapter Number** | Blue | `(255, 0, 0)` |
| **Page Number** | Green | `(0, 255, 0)` |
| **Description** | Yellow | `(0, 255, 255)` |
