
# DLC Behavior Analytics Suite

Graphical interface and experimental workflow for automated behavioral analysis of rodent tracking data from DeepLabCut (DLC). Specifically designed to quantify olfactory exploration (sniffing), locomotion, and anxiety-related behaviors with static or dynamic scent cues[cite: 5].

---

## Repository Structure

- **`docs/`**: Practical guides and documentation.
  * `Guida pratica a DeepLabCut.docx`: Step-by-step user guide for DeepLabCut model training and project setup in Italian[cite: 2].
  * `English version DLC guide.docx`: Practical guide for DeepLabCut in English[cite: 1].
  * `SIMBA_guide .docx`: Complete protocol for behavioral classification workflows using SimBA[cite: 3].
- **`gui/`**: Application and execution scripts.
  * `analisi_dlc_gui_ultimaversione_2.py`: Main PyQt6 application for automated behavioral extraction, dynamic/static ROI management, timeline verification, and metric consolidation[cite: 5].
  * `start_GUI_dlc.command`: Launch script for macOS[cite: 6].
- **`requirements/`**: Dependencies.
  * `requirements.txt`: List of required Python packages[cite: 4].

---

## Features

- **Olfactory Investigation (Sniffing):**
  * Vectorial head-to-stimulus angle calculation[cite: 5].
  * Proximity distance and nose-contact boundary metrics[cite: 5].
  * Support for both stationary and moving/dynamic scent stimuli via keyframe-interpolated ROIs[cite: 5].
- **Locomotion & Anxiety Markers:**
  * Freezing detection with duration filtering[cite: 5].
  * Stretch-Attend Postures (SAP) based on body length extension and velocity[cite: 5].
  * Thigmotaxis analysis relative to the defined arena boundary[cite: 5].
- **Automated Data Consolidation:**
  * Logs metrics per animal into a consolidated Excel spreadsheet (`Consolidated_Behavioral_Data.xlsx`)[cite: 5].
  * Generates high-resolution figures: temporal sequencing rasters, trajectory navigation plots, and freezing-excluded occupancy heatmaps[cite: 5].
- **Integrated Video Timeline Player:**
  * Manual review, zoom, and frame-by-frame correction of detected behavioral bouts[cite: 5].

---

## Installation

1. Make sure Python 3.9+ is installed[cite: 3].
2. Install the required Python packages from the `requirements/` directory:

```bash
pip install -r requirements/requirements.txt

```

---

## How to Run

You can launch the interface using the Python script inside `gui/`:

```bash
python3 gui/analisi_dlc_gui_ultimaversione_2.py

```

Or on macOS, double-click:

```bash
gui/start_GUI_dlc.command

```
