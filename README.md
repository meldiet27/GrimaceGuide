# Grimace Guide

An application for evaluating grimace scores in cats using computer vision and landmark detection.

# Folder Structure

```
GrimaceGuide/
│
├── main.py                  # Application entry point
├── requirements.txt         # Dependencies
├── README.md                # Documentation
│
├── grimaceguide/           # Package directory
│   ├── __init__.py         # Package initializer
│   ├── config.py           # Configuration settings
│   ├── database.py         # Database management
│   ├── api.py              # API integration
│   ├── fgsScoreCalc.py     # FGS Score Calculation
│   ├── utils.py            # Utility functions
│   │
│   ├── ui/                 # UI components
│   │   ├── __init__.py     # UI package initializer
│   │   ├── app.py          # Main application class
│   │   ├── widgets.py      # Custom widgets
│   │   └── popups.py       # Custom popup dialogs
│   │
│   └── resources/          # Application resources
│       └── __init__.py     # Resources package initializer
│
└── imagesGUI/              # Guide images directory
```

## Features

* Upload and display images
* Process images with both API and local model
* Display and visualize grimace scores
* Store processing history in a database

### General Description of Input Data

The input samples consist of still images of domestic cats captured in well-lit environments to ensure clear visibility of facial landmarks. Each image provides a frontal view of the cat’s face to facilitate detailed analysis of the five Feline Grimace Scale (FGS) action units:

1. Ear position
2. Eye openness
3. Muzzle shape
4. Whisker position
5. Head alignment

The FGS assigns a score of 0–2 for each AU (0 = absent, 1 = moderate/uncertain, 2 = obvious), for a maximum total of 10. Scores ≥ 4 (i.e., ≥ 0.39 when normalized) suggest likely acute pain requiring intervention. The FGS is validated for use by both veterinary professionals and caregivers for assessing acute pain in cats.

### Feature Processing Details

For each action unit, the program analyzes specific visual cues using computer vision techniques and landmark-based processing:

1. **Ears**

   * *Scoring:* 0 = ears forward; 1 = slightly pulled apart; 2 = flattened and rotated outward
   * *Processing Focus:* Edge detection and geometric landmark extraction to compute ear tip orientation and angle.

2. **Eyes**

   * *Scoring:* 0 = fully open; 1 = partially open; 2 = squinted
   * *Processing Focus:* Contour detection and measurement of vertical eye aperture and eyelid curvature.

3. **Muzzle**

   * *Scoring:* 0 = relaxed round shape; 1 = mildly tense; 2 = tense elliptical shape
   * *Processing Focus:* Contour analysis and shape descriptors (e.g., aspect ratio, solidity) to quantify muzzle tension.

4. **Whiskers**

   * *Scoring:* 0 = loose and curved; 1 = slightly curved/straight; 2 = straight and forward-pointing
   * *Processing Focus:* High-contrast line detection and curvature analysis to track whisker orientation.

5. **Head Position**

   * *Scoring:* 0 = head above shoulder line; 1 = aligned with shoulder line; 2 = head below shoulder line or tilted down
   * *Processing Focus:* Spatial analysis relative to a reference shoulder line to determine vertical head displacement.

A normalized FGS score is computed by summing the AU scores and dividing by the maximum possible (10). A normalized score ≥ 0.39 triggers an alert for potential pain.

### Examples

1. Presence of 0 Action Units (AUs)

   * Ears facing forward
   * Eyes fully open
   * Muzzle relaxed with circular shape
   * Whiskers relaxed and curved
   * Head above the shoulder line
   * *Expectation:* Software should detect 0 AUs and highlight each relaxed feature.

2. Presence of 1 AU

   * Ears slightly pulled apart
   * Eyes partially open (percentage of openness measured)
   * Muzzle mildly tense or uncertain
   * Whisker curvature degree measured
   * Head alignment assessed relative to shoulders
   * *Expectation:* Software should detect 1 AU and annotate each metric (ear separation, eye openness, muzzle tension, whisker angle, head alignment).

3. Presence of 2 AUs

   * Ears flattened and rotated outward
   * Eyes squinted
   * Muzzle tense with an elliptical shape
   * Whiskers straight and moved forward
   * Head below the shoulder line (chin toward chest)
   * *Expectation:* Software should detect 2 AUs and mark each action unit accordingly.


## Installation

1. Clone this repository

   ```bash
   gh repo clone ProjectLantier/GrimaceGuide
   ```
2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:

   ```bash
   .\venv\Scripts\activate   # Windows
   source venv/bin/activate    # macOS/Linux
   ```
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:

```bash
python main.py
```

## References and Datasets

* **Feline Grimace Scale website**: [https://www.felinegrimacescale.com/about](https://www.felinegrimacescale.com/about)
* **Fully automated deep learning models with smartphone applicability for prediction of pain using the Feline Grimace Scale**: [https://www.nature.com/articles/s41598-023-49031-2](https://www.nature.com/articles/s41598-023-49031-2)
* **Explainable automated pain recognition in cats**: [https://www.nature.com/articles/s41598-023-35846-6?fromPaywallRec=false](https://www.nature.com/articles/s41598-023-35846-6?fromPaywallRec=false)
* **Researchers tap AI to read pain in animals**: [https://news.vin.com/default.aspx?pid=210\&Id=12153545\&f5=1](https://news.vin.com/default.aspx?pid=210&Id=12153545&f5=1)
* **Automated Detection of Cat Facial Landmarks**: [https://link.springer.com/article/10.1007/s11263-024-02006-w](https://link.springer.com/article/10.1007/s11263-024-02006-w)
* **catFLW dataset**: [https://www.kaggle.com/datasets/georgemartvel/catflw/discussion/533260](https://www.kaggle.com/datasets/georgemartvel/catflw/discussion/533260)
* **CatFLW: Cat Facial Landmarks in the Wild Dataset**: [https://arxiv.org/abs/2305.04232](https://arxiv.org/abs/2305.04232)
