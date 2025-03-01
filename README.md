# Stylometric Analysis Toolkit

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![CI Status](https://github.com/dayanadithyan/stylometry-dh/actions/workflows/python-ci.yml/badge.svg)

A Python package for analyzing writing style through quantitative metrics.

Please read: This is still in alpha. Analytics are fine but a ways to go before releasing a functional library.

Update 2/27: First features working at basic level
![image](https://github.com/user-attachments/assets/f826b33a-6858-4204-9e58-8a33d802001d)


## Features

- Word length analysis
- Type-Token Ratio (TTR)
- Hapax Legomena
- Flesch-Kincaid Readability
- Batch processing of text files
- Data visualization

## Installation

```bash
git clone https://github.com/dayanadithyan/stylometry-dh.git
cd stylometry-dh
pip install -r requirements.txt
```

## Structure

```markdown
stylometry-dh/
├── .github/
│   └── workflows/
│       └── tests.yml
├── data/                   
├── examples/
│   ├── analysis_demo.ipynb
│   └── word_length_hist.png
├── src/
│   └── stylometry/
│       ├── __init__.py
│       └── analyzer.py
├── tests/
│   └── test_analyzer.py
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```
