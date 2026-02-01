# **`README.md`**

# AIMM-X: An Explainable Market Integrity Monitoring System with Multi-Source Attention Signals and Transparent Scoring

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2601.15304v1-b31b1b.svg)](https://arxiv.org/abs/2601.15304v1)
[![Journal](https://img.shields.io/badge/Journal-ArXiv%20Preprint-003366)](https://arxiv.org/abs/2601.15304v1)
[![Year](https://img.shields.io/badge/Year-2026-purple)](https://github.com/chirindaopensource/explainable_market_integrity_monitoring_system)
[![Discipline](https://img.shields.io/badge/Discipline-Market%20Microstructure%20%7C%20RegTech-00529B)](https://github.com/chirindaopensource/explainable_market_integrity_monitoring_system)
[![Data Sources](https://img.shields.io/badge/Data-Polygon.io%20%7C%20Reddit%20%7C%20Wikipedia-lightgrey)](https://polygon.io/)
[![Core Method](https://img.shields.io/badge/Method-Hysteresis%20Segmentation-orange)](https://github.com/chirindaopensource/explainable_market_integrity_monitoring_system)
[![Analysis](https://img.shields.io/badge/Analysis-Factor%20Decomposition-red)](https://github.com/chirindaopensource/explainable_market_integrity_monitoring_system)
[![Validation](https://img.shields.io/badge/Validation-Retrospective%20Case%20Studies-green)](https://github.com/chirindaopensource/explainable_market_integrity_monitoring_system)
[![Robustness](https://img.shields.io/badge/Robustness-Rolling%20Baseline%20Z--Scores-yellow)](https://github.com/chirindaopensource/explainable_market_integrity_monitoring_system)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![YAML](https://img.shields.io/badge/YAML-%23CB171E.svg?style=flat&logo=yaml&logoColor=white)](https://yaml.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen)](https://github.com/chirindaopensource/explainable_market_integrity_monitoring_system)

**Repository:** `https://github.com/chirindaopensource/explainable_market_integrity_monitoring_system`

**Owner:** 2026 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2026 paper entitled **"An Explainable Market Integrity Monitoring System with Multi-Source Attention Signals and Transparent Scoring"** by:

*   **Sandeep Neela** (Independent Researcher)

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, auditable, and extensible pipeline that executes the entire research workflow: from the ingestion and rigorous validation of market microstructure and attention data to the detection of suspicious trading windows via hysteresis segmentation, culminating in the generation of interpretable integrity scores and factor attributions.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `run_aimm_x_pipeline`](#key-callable-run_aimm_x_pipeline)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the analytical framework presented in Neela (2026). The core of this repository is the iPython Notebook `explainable_market_integrity_monitoring_system_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings. The pipeline addresses the critical challenge of **market integrity monitoring** by moving away from opaque, proprietary "black-box" models toward a transparent, auditable "glass-box" approach.

The paper argues that effective surveillance requires explainability—analysts must understand *why* a window was flagged—and accessibility to public data sources. This codebase operationalizes the proposed solution: **AIMM-X**, a system that:
-   **Validates** data integrity using strict OHLC consistency checks ($H_t \ge \max(O_t, C_t)$) and precise missingness semantics (NaN vs. 0).
-   **Fuses** multi-source attention signals (Reddit, StockTwits, News, Wikipedia, Google Trends) into a unified metric of public interest.
-   **Detects** anomalies using a robust **Hysteresis State Machine** (Schmitt Trigger) that prevents alert fragmentation.
-   **Scores** windows using a linear **Integrity Score ($M$)** decomposed into six interpretable factors ($\phi_1 \dots \phi_6$), enabling clear attribution of alerts to price shocks, volatility anomalies, or attention spikes.

## Theoretical Background

The implemented methods combine techniques from Financial Econometrics, Signal Processing, and Explainable AI.

**1. Multi-Source Attention Fusion ($A_{i,t}$):**
A unified attention signal is constructed by aggregating normalized proxies from diverse sources, capturing the "hype" dimension of market activity.
$$ A_{i,t} = \sum_{s \in \mathcal{S}} w_s \cdot \tilde{a}_{s,i,t} $$
where $\tilde{a}_{s,i,t}$ represents the rolling z-score of source $s$ for ticker $i$ at time $t$.

**2. Statistical Deviation Detection:**
The system employs dynamic baselines to adapt to changing market regimes, computing standardized deviations (z-scores) for returns ($r$), volatility ($\sigma$), and attention ($A$).
$$ z_{i,t}^{(x)} = \frac{x_{i,t} - \mu_{i,t}^{(x)}}{\hat{\sigma}_{i,t}^{(x)} + \epsilon} $$
A composite strength score $s_{i,t}$ aggregates these deviations to drive detection.

**3. Hysteresis-Based Segmentation:**
To avoid "chattering" (rapid on/off switching of alerts due to noise), the system uses dual-threshold hysteresis logic:
-   **Trigger:** A window opens when $s_{i,t} > \theta_{\text{high}}$.
-   **Sustain:** A window remains open while $s_{i,t} > \theta_{\text{low}}$.
-   **Exit:** A window closes only after $s_{i,t} \le \theta_{\text{low}}$ for a specified gap tolerance $g$.

**4. Interpretable Integrity Score ($M$):**
Detected windows are ranked by a score $M(w)$ that is fully decomposable into additive evidence factors:
$$ M(w) = \sum_{k=1}^{6} \omega_k \cdot \phi_k(w) $$
Factors include Return Shock Intensity ($\phi_1$), Volatility Anomaly ($\phi_2$), Attention Spike Magnitude ($\phi_3$), and Co-movement Alignment ($\phi_4$).

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/explainable_market_integrity_monitoring_system/blob/main/explainable_market_integrity_monitoring_system_ipo_main.png" alt="AIMM-X System Architecture" width="100%">
</div>

## Features

The provided iPython Notebook (`explainable_market_integrity_monitoring_system_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Task Architecture:** The pipeline is decomposed into 17 distinct, modular tasks, each with its own orchestrator function.
-   **Configuration-Driven Design:** All study parameters (thresholds, weights, lookback windows) are managed in an external `config.yaml` file.
-   **Rigorous Data Validation:** A multi-stage validation process checks schema integrity, OHLC consistency, and exchange calendar alignment.
-   **Deterministic Execution:** Enforces reproducibility through seed control, deterministic sorting, and rigorous logging of all stochastic outputs.
-   **Comprehensive Audit Logging:** Generates detailed logs of every processing step, including quarantine counts and filter statistics.
-   **Reproducible Artifacts:** Generates structured `PipelineResult` objects containing raw window lists, filtered top-N tables, and factor summary statistics.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Configuration & Validation (Task 1):** Loads and validates the study configuration, enforcing parameter constraints and determinism requirements.
2.  **Data Ingestion & Cleansing (Tasks 2-3):** Validates panel schema, enforces OHLC consistency, and strictly handles missingness semantics (NaN vs 0).
3.  **Calendar Enforcement (Task 4):** Aligns data to the canonical NYSE/Nasdaq trading session grid.
4.  **Attention Processing (Tasks 5-7):** Aligns, normalizes, and fuses multi-source attention signals into a unified metric.
5.  **Feature Engineering (Tasks 8-9):** Computes log returns and rolling realized volatility proxies.
6.  **Deviation Detection (Tasks 10-11):** Computes rolling baselines, z-scores, and the composite strength score.
7.  **Window Segmentation (Task 12):** Applies the hysteresis state machine to detect suspicious time intervals.
8.  **Scoring & Attribution (Tasks 13-14):** Computes $\phi$-factors and the composite Integrity Score $M$ with full decomposition.
9.  **Ranking & Filtering (Task 15):** Ranks windows by score and applies warmup/artifact filters.
10. **Artifact Generation (Task 16):** Produces final output tables and summary statistics.
11. **Orchestration (Task 17):** Unifies all components into a single `run_aimm_x_pipeline` function.

## Core Components (Notebook Structure)

The notebook is structured as a logical pipeline with modular orchestrator functions for each of the 17 major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callable: `run_aimm_x_pipeline`

The project is designed around a single, top-level user-facing interface function:

-   **`run_aimm_x_pipeline`:** This master orchestrator function runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project, managing data flow between validation, cleansing, detection, scoring, and reporting modules.

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `scipy`, `pyyaml`.
-   Optional dependencies: `exchange_calendars` (for precise trading session generation).

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/explainable_market_integrity_monitoring_system.git
    cd explainable_market_integrity_monitoring_system
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy scipy pyyaml exchange_calendars
    ```

## Input Data Structure

The pipeline requires a primary DataFrame `df_raw_panel` with a MultiIndex `(date, ticker)` and the following columns:

**Market Microstructure:**
1.  **`open_price`**: Float.
2.  **`high_price`**: Float, $\ge \max(Open, Close)$.
3.  **`low_price`**: Float, $\le \min(Open, Close)$.
4.  **`close_price`**: Float, $>0$.
5.  **`volume`**: Float/Int, $>0$.

**Attention Signals (Nullable):**
1.  **`reddit_posts`**: Float (count).
2.  **`stocktwits_msgs`**: Float (count).
3.  **`wiki_views`**: Float (count).
4.  **`news_articles`**: Float (count).
5.  **`google_trends`**: Float (index).

*Note: `NaN` in attention columns represents "No Coverage", while `0.0` represents "No Activity".*

## Usage

The notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell, which demonstrates how to use the top-level `run_aimm_x_pipeline` orchestrator:

```python
# Final cell of the notebook

# This block serves as the main entry point for the entire project.
if __name__ == '__main__':
    # 1. Load the master configuration from the YAML file.
    config = load_study_configuration("config.yaml")
    
    # 2. Load raw datasets (Example using synthetic generator provided in the notebook)
    # In production, load from CSV/Parquet: pd.read_parquet(...)
    df_raw_panel = generate_synthetic_panel(config)

    # 3. Execute the entire replication study.
    result = run_aimm_x_pipeline(df_raw_panel, config)
    
    # 4. Access results
    print(result.df_top_n.head())
    print(result.audit_log)
```

## Output Structure

The pipeline returns a `PipelineResult` object containing:
-   **`config_snapshot`**: The resolved configuration dictionary used for the run.
-   **`audit_log`**: A structured log of execution metadata, validation stats, and step completion.
-   **`df_windows_raw`**: The complete set of detected windows with all scores and factors.
-   **`df_windows_filtered`**: The subset of windows passing quality filters (warmup, artifacts).
-   **`df_top_n`**: The top-ranked suspicious windows formatted for reporting.
-   **`df_phi_summary`**: Summary statistics for factor contributions.
-   **`intermediate_series`**: Dictionary containing computed time-series ($r$, $\sigma$, $A$, $s$, z-scores) for debugging.

## Project Structure

```
explainable_market_integrity_monitoring_system/
│
├── explainable_market_integrity_monitoring_system_draft.ipynb   # Main implementation notebook
├── config.yaml                                                  # Master configuration file
├── requirements.txt                                             # Python package dependencies
│
├── LICENSE                                                      # MIT Project License File
└── README.md                                                    # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify study parameters such as:
-   **Universe:** `universe_tickers` list.
-   **Detection Logic:** `baseline_window_B`, `theta_high`, `theta_low`, `gap_tolerance_g`.
-   **Scoring Weights:** `alpha` weights for composite score, `omega` weights for integrity score.
-   **Filters:** `exclude_warmup`, `max_z_score_cutoff`.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **High-Frequency Data:** Adapting the pipeline for 5-minute or 1-minute bars.
-   **Real-Time API Integration:** Connecting to live feeds for Reddit/Twitter data.
-   **Advanced Normalization:** Implementing robust scalers (e.g., Median Absolute Deviation) for fat-tailed distributions.
-   **Causal Inference:** Integrating Granger causality tests to determine lead-lag relationships between attention and price.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{neela2026aimmx,
  title={AIMM-X: An Explainable Market Integrity Monitoring System Using Multi-Source Attention Signals and Transparent Scoring},
  author={Neela, Sandeep},
  journal={arXiv preprint arXiv:2601.15304v1},
  year={2026}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2026). Explainable Market Integrity Monitoring System: An Open Source Implementation.
GitHub repository: https://github.com/chirindaopensource/explainable_market_integrity_monitoring_system
```

## Acknowledgments

-   Credit to **Sandeep Neela** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **Pandas, NumPy, SciPy, and PyYAML**.

--

*This README was generated based on the structure and content of the `explainable_market_integrity_monitoring_system_draft.ipynb` notebook and follows best practices for research software documentation.*
