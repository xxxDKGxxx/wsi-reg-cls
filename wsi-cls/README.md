# wsi-cls

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is a classification project focused on orthodontic data, built using **Kedro**. It includes pipelines for data processing, data science (modeling), and reporting.

## Prerequisites

* Python 3.12 (Recommended)
* [uv](https://github.com/astral-sh/uv) for fast dependency management

## How to install dependencies

The recommended way to set up the environment is using `uv` with Python 3.12:

```bash
# Create a virtual environment with Python 3.12
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

Alternatively, using standard `pip`:

```bash
pip install -r requirements.txt
```

## How to run your Kedro pipeline

To run the entire pipeline:

```bash
kedro run
```

To run a specific pipeline (e.g., `data_processing`):

```bash
kedro run --pipeline=data_processing
```

## How to visualize your Kedro pipeline

To visualize the pipeline structure and dependencies interactively:

```bash
kedro viz
```

This command launches a local server and opens your browser to display the data flow.

## Project Structure

* `conf/`: Configuration files (catalog, parameters, etc.)
* `data/`: Data storage (raw, intermediate, primary, etc.)
* `src/wsi_cls/pipelines/`: Pipeline definitions (data_processing, data_science, reporting)
* `results/`: Output from various experimental runs
* `notebooks/`: Exploration and experimentation notebooks

## How to test your Kedro project

Run the tests as follows:

```bash
pytest
```
