# wsi-reg

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is a regression project built using **Kedro**. It follows a modular structure for data processing, modeling, and reporting.

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

To run a specific pipeline:

```bash
kedro run --pipeline=<pipeline_name>
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
* `src/wsi_reg/pipelines/`: Pipeline definitions
* `notebooks/`: Exploration and experimentation notebooks (e.g., `missing_values.ipynb`)

## How to test your Kedro project

Run the tests as follows:

```bash
pytest
```
