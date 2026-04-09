# Data Availability

This repository does **not** include raw market data, raw sentiment scrape files, or full derived datasets.

## Why the data is not distributed

The research pipeline in this project was built using:

- broker-exported hourly FX price data
- scraped retail FX sentiment snapshot data
- derived merged research datasets built from those sources

At the time of publication, redistribution rights for these underlying data sources are uncertain. To avoid sharing material that may be subject to third-party licensing or terms of use, the repository excludes:

- raw FX price files
- raw sentiment snapshot files
- full merged datasets
- full feature-engineered outputs derived from the raw data

## What is included

The repository includes:

- source code for the dataset-building pipeline
- documentation describing the expected inputs
- methodology and design notes
- coverage and alignment logic
- feature-engineering logic
- target-construction logic for research and ML workflows

This is intended to make the workflow understandable and reproducible for users who have their own lawful access to similar data.

## Reproducing the pipeline

To reproduce the dataset locally, you will need to supply your own input data in the expected format, including:

- hourly FX CSV files
- sentiment snapshot CSV files

The repository documents the expected structure of those inputs and the transformations applied during dataset assembly.

## Synthetic / sample data

Where appropriate, this repository may include small synthetic or illustrative sample files that are manually created for demonstration purposes only. These are provided to show the expected schema and pipeline behavior and are not intended to represent real market data.

## No grant of third-party data rights

Nothing in this repository grants any right to use, copy, redistribute, sublicense, or commercially exploit third-party data sources.

Users are responsible for ensuring that any data they use with this code is obtained and used in compliance with the applicable license terms, contractual restrictions, and local laws.

## Contact

If you have questions about the repository structure or the expected input formats, please refer to the project documentation or open an issue.