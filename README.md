# GeoLGM Trainer

GeoLGM Trainer is a compact training system for image classification with geospatial style metadata. It is config driven, tracks experiments in SQLite and JSONL, saves artifacts, profiles throughput, and provides a Streamlit dashboard.

## Purpose
This repo demonstrates production patterns for large scale training. It covers data validation, indexing, caching, resume safe training, profiling, and repeatable runs.

## Structure
The project contains configs, source code, dashboards, scripts, and tests. Runs are stored under the runs folder with a config snapshot, checkpoints, metrics logs, and artifacts.

## Setup
Install the package in editable mode. Then generate the example dataset, validate the metadata, build an index, and start training. After the run completes, launch the dashboard to review metrics and artifacts.

## Notes
The sample dataset uses CIFAR 10 images with synthetic metadata around San Francisco. Evaluation reports overall accuracy plus region and time buckets.

## Future work
Distributed training, parallel preprocessing, and container orchestration are natural extensions.
