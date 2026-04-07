# Quantifying-similarity-of-zeolite-frameworks-by-a-new-structural-descriptor-based-on-ring-fractions

## Overview
This repository provides the implementation for quantifying the similarity of zeolite frameworks using a structural descriptor based on ring fractions. 

A graph-based representation is constructed from the similarity matrix, and community detection is performed using the Louvain algorithm.

The generated graph can be visualized using **Gephi**.

---

## Requirements

Install dependencies using:

```bash
conda env create -f environment.yml
```

## How to Run

Run the main script:
```bash
python Louvain-community-detection.py
```

## Output

After execution, the following files will be generated:

Graph file (for Gephi)：
rings_graph.gexf

This file can be directly imported into Gephi for visualization.

## Notes
The threshold parameter (e.g., similarity cutoff) affects graph connectivity
Adjust parameters in the script for different clustering resolutions
