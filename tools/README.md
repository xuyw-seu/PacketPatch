# tools/ — Utility Tools

This directory contains utility scripts for data preprocessing, analysis, and visualization related to the PacketPatch project.

## Overview

The tools in this directory support the end-to-end workflow of adversarial packet generation:

```
Raw PCAP → [Data Preprocessing] → Tokenized Data → [Training] → [Generation] → [Evaluation]
              ↑ Tools in this directory support these stages ↑
```

---

## Tool Categories

### Data Preprocessing (to be added)
Tools for converting raw PCAP files to the tokenized format expected by PacketPatch:
- PCAP file cleaning and filtering
- Byte sequence extraction
- BPE tokenization using the ET-BERT vocabulary
- Train/val/test splitting

### Data Analysis (to be added)
Tools for analyzing the generated adversarial packets:
- Perturbation vector analysis
- Packet length distribution before/after perturbation
- Feature shift analysis

### Evaluation (to be added)
Tools for evaluating adversarial effectiveness:
- Defense Success Rate (DSR) calculation
- Bandwidth Overhead (BWO) measurement
- Latency measurement

---

## Integration with ET-BERT

Since PacketPatch is built on the ET-BERT framework, many preprocessing tools from the [ET-BERT repository](https://github.com/linwhitehat/ET-BERT) can be directly used or adapted. These include:

- PCAP to text conversion
- BPE vocabulary building
- Dataset construction scripts

---

## Usage

Any tools added to this directory should include:
1. A clear docstring explaining the tool's purpose
2. Command-line argument parsing for configurable parameters
3. Example usage in comments

### Tool Naming Convention

Follow these conventions for consistency:
- `traffic_` prefix: Network traffic related tools
- `packet_` prefix: Packet-level operations
- Descriptive middle name: e.g., `feature_extractor`, `line_graph_plotter`
- `.py` suffix

---

## Dependencies

Tools may require the following Python packages:
- `scapy` — PCAP file manipulation
- `numpy` — Numerical computation
- `matplotlib` — Visualization
- `scikit-learn` — Evaluation metrics

---

> **Note**: This directory is intended to grow as more utility tools are developed. Contributions are welcome.
