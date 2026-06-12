# image/ — Architecture and Result Figures

This directory contains the key figures from the PacketPatch paper for reference and documentation purposes.

## Figures

| Figure | Description | Paper Reference |
| ------------------------------------------ | --------------------------------------------------- | --------------- |
| `Architecture.png` | Overall PacketPatch system architecture and workflow | Figure 2 |
| `Defense Effectiveness.png` | Defense Success Rate (DSR) across datasets | Results |
| `Comparison with Baseline Methods.png` | Performance comparison against baseline methods | Results |

## Figure Descriptions

### 1. System Architecture (`Architecture.png`)

The four-phase workflow of PacketPatch:

1. **PatchGenerator Training** — Train the BERT-based perturbation generator
2. **PacketPatch Deployment** — Deploy on both client and server proxies
3. **Adversarial Packet Construction** — Generate and inject perturbations
4. **Original Packet Restoration** — Remove perturbations and recover original data

### 2. Defense Effectiveness (`Defense Effectiveness.png`)

Defense Success Rate (DSR) of PacketPatch across three public datasets (ISCX-TOR, ISCX-VPN, USTC), demonstrating the effectiveness of adversarial packet generation against B-ETC models.

### 3. Comparison with Baseline Methods (`Comparison with Baseline Methods.png`)

PacketPatch performance compared against white-box and random baseline methods under strict black-box conditions, showing superior defense effectiveness with controllable bandwidth overhead (≤10%).

## Usage in Documentation

These figures can be referenced in the root `README.md` and other documentation:

```markdown
![Architecture](image/Architecture.png)
![Defense Effectiveness](image/Defense%20Effectiveness.png)
![Comparison with Baseline Methods](image/Comparison%20with%20Baseline%20Methods.png)
```

---

> **Note**: Figures are extracted from the published paper. Please refer to the original paper for full-resolution versions.
