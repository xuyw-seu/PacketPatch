# image/ — Architecture and Result Figures

This directory contains the key figures from the PacketPatch paper for reference and documentation purposes.

## Recommended Figures to Include

| Figure | Description | Paper Reference |
|--------|-------------|-----------------|
| `Architecture.png` | Overall PacketPatch system architecture and workflow | Figure 2 |
| `PatchGenerator.png` | PatchGenerator model architecture (BERT-based) | Figure 3 |
| `Workflow.png` | Perturbation vector generation process | Figure 5 |
| `Deployment.png` | Symmetric proxy deployment architecture | Figure 6 |

## Figure Descriptions

### 1. System Architecture (Figure 2 in paper)
The four-phase workflow of PacketPatch:
1. **PatchGenerator Training** — Train the BERT-based perturbation generator
2. **PacketPatch Deployment** — Deploy on both client and server proxies
3. **Adversarial Packet Construction** — Generate and inject perturbations
4. **Original Packet Restoration** — Remove perturbations and recover original data

### 2. PatchGenerator Model (Figure 3 in paper)
The BERT architecture adapted for packet byte processing:
- **Input Layer** — Non-overlapping bi-gram byte encoding (stride=2)
- **Embedding Layer** — Token + Position + Segment embeddings (768-dim each)
- **Encoder Layer** — 12 Transformer blocks with multi-head self-attention
- **Output Layer** — SCP head (classification) + Span-MBM head (byte generation)

### 3. Perturbation Generation (Figure 5 in paper)
Four-stage generation process:
1. Input Preparation — Separate header from payload
2. Sequence Construction — Assemble with [MASK] tokens and random header
3. Perturbation Generation — Single forward pass with argmin/argmax selection
4. Vector Assembly — Concatenate Pad1 + hr + Pad2

### 4. Proxy Deployment (Figure 6 in paper)
Client-server proxy architecture based on SOCKS5 protocol with TCP_NODELAY optimization.

## Adding Figures

To include figures in this directory:

1. Extract high-resolution figures from the paper PDF
2. Save as PNG format with descriptive filenames
3. Update this README with any additional figures added

## Usage in README

These figures can be referenced in the root `README.md` and other documentation:

```markdown
![PacketPatch Architecture](image/Architecture.png)
```

---

> **Note**: Figures are extracted from the published paper. Please refer to the original paper for full-resolution versions.
