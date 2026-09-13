# 3D Digital Twin

*Turning a couple of ordinary cameras into a markerless motion-capture rig — no suits, no marker balls, just video and math.*

---

## What this actually is

This repo is the working notebook for a personal experiment: **can you reconstruct a real person's 3D pose from regular 2D camera footage, well enough to drive a 3D digital twin of them?**

Professional motion capture needs marker suits, IR cameras, and calibration rigs most people will never have access to. The bet behind this project is that modern pose estimators and feature-matching networks are now good enough to skip almost all of that — you just need two (or more) synced cameras and the right pipeline stitching the 2D observations back into 3D space.

This repo *is* that pipeline, still being built and tested in the open.

## The pipeline, piece by piece

Each file here corresponds to one stage of turning flat video into a 3D skeleton:

```
   Camera A ──┐                                          
              ├─► 2D keypoints (rtmpose.py) ──┐
   Camera B ──┘                               ├─► Triangulation (triangulation_*.ipynb) ──► 3D pose ──► anny_testing1.ipynb
              2D↔2D correspondence (fe.py, xfeat_c/) ──┘
```

| File / folder | Role in the pipeline |
|---|---|
| **`rtmpose.py`** | A lightweight ONNX Runtime wrapper around **RTMPose** — runs the crop/affine-warp preprocessing, model inference, and SimCC-based heatmap decoding needed to pull 2D human keypoints out of a single image. This is where each camera's 2D skeleton comes from. |
| **`fe.py`** | A trimmed-down implementation of **LoFTR** (Local Feature Transformer) — coarse-to-fine, attention-based dense feature matching between two images. Used to find corresponding points across the two camera views, which is what makes triangulation possible without manual calibration markers. |
| **`xfeat_c/`** | A parallel, lighter-weight feature-matching path (**XFeat**), kept alongside LoFTR as a faster/cheaper alternative for the same correspondence problem — worth comparing against `fe.py` for speed vs. accuracy trade-offs. |
| **`triangulation_mine.ipynb`** | The first hand-built attempt at the triangulation math — projecting matched 2D points from both views back into a shared 3D coordinate system. |
| **`triangulation_claude.ipynb`** | A second pass at the same problem, developed with Claude's help — likely refining the camera geometry, error handling, or numerical stability of the triangulation step. |
| **`triangulation_claude_rtm.ipynb`** | The version where the triangulation logic is finally wired directly into RTMPose's keypoint output — closing the loop from "raw video" to "3D joint positions." |
| **`anny_testing1.ipynb`** | Experiments feeding the reconstructed 3D pose into **Anny**, a rigged 3D humanoid base mesh — the actual "digital twin" the reconstructed motion is meant to animate. |

## Why it's shaped like this

The multiple triangulation notebooks aren't clutter — they're a visible trail of iteration: try it manually, get help refining the math, then integrate it with the real pose-detection model instead of toy data. That progression (`_mine` → `_claude` → `_claude_rtm`) is basically a lab notebook of the project finding its footing, kept intentionally rather than squashed into one "final" file.

Similarly, having both `fe.py` (LoFTR) and `xfeat_c/` (XFeat) isn't redundancy — it's two candidate answers to the same question ("how do I match points across views?") being kept side by side until one wins out.

## Current state

This is an active research/prototyping repo, not a packaged tool. Expect:
- No single "run this and it works" entry point yet
- Overlapping experiments rather than one canonical script
- Notebooks that were used for testing ideas, not polished demos

## Where this is headed

The natural next milestones, based on what's already here:
- [ ] Settle on one matcher (LoFTR vs. XFeat) for the production pipeline
- [ ] Fold the best triangulation logic into a single, reusable module
- [ ] Extend from two cameras to a full multi-camera rig
- [ ] Drive the Anny model live, in real time, from webcam input

---

*If you're the future version of me (or someone else) picking this repo back up: start with `triangulation_claude_rtm.ipynb` — it's the most complete end-to-end path through the pipeline as it currently stands.*
