# CLoNeD: Closed-Loop Visual Neuromodulation under Neural Drift

**How do you optimize for a brain region that keeps changing, when all you ever see is a single number?**

📄 [Project Website](https://sogandstormesalehi.github.io/cloned.github.io/)

<img width="1120" height="686" alt="method_overview" src="https://github.com/user-attachments/assets/9357ff26-9a4d-4daf-9dc3-c28b9240272f" />

---

## Abstract

Closed-loop neural stimulus optimisation finds images that maximally drive a target brain region using only scalar neural feedback. We build on surrogate-assisted genetic search over a structured prompt space. A core assumption shared across existing approaches is that the target region remains fixed throughout the session. In practice this is routinely violated: neural responses drift within minutes due to fatigue, habituation, and fluctuating attention, and the algorithm observes only a single blended scalar reward and is never told a shift has occurred. We study how to make surrogate-assisted search robust to this drift, simulating it by smoothly interpolating the reward across multiple fMRI-characterised brain regions so the algorithm must transition between targets blindly. Our central finding is that the benefit of state-awareness depends critically on the correlation structure between successive targets. When regions share image preferences, the incoming target rises before the transition as a side-effect of optimising the outgoing one, a free ride that stateless search exploits fully while state-aware methods disrupt it. When regions prefer opposing content, state-awareness wins by actively forgetting the pre-drift landscape. We introduce an adaptive window controller that infers the drift regime from the reward signal alone and adjusts the surrogate's memory accordingly, avoiding both failure modes without requiring any prior knowledge of the underlying cause for drift.


---

## Key Findings

**The benefit of state-awareness depends critically on the correlation structure between successive targets.**

- **Anti-correlated pairs** (e.g. face-selective → place-selective): stateless search fails entirely since it stays anchored to the pre-drift landscape and never finds the new target. State-aware methods win decisively by actively forgetting what worked before.

- **Highly correlated pairs** (e.g. neighbouring visual areas with shared preferences): the new target is already rising before the transition as a side-effect of optimising the old one — a *free ride*. Classic state-aware methods disrupt this by suppressing content that was already working. Stateless wins here.

- **Adaptive state-aware** avoids both failure modes: it expands its memory window when no drift is detected (recovering stateless behaviour on the free ride), and shrinks it aggressively when the reward signal suggests a genuine shift.

**The advantage compounds across multiple transitions.** In three- and four-region sequences, state-aware methods perform similarly to stateless at the first transition but increasingly outperform it at subsequent ones. By the third and fourth phase, stateless fails completely on most seeds while both state-aware variants track all targets sequentially.

**Speed matters too.** Measured by post-transition AUC (mean new-target activation from the transition to the end of the session), state-aware methods significantly outperform stateless on anti-correlated pairs across all experiment complexities. On correlated pairs the pattern reverses, and the adaptive controller is the only method that avoids degradation in both regimes.

---

## Method

The base algorithm — surrogate-assisted genetic search over a structured text prompt space for ROI maximisation — is prior work from the [Visual Intelligence and Learning lab](https://vilab.epfl.ch/). CLoNeD extends it with three drift-aware components:

- **Recency windowing**: restricts surrogate training to recent observations, preventing stale pre-drift history from dominating
- **State-conditioned surrogate**: augments each training point with a session state descriptor (recent mean, slope, variance of the reward), letting the surrogate condition on current dynamics
- **Adaptive window controller**: monitors the reward trajectory and adjusts the memory window length: shrinks it when a quiet reward decline signals a correlated drift ending, grows it back during stable improvement

The search operates over a structured prompt space of ~10²³ candidates across 17 semantic dimensions. A CLIP text surrogate pre-screens candidates each generation, and a BERG fMRI encoding model provides oracle scores.

---

## Installation

```bash
pip install brainscore_vision brainscore_core brainscore-brainio matplotlib torch transformers \
    torchvision requests diffusers pillow resmem open-clip-torch lpips piq xgboost \
    scikit-learn accelerate openai scikit-learn scipy umap-learn tqdm wandb

pip install -U git+https://github.com/gifale95/BERG.git@79b5721d749d010fc4001da076a079eeb81a927f
```

---

## Running

```bash
python -m cloned.main
```

---

## Repository Structure

```
cloned/
├── algorithms/       # Genetic search, surrogate, drift explorer
├── rewards/          # ROI reward classes, drift reward wrappers
├── generators/       # SDXL-Turbo image generation
├── spaces/           # Structured prompt space
├── analysis/         # Figures, statistics, crossover tables
└── main.py           # Entry point
```

```
