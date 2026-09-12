# AEVUM-v0 / Stage 1 — Experiment Report

Running log of the work to validate the core AEVUM hypothesis (tech_spec.md
section 58): *can a continuous decoder maintain a coherent speech trajectory
between irregular events, driven only by prediction innovations?*

Per tech_spec.md section 55/56, we are deliberately **not** implementing the
full architecture at once. Stage 1 isolates the most basic and unusual claim
first: a causal frontend + multi-timescale continuous-time dynamics +
causal generator, with no quantization, predictor, or event gate (every step
is dense). If this cannot reconstruct speech, nothing built on top of it
(VQ, predictor, event gate, factorization) is worth attempting yet.

## Environment

- GPU: NVIDIA GeForce RTX 3060, 12 GB VRAM, driver CUDA UMD 13.4
- Framework: PyTorch 2.6.0+cu124, torchaudio
- Package/env management: uv

## Plan for this session

Before committing to a full LibriSpeech training run, de-risk in order:

1. **Single-clip overfit sanity check** — train the Stage 1 autoencoder on one
   fixed synthetic waveform for a few hundred steps, no dataset download
   required. Goal: catch fundamental bugs (broken gradients through the
   continuous-time recurrence, dead adaptive-tau gates, generator not
   learning) cheaply, and confirm loss actually decreases.
2. **Throughput benchmark** — measure steps/sec at a realistic batch size and
   segment length, since the encoder/decoder are sequential Python loops over
   ~100 steps/sec of audio. This determines whether a full training run is
   feasible in reasonable wall-clock time on this GPU, or whether the loop
   needs optimizing first.
3. **Audio dumping** — add periodic reconstruction sample saving to the
   training script, since loss numbers alone don't tell us whether the output
   sounds like speech (tech_spec.md section 51 explicitly warns against
   trusting a single aggregate metric).
4. **Define a success criterion for Stage 1** before running on real data.
5. Only then: a real training run on LibriSpeech (starting with the smaller
   `dev-clean` split, not the 6 GB `train-clean-100`, to iterate faster).

---

## Run 1 — Single-clip overfit sanity check

`uv run python scripts/sanity_overfit.py --steps 500` (batch_size=4, seconds=2.0,
lr=3e-4, synthetic wandering-pitch multi-tone clip, no dataset).

```text
step    0  total 10.0798  wav 0.4563  mel 4.1527  stft 5.4708
step   25  total  4.5428  wav 0.4605  mel 1.6740  stft 2.4083
step   50  total  4.1812  wav 0.4641  mel 1.5232  stft 2.1939
step   75  total  3.8529  wav 0.4613  mel 1.3219  stft 2.0696
step  100  total  3.7151  wav 0.4560  mel 1.2621  stft 1.9969
step  125  total  3.4225  wav 0.4421  mel 1.1264  stft 1.8539
step  150  total  3.2573  wav 0.4345  mel 1.0886  stft 1.7342
step  175  total  3.1021  wav 0.4264  mel 1.0532  stft 1.6225
step  200  total  2.9361  wav 0.4316  mel 0.9846  stft 1.5198
step  225  total  3.2397  wav 0.4274  mel 1.1881  stft 1.6242
step  250  total  3.5448  wav 0.4281  mel 1.2629  stft 1.8538
step  275  total  3.1967  wav 0.3925  mel 1.1735  stft 1.6308
step  300  total  3.1883  wav 0.4075  mel 1.1459  stft 1.6348
step  325  total  2.9985  wav 0.3839  mel 1.1078  stft 1.5068
step  350  total  3.1217  wav 0.4113  mel 1.1583  stft 1.5520
step  375  total  2.6883  wav 0.3495  mel 0.9928  stft 1.3460
step  400  total  4.0175  wav 0.4846  mel 1.5130  stft 2.0199
step  425  total  3.2997  wav 0.4133  mel 1.2498  stft 1.6366
step  450  total  2.7634  wav 0.3500  mel 1.0376  stft 1.3757
step  475  total  2.6854  wav 0.3299  mel 1.0261  stft 1.3294
step  499  total  3.9036  wav 0.4788  mel 1.4786  stft 1.9463

--- summary ---
initial loss: 10.0798
final loss:   3.9036
reduction:    61.3%
avg step time (steady-state): 2124.0 ms  (0.47 steps/sec)
batch_size=4 seconds=2.0 frames=200
samples written to: outputs\sanity
```

Listening test (user): reconstruction is audibly present but noticeably
different from the target — consistent with the loss not having converged.

### Analysis

**Positive — the core mechanism works.** Loss drops ~70-75% on mel/STFT within
the first 200 steps, gradients clearly flow through the full chain (causal
frontend -> multi-timescale continuous-time cells -> continuous decoder ->
causal generator) with no dead branches or collapse. This was the main
question for Stage 1 and it's answered: the architecture is trainable.

**Problem 1 — training is unstable, not just slow to converge.** Loss is not
monotonic after step ~200: it oscillates and spikes sharply at steps 400
(4.02, worse than step 0-100 region) and 499 (3.90, the *final* reported
value — worse than the 2.69 seen at steps 375 and 475). For a single fixed
2-second clip this should be a nearly noise-free optimization landscape; this
much oscillation points to the learning rate being too high for this
architecture, compounded by the complete absence of gradient clipping in the
training loop. Continuous-time recurrent dynamics (adaptive `tau`, `exp(-dt/tau)`
decay) are known to be prone to sharp/exploding gradients; this is a plausible
mechanism and a concrete, fixable gap in `sanity_overfit.py` /
`scripts/train_stage1.py` rather than an architectural dead end.

Practical consequence: the script currently only saves `recon_step0.wav` and
the *final* step's reconstruction — which this run shows is not reliable,
since the final step can land on a bad spike. There is no "best checkpoint"
tracking.

**Problem 2 — throughput is too low for a real dataset run.** 2124 ms/step at
batch_size=4, 2 s clips (200 frames) = 0.47 steps/sec. At that rate the
`train_stage1.py` default of 20,000 steps would take ~12 hours of pure
forward/backward, before counting data loading. Peak VRAM usage measured
earlier (~315 MB at batch=4, 1 s) is nowhere near the 12 GB budget, which
points at the root cause: the encoder and decoder are sequential Python loops
over ~200+200 recurrent steps per training step, each launching several tiny
CUDA kernels. This is kernel-launch/Python-overhead bound, not
compute-bound — meaning **batch size can likely be increased substantially
almost for free** (more audio processed per wall-clock second, without a
proportional increase in step time), which needs to be confirmed with a
benchmark before deciding whether deeper optimization (e.g. `torch.compile`,
CUDA graphs) is required.

### Fixes applied

1. Added gradient clipping (`clip_grad_norm_`, default max norm 1.0) to
   `scripts/sanity_overfit.py`; lowered default LR from 3e-4 to 1e-4.
2. `scripts/sanity_overfit.py` now tracks and saves `recon_best.wav` (lowest
   loss seen), not just the final step, plus reports `grad_norm` per logged
   step and max pre-clip grad norm in the summary.
3. Added `scripts/benchmark_throughput.py`: step-time sweep across batch
   sizes on random data, to test the kernel-launch-overhead hypothesis
   before deciding on a batch size for real training.

---

## Run 2 — Throughput benchmark (batch-size sweep)

`uv run python scripts/benchmark_throughput.py --batch-sizes 4 16 32 64 128`
(2 s clips, random data, 10 steps/size after 3 warmup steps).

```text
batch_size   step_time_ms   audio_s/wall_s  peak_vram_mb
         4         2159.1             3.71          82.4
        16         2192.9            14.59          88.7
        32         2398.6            26.68          94.8
        64         2508.1            51.03         106.8
       128         7784.9            32.88         131.7
```

### Analysis

**Hypothesis confirmed for batch 4-64.** Step time is almost flat (2159 ms ->
2508 ms, +16%) while batch size grows 16x (4 -> 64), so throughput
(audio-seconds processed per wall-clock second) scales close to linearly:
3.71 -> 51.03 (~13.8x for a 16x batch increase). This confirms the bottleneck
in this range is kernel-launch/Python overhead from the sequential per-frame
loop, not GPU compute — batch size is nearly free here, and VRAM stays
trivial (<110 MB of 12 GB).

**batch=128 breaks the trend.** Step time jumps 3.1x (2508 -> 7785 ms) for a
2x batch increase, and throughput actually *drops* versus batch=64 (32.88 vs
51.03). VRAM is still tiny (131.7 MB), so this isn't an OOM/memory-pressure
effect — more likely a kernel-algorithm-selection or allocator artifact at
that shape. Not investigating now since batch=64 is already a good
operating point; worth revisiting if we later need bigger batches (e.g. for
longer segments in Stage 6).

**Practical takeaway:** batch=64 gives ~51 audio-seconds processed per
wall-clock second (forward+backward only, no data loading). At that rate,
LibriSpeech `dev-clean` (~5.4 h of audio) is ~19,440 s / 51 ≈ 6.4 minutes of
pure compute per epoch, and `train-clean-100` (~100 h) is ≈ 2 hours per
epoch. This is a much better outlook than the initial 0.47 steps/sec at
batch=4 suggested — the model just needed a bigger batch, not an
architectural rewrite.

### Next step

Re-run the single-clip overfit sanity check with the grad-clipping fix to
confirm the loss curve is now monotonic (batch size doesn't matter for this
check — a repeated identical clip across the batch dimension carries no
extra information, it only affects wall time, and batch=4 is already cheap).
Only after that looks clean does it make sense to move to a real
LibriSpeech run.

---

## Run 3 — Single-clip overfit, with grad clipping (max norm 1.0) + LR 1e-4

`uv run python scripts/sanity_overfit.py --steps 500` (same synthetic clip,
batch_size=4, seconds=2.0).

```text
step    0  total 10.0798  wav 0.4563  mel 4.1527  stft 5.4708  grad_norm  5.922
step   25  total  4.9910  wav 0.4564  mel 1.9958  stft 2.5388  grad_norm 18.738
step   50  total  4.3864  wav 0.4595  mel 1.6211  stft 2.3058  grad_norm 19.280
step   75  total  4.1869  wav 0.4638  mel 1.4909  stft 2.2321  grad_norm  8.760
step  100  total  4.1336  wav 0.4792  mel 1.4522  stft 2.2022  grad_norm 16.183
step  125  total  3.9581  wav 0.4651  mel 1.3790  stft 2.1140  grad_norm 10.061
step  150  total  3.8765  wav 0.4661  mel 1.3217  stft 2.0888  grad_norm 24.811
step  175  total  3.8376  wav 0.4716  mel 1.3004  stft 2.0656  grad_norm 13.466
step  200  total  3.6500  wav 0.4642  mel 1.1932  stft 1.9927  grad_norm 32.978
step  225  total  3.5958  wav 0.4674  mel 1.1785  stft 1.9498  grad_norm 16.575
step  250  total  3.5444  wav 0.4642  mel 1.1472  stft 1.9330  grad_norm 19.337
step  275  total  3.5096  wav 0.4622  mel 1.1465  stft 1.9008  grad_norm 23.732
step  300  total  3.4260  wav 0.4620  mel 1.1203  stft 1.8438  grad_norm 18.486
step  325  total  3.3517  wav 0.4572  mel 1.0840  stft 1.8104  grad_norm 23.471
step  350  total  3.3796  wav 0.4575  mel 1.1283  stft 1.7939  grad_norm 19.366
step  375  total  3.2407  wav 0.4502  mel 1.0509  stft 1.7396  grad_norm 20.489
step  400  total  3.2481  wav 0.4469  mel 1.0909  stft 1.7103  grad_norm 22.036
step  425  total  3.2273  wav 0.4464  mel 1.0875  stft 1.6934  grad_norm 22.238
step  450  total  3.1046  wav 0.4405  mel 1.0157  stft 1.6484  grad_norm 24.591
step  475  total  3.0575  wav 0.4346  mel 1.0023  stft 1.6206  grad_norm 28.955
step  499  total  2.9737  wav 0.4352  mel 0.9540  stft 1.5845  grad_norm 23.946

--- summary ---
initial loss: 10.0798
final loss:   2.9737
best loss:    2.9715  (step 486)
reduction (final): 70.5%
reduction (best):  70.5%
max grad norm (pre-clip): 104.291
avg step time (steady-state): 2150.1 ms  (0.47 steps/sec)
batch_size=4 seconds=2.0 frames=200
```

### Analysis

**Instability fixed.** The loss curve is now monotonic (within normal
optimization noise) for all 500 steps — no spikes, `final` (2.9737) and
`best` (2.9715, step 486) are essentially the same value. This confirms the
run 1 oscillation was caused by missing gradient clipping / too-high LR, not
an architectural problem. Reduction is also better than run 1: 70.5% vs
61.3%.

**Gradients are consistently large.** Pre-clip grad norm ranges ~6-33 across
logged steps with a max of 104.3, against a clip threshold of 1.0 — meaning
clipping is engaging on essentially every step. Training is stable *because*
of the clip, not because raw gradients are naturally well-behaved. Not a
blocker (this is exactly what clipping is for), but worth watching during
real training: if convergence is too slow, revisiting the clip threshold or
adding LR warmup would be the first things to try, in that order.

**The wav-loss plateau from run 1 persists and is now clearer.** With a
smooth curve to look at: `mel` drops 77% (4.15 -> 0.95) and `stft` drops 71%
(5.47 -> 1.58), while `wav` barely moves (0.456 -> 0.435, ~4.6%) and is
non-monotonic throughout (bounces between ~0.43-0.48 the whole run). The
model is matching spectral magnitude much better than it's matching the
waveform sample-for-sample — plausibly a phase/fine-detail limitation of the
nearest-neighbor-upsample causal generator. This is a known weak spot of
plain L1-on-waveform objectives in neural vocoders generally, so it's not
alarming on its own, but it means **wav loss should not be used as the
primary signal for judging Stage 1 quality** — mel/STFT trends and actual
listening matter more.

### Stage 1 success criterion (defined now, before the first real-data run)

Given Stage 1's purpose (tech_spec.md section 41, Stage 1: "obtain a stable
causal speech autoencoder", not yet about bitrate), the bar for moving on to
Stage 2 (adding VQ) is deliberately lightweight:

- Reconstruction loss (mel/STFT-driven) decreases stably (no run-1-style
  oscillation) over a training run on real LibriSpeech utterances.
- Listening to a handful of reconstructed validation clips: the speech is
  intelligible and recognizably the same utterance/speaker as the target —
  it does not need to be high-fidelity or artifact-free yet.

No formal metrics (WER/PESQ/speaker-embedding, tech_spec.md section 50) are
required at this sub-stage — those are proportionate once we're validating
the full v0 pipeline (event gate active), not this architecture-only check.

### Next step

Sanity check and throughput are both green. Move to a real training run on
LibriSpeech `dev-clean` (smaller/faster than `train-clean-100`) using
`scripts/train_stage1.py`, after porting the same fixes (grad clipping,
lower default LR) into it and adding periodic reconstruction-sample dumping
so the run can actually be listened to, not just judged by loss numbers.

---

## Run 4 — LibriSpeech dev-clean, real-data training (interrupted at step 500)

`uv run python scripts/train_stage1.py --steps 5000` (defaults: batch_size=64,
lr=1e-4, grad_clip_norm=1.0, `dev-clean`).

Train loss (noisy batch-to-batch, different 64 real utterances every step):
15.20 (step 0) -> dips and a 100-300 step plateau around 5.6-5.9 -> resumes
descending from step 350 -> 4.61 (step 500). `wav` component stayed flat/
noisy the whole run (0.037-0.064, no net improvement). Held-out fixed-clip
val loss: 15.5154 (step 0) -> **5.3295 (step 500, new best)**, a 65.7% drop.

**Listening result (user): `val_step500.wav` is noise, not recognizable
speech** — `val_step0.wav` was silence. Loss dropping while the reconstruction
stays unintelligible is the central open problem right now; training was
paused (~step 800) to investigate with a cleaner, cheaper test before
spending more wall-clock time on the full dataset run.

---

## Run 5 — Real-clip overfit test (single real LibriSpeech utterance)

Motivation: isolate whether the noise problem is a *generalization* issue
(hard to learn across thousands of diverse real utterances in ~500-800
steps) or a more fundamental *architecture/loss* issue that would show up
even in the easiest possible setting — memorizing one fixed real clip, no
generalization required at all. Added `--source real` to
`scripts/sanity_overfit.py` (loads the same fixed clip, `dataset[0]` from
`dev-clean`, that `train_stage1.py` uses for its held-out sample).

`uv run python scripts/sanity_overfit.py --source real --steps 500`:

```text
initial loss: 17.3089
final loss:    3.6588
best loss:     3.6419  (step 498)
reduction (best): 79.0%
max grad norm (pre-clip): 885.451
avg step time (steady-state): 2340.7 ms  (0.43 steps/sec)
```

**Result: loss reduction (79.0%) is the best of any run so far — better than
the synthetic overfit (70.5%) and the real-data training run's val loss
(65.7%) — but the user reports the audio is still noise, not recognizable
speech** (envelope/amplitude pattern looked similar, content did not).

### Analysis

This is the most decisive result yet, because it rules out the
"not-enough-steps-to-generalize" explanation entirely: with zero
generalization required (one fixed clip, repeated every step), the model
still cannot produce intelligible speech, despite achieving its lowest loss
of any experiment. The problem is not (or not only) about training budget —
it reproduces at the easiest possible task.

**Key new data point: `max grad norm (pre-clip)` jumped from 104.291 (Run 3,
synthetic tone) to 885.451 (Run 5, real speech) — 8.5x higher, same
architecture, same hyperparameters, same clip length, only the target signal
changed.** This reopens the gradient-norm investigation from Run 3 with a
sharper question: real speech's sharp transients (vs. the smooth synthetic
tone) are clearly implicated — the diagnostic instrumentation proposed after
Run 3 (per-block grad norms, tau/alpha logging) is now the direct next step,
run on this exact real clip, to find out which block is actually responsible
before touching any code.

---

## Run 6 — Per-block gradient / tau-alpha diagnostics on the real clip

Added `ContinuousTimeCell.start_recording()/stop_recording()` (accumulates
per-step tau/alpha values) and `scripts/diagnose_gradients.py` (splits
gradient norm into 17 groups: encoder/decoder x {fast,mid,slow} x
{time_constant, candidate}, plus cross-connections, frontend, head, and
generator). Verified without training that the grouping covers all
3,888,129 parameters with nothing falling into an `other` bucket.

`uv run python scripts/diagnose_gradients.py --source real --steps 200`
(same real clip as Run 5):

```text
step    0  total 17.3089   generator=26.645   (all *.tau groups: 0.001-0.044, smallest of all groups)
step   25  total  9.0537   generator=636.596  (all *.tau groups: 0.004-0.248, still smallest)
step   50  total  7.3938   generator=316.713  (all *.tau groups: 0.003-0.182, still smallest)
```

Tau/alpha values themselves stayed in sane, stable ranges throughout (e.g.
`encoder.fast` tau ~0.044-0.046s, alpha ~0.80; `decoder.slow` tau ~1.3-3.8s,
alpha ~0.99) — no branch's time constant collapsed to its boundary or
exploded.

### Analysis

**The tau hypothesis from Run 3/5 is refuted by direct measurement.** Every
`*.tau` group (the `time_constant` linear layers, across all 6 branches) has
the *smallest* gradient norm of every group at every logged step — the
opposite of what the `d(alpha)/d(tau) ~ 1/tau^2` analysis predicted would
dominate. The analytical concern about the fast branch's boundary
sensitivity was reasonable but empirically wrong as an explanation here.

**`generator` gradient norm dominates completely and matches the scale of
the problem:** 26.6 -> 636.6 -> 316.7 across the first 50 steps, one to two
orders of magnitude above every other group (next-largest around 15-17). At
step 25 alone it nearly reproduces the 885 aggregate max norm seen in Run 5.

**New hypothesis: the log-compression epsilon in `ReconstructionLoss` is too
small.** `docs/../src/aevum/training/losses/reconstruction.py` computes
`log(mag + eps)` with `eps=1e-5` (mel) / `1e-7` (multi-res STFT) on the
*predicted* magnitude spectrogram. At initialization the generator's output
is near-zero everywhere, so predicted magnitude is near-zero across most
time-frequency bins; the L1 gradient through `log(mag_hat + eps)` behaves
like `1/(mag_hat + eps)`, which explodes as `mag_hat -> 0`. Real speech's
broadband/transient content (fricatives, plosives, noise bursts) carries
real target energy across far more frequency bins than the synthetic tone's
3 narrow harmonics — plausibly explaining both *why* it's the generator
specifically (it's the only block whose output feeds directly into the
mel/STFT magnitude computation) and *why* real speech produces an 8.5x
larger blow-up than the synthetic tone (more bins with a large near-zero-vs.
-real-energy mismatch).

This also offers a candidate explanation for the noisy output itself: with
clipping engaged almost every step, the update *direction* is dominated by
this generator-eps artifact rather than by the comparatively tiny,
presumably more meaningful gradient signal from other blocks — the model may
be spending most of its optimization budget reacting to a loss-function
artifact rather than learning correct speech structure.

### Next step (proposed, not yet run)

Increase `eps` in `ReconstructionLoss` (e.g. `1e-5`/`1e-7` -> `1e-2`, or an
equivalent softer compression) and re-run `diagnose_gradients.py --source
real` to check whether `generator`'s gradient norm drops to a sane range. If
it does, re-run the real-clip overfit test (Run 5) to check whether the
reconstruction actually becomes intelligible speech this time, before
returning to the full LibriSpeech training run.

---

## Run 7 — Re-diagnosing after raising eps to 1e-2

Changed `ReconstructionLoss`/`MultiResolutionSTFTLoss` eps from `1e-5`/`1e-7`
to a shared `1e-2`. Re-ran `uv run python scripts/diagnose_gradients.py
--source real --steps 200` on the same real clip as Run 6:

```text
                 generator grad norm      total loss
before (eps=1e-5/1e-7):  step 0    26.6         17.31
                         step 25  636.6          9.05
                         step 50  316.7          7.39

after  (eps=1e-2):       step 0    18.8          5.59
                         step 25    6.6          5.39
                         step 50    1.9          5.36
```

(Lower absolute loss with the larger eps is expected/not comparable across
settings — log-compression with a larger eps compresses the log-scale more,
it doesn't mean "better reconstruction" by itself.)

### Analysis

**Hypothesis confirmed, strongly.** The generator's gradient norm didn't
just shrink, its *trajectory reversed*: before, it exploded upward (26.6 ->
636.6 -> 316.7); after, it decays monotonically (18.8 -> 6.6 -> 1.9) toward a
sane single-digit range within 50 steps. All other groups stayed roughly the
same small scale as Run 6 (tau groups still smallest of all). With clipping
at norm 1.0, the update direction is now barely perturbed by this artifact
instead of being dominated by it ~19x versus ~885x at the peak.

### Next step

Re-run the real-clip overfit test (`scripts/sanity_overfit.py --source real
--steps 500`) with the new eps and listen to the result: does the
reconstruction become intelligible speech now, or does it still sound like
noise despite the healthier gradient behavior? This is the test that
actually matters — a well-behaved gradient norm is necessary but not
sufficient proof that the underlying noise problem is fixed.

---

## Run 8 — Real-clip overfit, eps=1e-2 (500 steps)

`uv run python scripts/sanity_overfit.py --source real --steps 500`, same
real clip as Runs 5-7, with the fixed eps.

```text
step    0  total 5.5930  wav 0.0513  mel 2.7737  stft 2.7680  grad_norm  18.843
step   75  total 5.2155  wav 0.0824  mel 2.0624  stft 3.0707  grad_norm 153.114  <- one-off spike
step  100  total 3.7905  wav 0.0482  mel 1.7667  stft 1.9757  grad_norm   6.917
step  200  total 2.7137  wav 0.0469  mel 1.1045  stft 1.5623  grad_norm  10.611
step  300  total 2.4750  wav 0.0482  mel 0.9707  stft 1.4561  grad_norm  20.836
step  400  total 2.2282  wav 0.0489  mel 0.8294  stft 1.3499  grad_norm  31.685
step  499  total 2.0672  wav 0.0482  mel 0.7407  stft 1.2784  grad_norm  25.281

best loss: 2.0229 (step 484), reduction (best) 63.8%
max grad norm (pre-clip): 162.919
```

Grad norm oscillates (one spike to 153 around step 75, otherwise mostly
single-to-low-double digits) but never returns to the old exploding regime
(compare to Run 5's 885 max) — consistent with Run 7's diagnostic. `wav`
stayed essentially flat the entire run (0.0513 -> 0.0482, no real
improvement) while `mel`/`stft` improved steadily and substantially.

**Listening result (user): still the same noise as before the eps fix.**

### Analysis

The eps fix is confirmed necessary but not sufficient: gradients are now
healthy, loss reduction is the cleanest/most monotonic of any real-data run
(63.8%), and the problem persists unchanged. This rules out gradient
explosion as *the* cause of the noise — it was a real, worth-keeping fix,
but not the answer to the actual question we care about.

The flat `wav` loss throughout (here and in every previous run) is the
strongest clue: `mel` and multi-res `stft` are magnitude-only losses —
literally blind to phase — and `wav` L1 isn't providing enough signal to
constrain phase either. This matches the classical "phase reconstruction
problem": matching a magnitude spectrogram exactly does not by itself
determine a clean waveform, and a model can satisfy magnitude losses while
producing something with correct short-term spectral energy but incoherent
phase across frames — which sounds like noise.

### Next step — cleanest possible isolating experiment

Rather than continue reasoning about tau/gradients/cross-connections/latent
dimensions, the proposal (from the user) is to ask one binary question
directly: **can `generator` alone synthesize this real clip at all**, given
the easiest possible input? Implemented in
`scripts/overfit_generator_only.py`: throw out frontend, encoder, continuous
dynamics, and decoder entirely; replace them with a fully free, directly
learnable latent tensor `Z in R^{1 x 384 x 200}` (one independent vector per
100 Hz frame, no bottleneck of any kind — "cheat code": 76,800 free
parameters to describe one 2-second clip). Train `Z` jointly with
`generator`'s own parameters, same reconstruction loss.

```text
learnable Z [1, 384, 200] -> generator -> wav -> ReconstructionLoss
```

- **Result A (still noise after a few thousand steps):** encoder and
  continuous dynamics are fully exonerated — the problem is either
  `generator`'s architecture or the reconstruction loss itself. Distinguish
  those two with `scripts/check_phase_problem.py` (already written,
  not yet run): reconstruct the *target*'s own true magnitude spectrogram
  via Griffin-Lim (discarding true phase, no model involved at all). If that
  also sounds noisy/metallic, the loss formulation (magnitude-only,
  phase-blind) is the shared root cause of both failures. If it sounds
  clean, `generator`'s architecture specifically is the deficient part.
- **Result B (clean/near-original speech):** `generator` is fine; the
  bottleneck is upstream — something about how the
  encoder/dynamics/decoder chain compresses information into the latent it
  hands to the generator.

This also runs much faster per step than every previous experiment: it
skips both 200-step sequential Python recurrence loops (encoder + decoder),
leaving only a fixed-depth conv stack.

Command: `uv run python scripts/overfit_generator_only.py --steps 3000`.

---

## Run 9 — Generator-only isolation test: Result B

`uv run python scripts/overfit_generator_only.py --steps 3000` (same real
clip, free latent Z in R^{1x384x200}, no encoder/dynamics/decoder at all).

```text
step    0  total 5.9731  wav 0.0680  mel 2.7869  stft 3.1183
step  500  total 0.6126  wav 0.0530  mel 0.1067  stft 0.4529
step 1000  total 0.3609  wav 0.0419  mel 0.0481  stft 0.2710
step 2000  total 0.3185  wav 0.0303  mel 0.0677  stft 0.2206
step 2999  total 0.2159  wav 0.0245  mel 0.0368  stft 0.1546
best loss: 0.1946
```

**Result B, decisively.** Best loss 0.1946 (96.7% reduction) is an order of
magnitude lower than the best any full-pipeline run achieved on this same
clip/loss (2.0229, 63.8%, Run 8). Critically, `wav` L1 actually decreases
substantially here (0.068 -> 0.0245, -64%) for the first time in this whole
investigation — every previous run had it stuck flat around 0.04-0.05
regardless of how much mel/stft improved. **User's listening verdict: "Один
в один"** (identical to target).

### Analysis

`generator` is exonerated. Given an unconstrained 100 Hz latent, it
reproduces this real speech clip essentially perfectly under the exact same
reconstruction loss that produced noise through the full pipeline. This
also weakens the "phase-blind magnitude loss" hypothesis from Run 8's
analysis as a *sufficient* explanation on its own — if the loss were
fundamentally incapable of specifying a clean waveform, this experiment
should have produced noise too, same as every other run. It didn't. So the
loss function can drive a clean reconstruction when the upstream
representation is good enough; the question now is squarely about what the
frontend/encoder/decoder chain does to the information before it reaches
the generator.

### Next step (proposed by the user): add the decoder back in

Second isolation step, `scripts/overfit_decoder_generator.py` (not yet run):
keep frontend and encoder removed, but reintroduce `ContinuousDecoder`
(its own norm+projection "head" is already part of its forward pass) between
the free latent and the generator:

```text
learnable Z [1, T, 384] -> ContinuousDecoder -> y -> generator -> wav
```

- **Result A (clean speech again):** decoder + generator both exonerated;
  the bottleneck narrows to frontend -> encoder dynamics -> encoder head.
- **Result B (noise again):** since generator just proved capable in
  isolation, this would point squarely at `ContinuousDecoder` — specifically
  whether its leaky-integrator update (`d_t = alpha*d_{t-1} +
  (1-alpha)*u_t`) over-smooths information between steps.

Command: `uv run python scripts/overfit_decoder_generator.py --steps 4000`.

---

## Run 10 — Decoder+generator isolation test: Result A (stopped early)

`uv run python scripts/overfit_decoder_generator.py --steps 4000` (same real
clip, free per-frame latent Z in R^{1x200x384} feeding `ContinuousDecoder`
directly, then `generator`). Stopped early at step ~400 — result already
clear, no need to run to completion:

```text
step    0  total 6.3062  wav 0.0835  mel 2.7974  stft 3.4253
step  100  total 1.8995  wav 0.0518  mel 0.7097  stft 1.1379
step  200  total 0.9789  wav 0.0540  mel 0.2394  stft 0.6855
step  300  total 0.6834  wav 0.0503  mel 0.1317  stft 0.5014
step  400  total 0.5915  wav 0.0469  mel 0.1132  stft 0.4314
```

**User's listening verdict at step 400: "практически снова точь в точь"**
(again essentially identical to target).

### Analysis

**Result A.** Both `ContinuousDecoder` and `generator` are exonerated —
`ContinuousDecoder`'s leaky-integrator dynamics (`d_t = alpha*d_{t-1} +
(1-alpha)*u_t`) are not over-smoothing information when fed a good enough
per-frame input; it can pass through what it's given well enough for clean
reconstruction. The bottleneck is now narrowed specifically to
**frontend -> encoder continuous dynamics -> encoder head** (the path that
turns real audio into `z_t` in the first place) — the only part of the
full pipeline not yet tested in isolation.

### Next step — encoder+generator isolation, and a standing hypothesis about the interface

`scripts/overfit_encoder_generator.py` (not yet run): removes the free-latent
"cheat" and the decoder both. Real audio goes through `frontend -> encoder
dynamics -> encoder head -> generator`, no decoder at all:

```text
real audio -> frontend -> encoder dynamics -> encoder head -> generator -> wav
```

The emerging picture, if this is also clean:

```text
free latent -> generator                 (Run 9)   clean
free latent -> decoder -> generator       (Run 10)  clean
audio -> encoder -> generator             (this)    clean?
audio -> encoder -> decoder -> generator  (full)     noise
```

That pattern would mean the problem is not any single block's capacity but
the **encoder-decoder interface**. User's specific hypothesis: `decoder`'s
leaky-integrator update (`d_t = alpha_t*d_{t-1} + (1-alpha_t)*f(z_t,
d_{t-1})`) forces each `z_t` to "leak" into the state over several steps.
A *free* latent Z can pre-compensate for this during its own training (it's
optimized jointly with the decoder from scratch to already account for that
smoothing). A real encoder instead has to simultaneously (a) analyze the
waveform, (b) build a meaningful instantaneous state, (c) anticipate how the
decoder will subsequently distort/smear that representation over time, and
(d) push all of that through a recurrent bottleneck — a much harder joint
optimization problem than either side solved separately.

**Candidate fix (explicitly not implemented yet — encoder+generator test
comes first):** a direct residual/skip highway from the encoder latent to
the generator input, alongside the decoder path:

```text
y_t = W_d[d_t^F; d_t^M; d_t^S] + W_skip * z_t
```

i.e. `encoder z_t` feeds `generator` both through `decoder` (for temporal
context / autonomous evolution between events) and directly (so decoder is
not the *only* channel for current information). This is the user's leading
hypothesis for the eventual fix, contingent on the encoder+generator test
result below.

Command: `uv run python scripts/overfit_encoder_generator.py --steps 3000`
(watch closely in the first 300-500 steps — no need to wait for completion
if the result is already clear, same as Run 10).

---

## Run 11 — Encoder+generator isolation test: diverges (new failure mode)

`uv run python scripts/overfit_encoder_generator.py --steps 3000` (real
audio -> frontend -> encoder fast/mid/slow -> encoder head -> generator, no
decoder, no free latent):

```text
step    0  total  5.4316  wav 0.0457  mel 2.7508  stft 2.6351  grad_norm    8.925
step  100  total  5.2396  wav 0.0453  mel 2.6469  stft 2.5473  grad_norm   16.403
step  125  total  5.0412  wav 0.0459  mel 2.5922  stft 2.4031  grad_norm   12.326
step  150  total  4.4249  wav 0.0586  mel 2.0304  stft 2.3359  grad_norm  378.710
step  175  total 11.4930  wav 0.3314  mel 2.3777  stft 8.7839  grad_norm 1649.750
step  200  total  4.9378  wav 0.0739  mel 2.1947  stft 2.6692  grad_norm  641.721
step  225  total 22.5702  wav 0.8290  mel 2.7780  stft 18.9632 grad_norm  981.581
```
(stopped early by the user — clearly diverging, no need to run further)

### Analysis

**A new, different failure mode.** Steps 0-125 look fine (loss trending
down slowly, grad_norm in the teens). Starting at step 150, grad_norm
explodes (378 -> 1650 -> 642 -> 982) and loss gets *worse* than at step 0 by
step 225 (22.57 vs 5.43). This is the first time this kind of instability
has appeared with `generator` in the loop *and* the fixed eps in place —
Runs 9 and 10 (free latent feeding generator / decoder+generator) never
showed anything like it, and they used the exact same generator,
reconstruction loss, LR, and clipping. The only thing different here is
that `generator` is now downstream of the encoder's real fast/mid/slow
continuous-time recurrence being trained on real audio, instead of a free,
directly-optimizable per-frame tensor.

This reopens (in a more specific form) the gradient-explosion question from
Runs 5-7 — this time implicating the encoder's *recurrent* dynamics
specifically (not tau in isolation, which Run 6 already cleared; possibly
the interaction between the recurrence and the generator once the encoder
is actually forced to represent real transient-heavy audio, rather than
tau, candidate, or generator weights on their own).

### Next step — strip the recurrence specifically

`scripts/overfit_frontend_generator.py` (not yet run): keep the real causal
`frontend` and `generator`, but replace the fast/mid/slow continuous-time
dynamics with a trivial per-frame linear projection (`Conv1d` with
`kernel_size=1` — no cross-timestep dependency beyond what the causal
frontend already has):

```text
real audio -> frontend -> Linear(384, 384) per frame -> generator -> wav
```

- **Result A (fast, clean, no explosion):** frontend and generator are both
  fine on real audio; the problem is specifically the encoder's
  continuous-time recurrent dynamics (`ContinuousTimeCell`) — next step
  would be taking that cell apart directly rather than testing more
  end-to-end combinations.
- **Result B (explodes/noisy again):** the problem is upstream of any
  recurrence — the frontend itself, its lack of normalization, stride/
  alignment, or the frontend->generator interface.

Command: `uv run python scripts/overfit_frontend_generator.py --steps 2000`.

---

## Run 12 — Frontend+generator isolation test: Result A (recurrence implicated)

`uv run python scripts/overfit_frontend_generator.py --steps 2000` (real
audio -> frontend -> per-frame `Linear(384,384)` -> generator, no
recurrence at all). Stopped early at step 875 — result already clear:

```text
step    0  total 5.4029  wav 0.0441  mel 2.7378  stft 2.6210  grad_norm  7.009
step  200  total 1.6089  wav 0.0511  mel 0.5867  stft 0.9711  grad_norm  9.563
step  500  total 0.6803  wav 0.0413  mel 0.1582  stft 0.4807  grad_norm  6.954
step  875  total 0.3997  wav 0.0274  mel 0.0768  stft 0.2955  grad_norm  4.889  best 0.3968
```

`grad_norm` stayed in single-to-low-double digits (5-28) the entire run —
no explosion, unlike Run 11 (which spiked 378->1650->642->982 by step 225
using the exact same frontend and generator, only with the fast/mid/slow
recurrence added back in).

### Analysis — full isolation matrix

```text
free latent -> generator                    clean  (Run 9)
free latent -> decoder -> generator          clean  (Run 10)
audio -> frontend -> Linear -> generator      clean  (this, Run 12)
audio -> frontend -> encoder(f/m/s) -> gen     diverges  (Run 11)
audio -> encoder -> decoder -> generator        noise  (Run 4/5/8, full pipeline)
```

Frontend, generator, and `ContinuousDecoder` are all individually cleared.
**The encoder's continuous-time recurrence (`ContinuousTimeCell` /
fast-mid-slow dynamics), specifically when trained on real audio rather
than a free tensor, is the common factor in every failing configuration.**

### Next step — which branch, and a proposed architectural fix

User's next test, grounded directly in the tau/alpha values measured in
Run 6/7 (alpha_fast~0.80, alpha_mid~0.964, alpha_slow~0.996 — i.e.
1-alpha = 20% / 3.6% / 0.4% new information admitted per 10 ms step): test
each branch **alone**, no cross-timescale connections, to see whether
severity of the problem tracks branch "speed":

```text
real audio -> frontend -> ONE ContinuousTimeCell (fast|mid|slow) -> per-frame projection -> generator -> wav
```

Prediction: `fast` should still permit reasonable reconstruction; `mid`
should be noticeably worse; `slow` should turn speech to mush — because a
state with alpha~0.996 can only admit ~0.4% new information per step, which
is a severe low-pass if it's also the *only* channel current acoustic
information must travel through to reach the generator.

Implemented in `scripts/overfit_single_branch.py --branch {fast,mid,slow}`
(500-1000 steps each is expected to be enough).

**Proposed architectural diagnosis (not yet implemented):** the continuous
states are currently forced to serve two different jobs at once — *what to
remember* (context/memory) and *how to transport all current acoustic
information downstream* (transport channel) — and those are different
jobs. A heavily-smoothed slow state may be exactly right for memory, but
wrong as the sole transport path for fast-changing detail. Proposed fix,
mirroring the decoder-side residual-highway idea from Run 10's analysis but
on the encoder side:

```text
z_t = W_instant * f_t + W_F*h_t^F + W_M*h_t^M + W_S*h_t^S
```

i.e. add a direct instantaneous path from the frontend feature `f_t`
straight into the fused latent, alongside (not instead of) the fast/mid/
slow states — so `f_t` carries consonants/transients/formant-transitions
that must not be smoothed, while the continuous dynamics contribute context,
speaker trajectory, rhythm, and predictive memory on top. This also
reframes continuous dynamics' role for the eventual event-driven codec: not
the transport channel for signal, but temporal memory layered over an
instantaneous representation — innovation (`e_t = z_t - z_hat_t`) should be
measurable before the encoder has a chance to smooth it away.

Planned test sequence: (1) the three single-branch ablations above to
confirm the low-pass-severity hypothesis experimentally, then (2) full
multi-timescale (with cross-connections) *plus* the direct residual
highway, to check whether that combination restores clean reconstruction on
the complete system.

---

## Run 13 — Single-branch ablations: fast stable, mid/slow explode

`uv run python scripts/overfit_single_branch.py --branch {fast,mid,slow}
--steps 1000` (real audio -> frontend -> ONE `ContinuousTimeCell` -> linear
projection -> generator, no cross-timescale connections).

```text
fast (tau 0.01-0.08s):  grad_norm stays 0.7-37 the whole run, loss 5.42 -> best 0.3815 (93.0% reduction), no explosions
mid  (tau 0.05-0.5s):   grad_norm spikes to 552, 1183, 1627, 2392, 2728, 3140, 5088, 3889 at various steps;
                        loss repeatedly jumps to 20-26 then partially recovers; best loss stalls at 4.5256 after step 275
slow (tau 0.3-5.0s):    grad_norm spikes to 1895, 2955, 10534(!), 7414, 2635, 1188; even more frequent/severe than mid;
                        loss repeatedly jumps to 25-26; best loss stalls at 3.1493 after step 300
```

**Confirms the severity gradient predicted from measured alpha values:**
fast (alpha~0.80, 20%/step new info) trains cleanly; mid (alpha~0.964,
3.6%/step) explodes repeatedly and its best loss stalls early; slow
(alpha~0.996, 0.4%/step) explodes even more severely (peak grad_norm 10,534
— the largest seen anywhere in this investigation) and its best loss stalls
even earlier. Notably, `slow` did briefly find a better loss (3.1493) before
getting knocked off course — capacity isn't dead, but training can't hold a
stable trajectory.

### Analysis — refined mechanism (not just "more smoothing")

The user's key insight: the failure mode is not merely aggressive low-pass
filtering (which would predict *slow learning*, not *explosions*) — it's
specifically that the previous hidden state enters the update **twice**:
once directly through the memory term (`alpha_t * h_{t-1}`), and again
through the candidate's own recurrent weights (`u_t =
phi(W_x x_t + W_hh h_{t-1})`). For a branch with `alpha` close to 1 (slow),
the old state barely decays *and* gets re-injected through a learned
nonlinear transform every step — a plausible positive-feedback mechanism
whose effect compounds across a 200-step backprop-through-time unroll. This
also explains why `fast` is immune: its strong per-step contraction
(`1-alpha~0.20`) dominates before any instability in `W_hh` can accumulate
across time.

### Next step — remove self-recurrence from the candidate only

Minimal, additive change (not a redesign): added `candidate_uses_hidden:
bool = True` to `ContinuousTimeCell.__init__` (`src/aevum/models/dynamics/
cell.py`). When `False`, the candidate becomes `u_t = phi(W_x x_t)` — no
`h_{t-1}` in its own input — while the memory/decay mechanics
(`tau_t`/`alpha_t`, still computed from both `x_t` and `h_{t-1}`, and
`h_t = alpha_t*h_{t-1} + (1-alpha_t)*u_t`) are completely unchanged.
Default is `True` (existing behavior, all current tests still pass
unchanged). `scripts/overfit_single_branch.py` gained a
`--no-self-recurrence` flag wiring this through, writing to a separate
`_no_self_rec`-suffixed output directory.

Plan: run `mid` first (`--branch mid --no-self-recurrence`). If grad_norm
settles into the 5-30 range with no repeated spikes, loss decreases
smoothly, and the reconstruction sounds like speech — the diagnosis is
essentially confirmed. Then repeat for `slow` with the same flag.

If confirmed, this doesn't weaken the architecture — it sharpens it: `h_t =
alpha_t*h_{t-1} (memory) + (1-alpha_t)*F(x_t) (innovation)`, matching
AEVUM's own stated philosophy (tech_spec.md section 7) more precisely than
the original two-recurrent-path design. Cross-timescale communication
(tech_spec.md section 9) can still be preserved as `u_t^M = phi(W_x x_t +
C_F*h_t^F + C_S*h_t^S)` — context from *other* branches, without a branch
self-exciting through its own `W_hh`.

Command: `uv run python scripts/overfit_single_branch.py --branch mid
--no-self-recurrence --steps 1000`.

---

## Run 14 — `--no-self-recurrence` on `mid`: still explodes, hypothesis narrows further

`uv run python scripts/overfit_single_branch.py --branch mid
--no-self-recurrence --steps 1000` (candidate is now `u_t = phi(W_x x_t)`,
no `h_{t-1}`; memory/decay mechanics unchanged):

```text
step  125  total 13.3707  grad_norm  993.479
step  200  total 24.4088  grad_norm  396.956
step  225  total  7.4881  grad_norm 3713.068
step  275  total 10.5259  grad_norm 2231.129
step  300  total 16.7945  grad_norm 1757.271
step  325  total  5.4559  grad_norm 2833.970
...
step  550  total 26.3984  grad_norm   51.153
step  625  total  5.1050  grad_norm 1321.069
step  750  total 23.3093  grad_norm 1722.847
```

**Removing the candidate's own recurrence barely changed anything** — same
scale and frequency of explosions as Run 13's `mid` (with self-recurrence).
This rules out the candidate's `W_hh h_{t-1}` term as *the* mechanism.

### Analysis — sharper hypothesis: state-controlling-its-own-decay-rate feedback

Even with `candidate_uses_hidden=False`, `h_{t-1}` still reaches the update
through `tau_t = f(x_t, h_{t-1})` -> `alpha_t = exp(-dt/tau_t)` -> `h_t`.
The cell is not the simple linear-in-`h_{t-1}` system it was intended to
be; expanding the Jacobian:

```text
d h_t / d h_{t-1} = alpha_t * I + (h_{t-1} - u_t) * d(alpha_t)/d(h_{t-1})
```

The second term — the hidden state influencing *its own decay rate*,
creating a `h_{t-1} -> tau_t -> alpha_t -> (how much of h_{t-1} survives)
-> h_t -> ...` loop — is now the leading suspect, especially for branches
where `alpha -> 1` (mid, slow): the state persists almost entirely across
steps *while simultaneously being allowed to modulate its own persistence*.

### Next step — remove `h_{t-1}` from tau instead (one experiment at a time)

Added to `ContinuousTimeCell`: `adaptive_tau: bool = True` (`False` replaces
the learned tau network entirely with a constant `fixed_tau`) and
`tau_uses_hidden: bool = True` (keeps tau learned/adaptive but drops
`h_{t-1}` from its input: `tau_t = f(x_t)` only). Defaults unchanged, all
existing tests still pass. `scripts/overfit_single_branch.py` gained
matching `--fixed-tau <value>` and `--tau-input-only` flags (mutually
exclusive — one experiment at a time, per plan). Verified without training:
`ContinuousTimeCell(candidate_uses_hidden=False, adaptive_tau=False,
fixed_tau=0.275)` runs and produces `alpha = exp(-0.01/0.275) = 0.9643` as
expected.

**Plan, in order:**

1. `mid`, `candidate_uses_hidden=False`, fully fixed `tau=0.275` (no learned
   tau network at all — `h_t = 0.9643*h_{t-1} + 0.0357*F(x_t)`, nothing
   else). If this stabilizes (`grad_norm` 5-30, smooth loss decrease,
   speech-like output) — the culprit is specifically **state-dependent
   adaptive tau**, not the leaky integration or long memory itself. If it
   *still* explodes, the problem is deeper — long leaky integration itself
   may not work as the main speech transport channel, and the next step
   would be sweeping fixed tau in {50, 100, 200, 300} ms to find the
   stability boundary.
2. Only after (1): keep tau adaptive/learned but input-only
   (`--tau-input-only`, `tau_t = f(x_t)`, no `h_{t-1}`) — a middle ground
   that keeps plosive-vs-vowel adaptivity without the state-controls-its-
   own-persistence feedback loop.

Nothing else changes (same generator, loss, LR, clipping) — the isolation
matrix so far already narrows the space substantially: plain
frontend->generator works, `fast` works, `mid`/`slow` adaptive explode,
`mid` without candidate self-recurrence still explodes the same way.

Command: `uv run python scripts/overfit_single_branch.py --branch mid
--no-self-recurrence --fixed-tau 0.275 --steps 1000`.

---

## Run 15 — `mid` with fully fixed tau: still explodes, both recurrence hypotheses closed

`uv run python scripts/overfit_single_branch.py --branch mid
--no-self-recurrence --fixed-tau 0.275 --steps 1000` (no candidate
self-recurrence, no learned tau network at all — `alpha` is the literal
constant `exp(-0.01/0.275) = 0.9643`):

```text
step  100  total  5.3597  grad_norm    4.378
step  125  total  7.9166  grad_norm   99.038
step  150  total  5.5898  grad_norm  176.145
step  175  total 23.4401  grad_norm 1819.649
step  200  total 23.6349  grad_norm 1878.826
step  350  total  8.3021  grad_norm 1243.094
step  400  total  5.3647  grad_norm  162.437
step  425  total 24.7285  grad_norm 1813.604
step  550  total 25.7691  grad_norm  656.894
step  625  total  7.8035  grad_norm 2006.976
```

### Analysis — both recurrence-instability hypotheses definitively closed

With `alpha` a true constant (not a function of `h_{t-1}` at all), the
state-to-state Jacobian is exactly `d(h_t)/d(h_{t-1}) = alpha*I` — a
contraction (`0 < alpha < 1`), not an expanding map:
`d(h_t)/d(h_{t-k}) = alpha^k * I -> 0`. There is no exploding-recurrence
mechanism left to blame. And yet the system explodes just as badly as
Run 13/14. **This rules out both the candidate's `W_hh h_{t-1}` term (Run
14) and the state-dependent adaptive-tau feedback (this run) as the cause.**
Something else is producing these gradients.

**New hypothesis: not an unstable dynamical system, but an ill-conditioned
optimization problem.** `mid`'s channel only admits `1-alpha ~= 3.6%` new
information per 10 ms step — a severe bandwidth limit if it's the *only*
path from frontend features to the generator. To reconstruct fast acoustic
detail (plosives, fricatives, formant transitions, phase) through a channel
that attenuates each step's contribution by ~28x, upstream weights
(candidate, projection, generator) are pushed to compensate with
correspondingly large gain. Once `candidate`'s `tanh` saturates or a
downstream projection is unbounded, that compensation shows up as huge,
unstable gradients — not because the recurrence itself is mathematically
exploding, but because the optimizer is fighting a badly-conditioned
problem. This is consistent with the full pattern across every run:
`fast` (`1-alpha~20%`) trains cleanly, `mid` (`~3.6%`) and `slow` (`~0.4%`)
both fail, worse as bandwidth shrinks — exactly tracking `1-alpha`, not any
recurrence-stability quantity.

### Next step — test with a direct instantaneous residual path, not less recurrence

No further recurrence-removal tests planned — the recurrence itself has
been cleared. Instead, test the resulting architectural hypothesis
directly: give the generator an unfiltered path to the frontend feature
alongside the (still bandwidth-limited) `mid` state, so `mid` is no longer
the *only* transport channel:

```text
z_t = f_t + P(h_t^mid)
```

Implemented as `--direct-residual` on `scripts/overfit_single_branch.py`
(adds the raw frontend feature to the projected hidden state before the
generator). Keeps everything else from Run 15 unchanged — same `mid`
branch, same `--no-self-recurrence --fixed-tau 0.275`, deliberately *not*
switching to `fast` or restoring adaptivity, so this isolates the residual
path's effect specifically on the branch that has been failing hardest
under controlled conditions.

If this stabilizes (`grad_norm` 5-30, smooth decrease, speech emerges), the
architectural conclusion is: **temporal (mid/slow) states cannot serve as
the primary information transport channel — they should be a context/
memory layer on top of an instantaneous representation, not instead of it**
(matching the residual-highway idea already proposed for the decoder side
in Run 10's analysis, now on the encoder side too). Crucially, this loses
nothing for the eventual event-driven codec: the direct path only needs to
exist *before* the compression bottleneck (predictor/innovation/event
gate/quantization) — the decoder still only ever receives transmitted
events, never a continuously-streamed `f_t`. Reframing:
`representation = instantaneous innovation + temporal context`, and
`transmitted information = representation - predicted representation`.

Command: `uv run python scripts/overfit_single_branch.py --branch mid
--no-self-recurrence --fixed-tau 0.275 --direct-residual --steps 1000`.

---

## Run 16 — `mid` + direct residual: Result A, decisively

`uv run python scripts/overfit_single_branch.py --branch mid
--no-self-recurrence --fixed-tau 0.275 --direct-residual --steps 1000`
(`z_t = f_t + P(h_t^mid)`, same problematic `mid` branch, same fixed
`tau=0.275`, same no-self-recurrence candidate — only the direct path is
new):

```text
step    0  total 5.5729  wav 0.0506  mel 2.7722  stft 2.7501  grad_norm 17.496
step  100  total 4.2784  wav 0.0439  mel 2.1418  stft 2.0927  grad_norm  2.451
step  250  total 1.5416  wav 0.0509  mel 0.5348  stft 0.9558  grad_norm  6.034
step  500  total 0.6948  wav 0.0401  mel 0.1600  stft 0.4948  grad_norm 10.323
step  750  total 0.4516  wav 0.0281  mel 0.0964  stft 0.3272  grad_norm  6.576
step  999  total 0.4816  wav 0.0212  mel 0.1352  stft 0.3252  grad_norm  6.210
best loss: 0.3473 (93.8% reduction)
```

Grad norm stayed in the calm 2-13 range the entire run — no explosions at
all, unlike every previous `mid`/`slow` configuration. `wav` L1 dropped
substantially and monotonically (0.0506 -> 0.0207 best) for the first time
on a recurrent branch (previously only the free-latent and
frontend-only-linear tests showed this). **User's listening verdict: "один
в один"** (identical to target).

### Analysis

**Confirms the diagnosis decisively.** `mid`'s recurrence itself was never
the problem — being forced to serve as the *sole* transport channel for
full-bandwidth acoustic detail was. Once the generator has an unfiltered
path to `f_t`, `mid`'s narrow-bandwidth state stops being a bottleneck the
optimizer has to fight, and training becomes as clean as the free-latent
and frontend-only experiments. This validates the residual-highway idea
(originally proposed for the decoder side in Run 10, now confirmed on the
encoder side) as the fix, not a workaround.

### Next steps — restore capabilities one at a time, most valuable first

Plan (user), each step gated on the previous one succeeding:

1. **`mid` + direct residual + restore adaptive tau** (still no candidate
   self-recurrence): does reinstating `tau_t = f(x_t, h_{t-1})` stay stable
   now that `mid` isn't the sole transport path? This is the capability
   that actually matters for AEVUM (dynamic temporal resolution) — worth
   testing before anything else.
2. If (1) holds: same for `slow` + direct residual + adaptive tau. `slow`
   no longer needs to transport speech itself — residual does that; this
   tests whether `slow` can now do its *actual* job (long memory) safely.
2. Assemble `fast+mid+slow` together — still no candidate self-recurrence,
   direct residual, **no cross-timescale connections yet** — with a gated
   fusion instead of plain concatenation:
   `z_t = W_x*f_t + g_F*P_F(h_F) + g_M*P_M(h_M) + g_S*P_S(h_S)`, gates
   `g_F/g_M/g_S` initialized small (~0.05-0.1) so the model starts close to
   the already-proven `frontend -> generator` baseline and has to *earn*
   using the temporal branches rather than depending on them from step 0.
3. Only after that is stable: restore cross-timescale connections
   (tech_spec.md section 9).

**Explicitly not restoring:** the candidate's own `W_hh*h_{t-1}` term (Run
14 showed no evidence it's needed — memory already exists via
`alpha_t*h_{t-1}`, and there's no reason to add a second recurrent
mechanism just because the original spec had one).

**Reframing the encoder** (no longer "audio -> continuous state -> speech
representation", but "instantaneous representation + fast/mid/slow temporal
memory -> fused z(t)") does not weaken AEVUM: the direct path exists only
*before* the compression bottleneck (predictor / innovation / event gate /
quantization) — the decoder still only ever receives transmitted events,
never a continuously-streamed `f_t`. Temporal dynamics go back to doing
what they're suited for (memory/context/prediction), not pretending to be a
wideband transport channel.

Command: `uv run python scripts/overfit_single_branch.py --branch mid
--no-self-recurrence --direct-residual --steps 1000` (no `--fixed-tau`, no
`--tau-input-only` — both default to the original full adaptive-tau
formula).

---

## Run 17 — `mid` + direct residual + adaptive tau restored: still clean

`uv run python scripts/overfit_single_branch.py --branch mid
--no-self-recurrence --direct-residual --steps 1000` (full adaptive
`tau_t = f(x_t, h_{t-1})` restored; only `candidate_uses_hidden=False`
and the direct residual carry over from Run 16):

```text
step    0  total 5.3890  grad_norm  6.947
step  250  total 1.1334  grad_norm 10.540
step  500  total 0.5953  grad_norm 10.815
step  750  total 0.4787  grad_norm  9.290
step  999  total 0.3775  grad_norm  5.642
best loss: 0.3466 (93.6% reduction)
```

`grad_norm` stayed calm throughout (4-17 range, same as Run 16 with fixed
tau) — restoring adaptive tau did not reintroduce any instability once the
direct residual is in place. Best loss (0.3466) essentially matches Run
16's fixed-tau result (0.3473).

### Analysis

**Step 1 of the recovery plan succeeds.** Adaptive tau is not inherently
dangerous — it only became a problem (Run 6/7's tau hypothesis, later
superseded) when `mid`/`slow` states were also required to be the sole
transport channel. With the direct residual removing that burden, the full
original `tau_t = f(x_t, h_{t-1})` formula is safe to keep. This recovers
AEVUM's dynamic-temporal-resolution capability without the instability.

### Next step

Per the plan: same test on `slow` (`--branch slow --no-self-recurrence
--direct-residual`, full adaptive tau). `slow` no longer needs to carry
speech itself — this checks whether it can now safely do its intended job
(long memory) without exploding, now that residual handles transport.

Command: `uv run python scripts/overfit_single_branch.py --branch slow
--no-self-recurrence --direct-residual --steps 1000`.

---

## Run 18 — `slow` + direct residual + adaptive tau: also clean

`uv run python scripts/overfit_single_branch.py --branch slow
--no-self-recurrence --direct-residual --steps 1000` (the branch that
previously produced the worst explosions of the entire investigation — peak
grad_norm 10,534 in Run 13):

```text
step    0  total 5.3893  grad_norm  6.963
step  250  total 1.2672  grad_norm  8.650
step  500  total 0.6332  grad_norm  7.736
step  750  total 0.4258  grad_norm  5.977
step  999  total 0.3807  grad_norm 11.252
best loss: 0.3412 (93.7% reduction)
```

Grad norm stayed calm (4-12 range) the entire run, essentially matching
`mid`'s Run 17 result (0.3466). **User's verdict: speech reconstructed on
listening, same as mid.**

### Analysis — recovery plan step 2 confirmed; root cause fully resolved

Both `mid` and `slow` — the two branches responsible for every explosion
seen since Run 11 — train cleanly with adaptive tau restored, once the
direct residual path removes the sole-transport-channel burden. This closes
the investigation that began at Run 11: the problem was never adaptive tau,
never the candidate's self-recurrence, never continuous-time dynamics as a
concept — it was specifically forcing a slow-decaying state to also carry
full-bandwidth instantaneous signal.

### Next step — assemble the full multi-timescale encoder

Per the user's plan: `fast + mid + slow` together, still no cross-timescale
connections yet (prove three independent branches + direct path work
together before adding communication between them), with a **gated fusion**
instead of plain concatenation:

```text
z_t = W_x*f_t + g_F*P_F(h_t^F) + g_M*P_M(h_t^M) + g_S*P_S(h_t^S)
```

`g_F/g_M/g_S` initialized small (~0.05-0.1), so training starts close to
the already-proven `frontend -> generator` baseline (Run 12) and the
temporal branches have to *earn* their contribution rather than being
relied on from step 0 — expected to be more stable than starting with
`g=1` and hoping three simultaneously-training recurrent branches don't
fight the direct path early on.

Configuration for this test: `candidate_uses_hidden=False`, `adaptive_tau=
True`, `direct_residual=True`, no cross-timescale connections. If this
overfits cleanly, the user considers the **Stage 1 encoder architecture
found**. Next after that: reintroduce `ContinuousDecoder` and repeat the
full `audio -> encoder -> decoder -> generator` sanity test (this is what
originally produced noise, starting at Run 4/5/8) to confirm the fix holds
end-to-end.

**Decoder design insight (for later, not yet implemented):** the same
principle likely applies to the decoder. Rather than forcing `d_t^F/d_t^M/
d_t^S` to be the sole channel for event information, give the decoder a
direct event-injection path too:

```text
y_t = W_event*e_t + P_F(d_t^F) + P_M(d_t^M) + P_S(d_t^S)   (when an event arrives)
y_t =              P_F(d_t^F) + P_M(d_t^M) + P_S(d_t^S)   (between events)
```

This maps cleanly onto AEVUM's actual event-driven design goal: direct
injection when there's new information to transmit, continuous dynamics
carrying the signal between events. Not yet tested — queued after the
encoder-side fix is confirmed end-to-end.

Command: `uv run python scripts/overfit_single_branch.py` does not support
multi-branch fusion yet — this next test needs a new script (not yet
written).

---

## Run 19 (setup) — `scripts/overfit_multiscale_encoder.py` written

Implements the full `fast+mid+slow` encoder described above: three
`ContinuousTimeCell`s (each `candidate_uses_hidden=False`, full adaptive
tau, no cross-timescale connections), each with its own projection, fused
via `z_t = W_x*f_t + g_fast*P_fast(h_fast) + g_mid*P_mid(h_mid) +
g_slow*P_slow(h_slow)`. `W_x` initialized to the identity (`Conv1d` with an
identity weight), gates initialized to `--gate-init` (default 0.1).

Verified with a 2-step smoke run (not real training, just a shape/crash
check): runs cleanly, `total` loss and `grad_norm` are sane (5.76, then
6.18), gate values move slightly as expected (`g_fast/g_mid/g_slow`
0.1 -> ~0.10, ~0.10, ~0.098). No shape or runtime errors.

Command for the real run: `uv run python scripts/overfit_multiscale_encoder.py
--steps 2000` (writes to `outputs/multiscale_encoder/`). This will be
slower per step than the single-branch tests — three sequential
`ContinuousTimeCell` calls per timestep instead of one — expect roughly
3x Run 13-18's per-step time.

---

## Run 19 — Full fast+mid+slow encoder + gated fusion: stopped early, looks clean

`uv run python scripts/overfit_multiscale_encoder.py --steps 2000`, stopped
at step 500/2000 — same stopping-early judgment call as Runs 10/12 (trend
already unambiguous):

```text
step    0  total 5.7583  grad_norm 24.081  g_fast=0.1010 g_mid=0.1010 g_slow=0.0990
step  100  total 3.1198  grad_norm  7.294  g_fast=0.0916 g_mid=0.0963 g_slow=0.0870
step  250  total 1.4295  grad_norm 11.950  g_fast=0.0881 g_mid=0.1040 g_slow=0.0814
step  400  total 0.8822  grad_norm 10.307  g_fast=0.0900 g_mid=0.1054 g_slow=0.0794
step  500  total 0.6746  grad_norm 10.186  g_fast=0.0902 g_mid=0.1058 g_slow=0.0778
best loss: 0.6438 (88.8% reduction at the 25% mark)
```

`grad_norm` stayed calm (3-12) the entire run — no explosions with all
three branches training simultaneously. Gates stayed near their 0.1 init
throughout (0.078-0.106 range) rather than collapsing to zero or
runaway-growing, meaning all three branches are contributing without any
one dominating pathologically. Trajectory closely tracks the single-branch
`mid`/`slow` + residual results (Run 17/18) at the same step count.

### Analysis

Consistent with success — matches every prior indicator that the direct
residual + small-init gated fusion generalizes cleanly from single branches
to the full three-branch encoder. Awaiting listening confirmation on
`outputs/multiscale_encoder/recon_best.wav` before declaring the Stage 1
encoder architecture found (per this report's own established discipline:
loss trends alone were misleading multiple times earlier in this
investigation, e.g. Run 5).

### Next step

If listening confirms clean/intelligible speech: the Stage 1 encoder
architecture is considered found. Then: reintroduce `ContinuousDecoder` and
repeat the full `audio -> encoder -> decoder -> generator` sanity test
(originally Run 4/5/8, where noise was first observed) to confirm the fix
holds end-to-end, before deciding whether to update the actual
`aevum.models` package (currently `ContinuousEncoder`/`autoencoder.py`
still reflect the pre-fix design) or continue iterating via standalone
scripts.

**Listening result (user, at step 500/2000, loss 0.6438):** minor
crackling/hoarseness, otherwise the same as target. Consistent with just
needing more steps, not a new problem — every prior single-branch +
residual test only reached clean "identical to target" quality around loss
0.3-0.4 (Run 16: 0.3473, Run 17: 0.3466, Run 18: 0.3412), and this combined
run was stopped well short of that range. User's call: sufficient
confirmation to move on to the full pipeline test.

---

## Run 20 (setup) — `scripts/overfit_full_pipeline_v2.py` written

Reassembles the complete pipeline with the fixed encoder:
`audio -> frontend -> {fast,mid,slow}+direct path (gated fusion) -> z_t ->
ContinuousDecoder -> generator -> wav`. Encoder side matches Run 19 exactly
(`candidate_uses_hidden=False`, full adaptive tau, no cross-timescale
connections, gates init 0.1, `W_x` identity-initialized); `ContinuousDecoder`
and `generator` are the unmodified `aevum.models` classes (both cleared in
Runs 9/10).

Verified with a 2-step smoke run: runs cleanly, no shape/runtime errors,
sane loss and grad_norm (5.53, then 6.98 — some step-to-step noise expected
this early, matching the pattern seen in every other run at step 0-1).

This is the test that matters most: it directly repeats the original
Run 4/5/8 setup (which produced noise) with the only change being the fixed
encoder. If this reconstructs clean speech, the root-cause diagnosis and
fix from Runs 11-19 are confirmed end-to-end.

Command: `uv run python scripts/overfit_full_pipeline_v2.py --steps 2000`
(slower per step than Run 19 — adds `ContinuousDecoder`'s own sequential
200-step loop on top of the three-branch encoder loop).

---

## Run 20 — Full pipeline with fixed encoder: explodes again, now in the decoder

`uv run python scripts/overfit_full_pipeline_v2.py --steps 2000` (fixed
`fast+mid+slow`+direct-residual encoder from Run 19, unmodified
`ContinuousDecoder` and `generator`):

```text
step    0  total  5.5305  grad_norm    16.011  gates ~0.10 each
step  175  total  3.8405  grad_norm    13.204  best 3.7050
step  200  total  3.9849  grad_norm   174.267  best 3.6237
step  225  total  6.2082  grad_norm   617.740
step  350  total  4.5261  grad_norm   463.687
step  375  total  7.8169  grad_norm   737.637
step  450  total  3.4173  grad_norm    84.384  best 3.1449   <- best of the whole run
step  600  total  6.7742  grad_norm 19545.418  (largest grad_norm in this entire investigation)
step  700  total 13.8431  grad_norm  3359.545
step  775  total 19.3905  grad_norm  3684.843
step  825  total  9.0613  grad_norm 13694.688
step  850  total  5.3479  grad_norm     5.488  <- "calm" from here on
step 1999  total  5.3454  grad_norm     1.183  wav=0.0431 (near the "silence" value seen at step 0 of Run 4)
```

Gates drift upward throughout the unstable region (`g_fast` 0.099->0.11+,
`g_slow` 0.10->0.15+) before the run settles into a low-gradient plateau at
`total~5.34` — worse than the run's own best (3.1449) and barely better
than step 0 (5.5305). This late "calm" phase is not convergence; it looks
like collapse into a degenerate near-silent solution (matching the
`wav~0.043`, "mostly silence" signature first seen at step 0 of Run 4/8,
before any real training).

### Analysis

**The exact failure signature from Runs 4/5/8/11 reappears, one level
downstream.** The encoder fix (Run 16-19) is confirmed working in isolation
(`encoder -> generator` direct, no decoder, was clean in Run 19). Reading
`src/aevum/models/decoder.py` confirms `ContinuousDecoder` is architecturally
identical to the *pre-fix* encoder in the relevant way: it feeds
`MultiTimescaleDynamics` (default `candidate_uses_hidden=True`, active
cross-timescale connections, adaptive tau) and produces output purely as
`to_output(norm(fused fast/mid/slow states))` — **no direct path from the
input `u_t`/`z_t` to the output at all.** This is precisely the
"slow-decaying state forced to be the sole transport channel" pattern
already diagnosed and fixed on the encoder side (see
`../lessons-continuous-time-dynamics.md`), now hitting the decoder instead:
the encoder is producing meaningful full-bandwidth `z_t`, and the decoder
has no way to pass it through without funneling everything through its own
low-bandwidth mid/slow states.

This also validates the decoder-side design insight queued back in Run
10/18's analysis ("give the decoder a direct event-injection path too") —
it wasn't just a nice-to-have symmetry argument, it's now empirically
necessary.

### Next step — same fix, decoder side, decoder itself untouched first

Following the same economical approach that worked for the encoder (Run
16: adding the residual alone fixed `mid` without first touching
self-recurrence or tau) — add a direct skip from `z_t` to the generator
input, alongside (not instead of) `ContinuousDecoder`'s own output,
**without modifying `ContinuousDecoder` itself yet**:

```text
y_for_generator = ContinuousDecoder(z_t) + g_decoder * W_skip(z_t)
```

`W_skip` identity-initialized (same trick as `W_x` on the encoder side),
`g_decoder` small-init (~0.1), so training starts close to a system where
`ContinuousDecoder`'s output barely matters and has to earn its
contribution — mirroring the encoder recovery exactly.

Implemented in `scripts/overfit_full_pipeline_v3.py` (not yet run/verified).
If this stabilizes, the decoder's own internal self-recurrence/cross-
connections/adaptive-tau can stay as-is (no need to repeat the whole
single-branch ablation sequence on the decoder side, unless this simple fix
doesn't work).

Command: `uv run python scripts/overfit_full_pipeline_v3.py --steps 2000`.

---

## Cross-project note

The general lesson from Runs 11-18 (a slow-decaying continuous-time state
cannot safely be the sole transport channel for high-bandwidth signal; give
the consumer a direct/residual path instead) is written up independently of
AEVUM-specific details at `../lessons-continuous-time-dynamics.md` (one
level up from this repo, at the Manifestro root), since it applies to any
continuous-time/leaky-integrator recurrent component and may be relevant to
other Manifestro projects.

---
