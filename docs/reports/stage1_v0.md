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
