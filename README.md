# Latent-Action World Model with a Wan2.1-Fun DiT forward model

Code for **[Cross-Embodiment Robot Foundation World Models with Latent Actions](https://arxiv.org/abs/2610.10846)**
· [Project page](https://lacwm.github.io/)

A latent-action world model whose forward (world) model is the **Wan2.1-Fun-1.3B-Control**
video-diffusion transformer. The whole pipeline runs on the **Wan VAE**: given a few history
frames and an action, the model denoises the future frames.

This release covers:

1. **Cross-embodiment pretraining** on ABC + AgiBot + DROID + EgoDex. The world model is
   conditioned **only on the latent action** inferred from video; ground-truth actions are
   used solely as the target of a motion decoder that anchors the latent space.
2. **Adaptation to an unseen embodiment (LIBERO)** in three stages, ending with an action
   projector that maps the new robot's raw actions into the pretrained latent space.

**To run pretraining on a new cluster, follow
[Launching pretraining on a new cluster](#launching-pretraining-on-a-new-cluster).**

## Architecture

```
rgb ──WanVAE(per-frame)──▶ per-frame latents ──inverse_model──▶ latent action z
                                                                     │
           ┌─────────────────────────────────────────────────────────┤
           ▼                                                         ▼
  ActionToControl(z) ──▶ control latent (16ch)          ActionToTokens(z) ──▶ clip_fea
           │                                                         │  (i2v cross-attention)
history ──WanVAE──▶ reference latent (16ch)                          │
future  ──WanVAE + noise──▶ noisy latent (16ch)                      │
           ▼                                                         ▼
     Wan DiT   x = noisy ⊕ [control | reference]  (in_dim 48)  +  clip_fea  ──flow-matching──▶ future
           │
  z ──motion decoder──▶ ground-truth motion   (pretraining only; L1, weight 0.1)
```

The DiT sees the latent action through **both** conditioning pathways — the control
channel (frame-aligned, enters at the patch embedding) and the repurposed i2v clip branch
(cross-attention). No ground-truth action, pose or camera signal reaches it directly.

| | Model class | where the latent action comes from |
|---|---|---|
| **Pretraining**, adaptation stage 1 | `LatentActionDiTModel` | inverse model, from video frames |
| Adaptation stages 2 and 3 | `ExplicitActionDiTModel` | action projector, from the robot's raw action + camera motion |

### Key facts

- **Wan DiT** (`WanTransformer3DModel`): `dim 1536, ffn 8960, 12 heads, 30 layers,
  in_dim 48, out_dim 16`, patch `(1,2,2)`, `model_type i2v`. Pretraining trains it
  **end-to-end (dense)**; adaptation uses LoRA (rank 1024) on `q/k/v/o`.
- **Wan VAE** (`AutoencoderKLWan`): 16-ch latent, 8× spatial / 4× temporal causal, run in
  **fp32**. Clips need `F ≡ 1 (mod 4)`, and the history length must be `1 + 4m` so the
  history/future split lands on a latent boundary.
- **Clip**: 17 frames = **9 history + 8 future**, the same in every stage. History frames
  are spaced 2× wider than future frames (`history_stride_mult: 2`, after Ctrl-World's
  sparse history), so one clip spans `40·m + 45 = 125` raw steps.
- **Views**: each camera is encoded separately by the VAE and the latents are concatenated
  along width (`latent_views`). 3 view slots in pretraining (missing views zero-padded,
  none dropped), 2 for LIBERO.
- **Latent action**: 128-d. The motion decoder splits it evenly — the first half decodes the
  robot action, the second half the camera motion.
- **Masked flow loss**: invalid frames and constant (padded) views are inputs only. The
  loss is computed on future frames.

## Launching pretraining on a new cluster

Everything below runs from the repo root on a Slurm cluster. Replace `<acct>` / `<part>`
with your Slurm account and partition (or `export SBATCH_ACCOUNT=... SBATCH_PARTITION=...`
once and drop the flags). Run steps 1–3 once; steps 4–7 are the training itself.

**What you need**

- 8 GPUs on one node for the run (80 GB-class recommended: the DiT is trained densely), plus
  1 GPU for the smoke test and batch sweep.
- Disk under the directory you run setup from, for the default 10,000 episodes per source:
  ABC ≈ 0.5 GB/episode while downloading (raw files are deleted once preprocessed), AgiBot
  ≈ 65 MB/episode, EgoDex ≈ 320 GB per part (zip removed after unpacking), DROID 10 chunks
  of two camera views, Wan weights ≈ 20 GB. Checkpoints are ≈ 20 GB each, up to three kept.
- Conda, and outbound access to Hugging Face and `ml-site.cdn-apple.com`.

### 1. Accounts

```bash
huggingface-cli login          # then accept the terms on both gated dataset pages:
#   https://huggingface.co/datasets/XDOF/ABC-130k
#   https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha
wandb login                    # or: export WANDB_MODE=offline   (training logs to wandb)
```

Batch jobs often cannot read your home-directory token; if a Slurm job fails with
`GatedRepoError`/`401`, pass it explicitly: `HF_TOKEN=$(cat ~/.cache/huggingface/token) sbatch --export=ALL,HF_TOKEN ...`.

### 2. Environment, weights and data

```bash
git clone https://github.com/qingh097/Latent_action_world_model.git && cd Latent_action_world_model
./setup_training.sh            # long: run it in tmux/screen or as a CPU batch job
```

Expect it to take a day or more, almost all of it downloading: the AgiBot archives are
streamed at Hub speed (~30 MB/s measured) and EgoDex comes as ~320 GB zips.

This creates the conda env `lacwm-dit`, downloads the Wan2.1-Fun weights and VideoX-Fun,
builds the cached null-prompt embedding, downloads and prepares the four datasets (only the
files the loaders read, capped at 10,000 episodes per source — the most pretraining reads),
builds the manifests, and writes `.lacwm_env` with every path. All launchers read
`.lacwm_env`, so nothing else needs exporting. It ends with a validation report.

It is idempotent: if it is interrupted, re-run it (or one step: `./setup_training.sh datasets`,
`manifests`, `validate`, with the `lacwm-dit` env activated). Toggle or cap a source with
e.g. `ABC_ENABLE=0` or `AGIBOT_LIMIT=5000`.

### 3. Check the data (CPU, a few minutes per source)

```bash
mkdir -p slurm_logs
sbatch --account=<acct> --partition=<part> tools/launch/verify_data.sbatch
```

The log must end in `all checks passed`. `camera_moves=False` notes for DROID, ABC (fixed
cameras) and AgiBot (constant head extrinsics) are expected.

### 4. Smoke test (1 GPU, ~5 minutes)

```bash
sbatch --account=<acct> --partition=<part> --gres=gpu:1 --job-name=pretrain_smoke \
  tools/launch/train.sbatch pretrain_smoke 1
```

Pass if the log `slurm_logs/pretrain_smoke_<jobid>.log` shows 12 iterations and a
`Validation completed for iteration 11` line, and contains no `ERROR __main__`. **Slurm
reports `COMPLETED` even when training crashed**, so always check the log.

### 5. Pick the batch size (1 GPU of the type you will train on)

```bash
sbatch --account=<acct> --partition=<part> --gres=gpu:1 --cpus-per-task=12 --mem=150G \
  tools/launch/batch_sweep.sbatch 4 8 16
```

Each size prints a few `loss:` lines or `out of memory`. Use the largest size that fits;
the config default is 16 per GPU (effective batch 128 on 8 GPUs). If 16 does not fit, add
`data_loader.batch_size=<n>` to the launch command below.

### 6. Launch

```bash
sbatch --account=<acct> --partition=<part> --gres=gpu:8 --job-name=pretrain \
  tools/launch/train.sbatch pretrain 8            # [data_loader.batch_size=<n>]
```

60,000 steps. The script asks for 3 days (`--time` overrides it). **Resuming is automatic**:
if the job hits its time limit or is preempted, submit the exact same command again and it
continues from the last checkpoint (saved every 1,000 steps).

### 7. Monitor, and what to send back

- Log: `slurm_logs/pretrain_<jobid>.log` (`[step] loss: ...` every 50 steps, validation every
  1,000). Weights & Biases project `lacwm`, run `pretrain_abc_agibot_droid_egodex`.
- `flow loss is fully masked` or a repeating `non-finite loss` warning means something is
  wrong — stop and report it.
- Checkpoints go to `$LACWM_RUNS/pretrain_abc_agibot_droid_egodex/` (`LACWM_RUNS` is in
  `.lacwm_env`): `snapshot.pt` (latest), `snapshot.<step>.pt` (newest 10k-step stamp) and
  the best-validation snapshot. **The deliverable is `snapshot.pt` after step 60,000**
  (plus the log). Visualizations of predicted vs. real futures are written next to them.

## Setup details

Python 3.10 + PyTorch 2.7.1 (cu128). To set up by hand instead of `setup_training.sh`:

```bash
conda env create -f environment.yaml      # env "lacwm-dit"
```

| variable | default (relative to the repo root) | what |
|---|---|---|
| `WAN_DIR` | `checkpoints/wan_fun_1.3b_control` | `alibaba-pai/Wan2.1-Fun-1.3B-Control` weights + cached null-prompt embedding |
| `VIDEOX_HOME` | `VideoX-Fun` | provides `videox_fun.models.*` |
| `LACWM_DATA` | `data` | datasets + manifests |

`null_prompt_umt5.pt` (a cached empty-prompt text embedding) is not part of the official
Wan release; generate it once, with `WAN_DIR` and `VIDEOX_HOME` set:

```bash
PYTHONPATH=$VIDEOX_HOME python tools/launch/gen_null_prompt.py   # writes $WAN_DIR/null_prompt_umt5.pt
```

All launchers read their paths from `tools/launch/env.sh` (which sources `.lacwm_env` when
present); export any variable before `sbatch` to override it. Outputs go under `LACWM_BASE`
(default `lacwm_out/`) unless `.lacwm_env` sets `LACWM_RUNS`.

## Data

| source | embodiment | views | camera motion |
|---|---|---|---|
| ABC-130k | bimanual YAM arms | top + 2 wrists | zero (fixed camera) |
| AgiBot World Alpha | humanoid, two arms | head + 2 wrists | head camera |
| DROID | Franka | exterior + wrist | zero (fixed camera) |
| EgoDex | human hands | egocentric | head camera |

**Camera motion** is the motion of the ego / third-person camera and nothing else. Robot
and hand motion belongs to the first half of the latent action, not the camera slot.

Only part of each dataset is read, and `setup_training.sh` (via `tools/data/fetch_data.py`
for ABC and AgiBot) fetches only that part:

| dataset | files the loaders open |
|---|---|
| ABC | per episode: `top.mp4`, `left_wrist.mp4`, `right_wrist.mp4`, `states.npz`, produced from the raw `episode.mcap` by `robot_wm/datasets/abc/preprocessing/` |
| AgiBot | `head_color`, `hand_left_color`, `hand_right_color` videos; the three `*_extrinsic_params_aligned.json`; `proprio_stats.h5`. Streamed out of the release's tar archives; fisheye, depth and `task_info/` are skipped. |
| DROID | LeRobot: `data/chunk-*/episode_*.parquet` + videos for `exterior_image_1_left` and `wrist_image_left` (or the TFDS release) |
| EgoDex | `<task>/<n>.hdf5` and `<task>/<n>.mp4` |

Manifests hold absolute paths and are built per machine by `tools/data/build_manifests.py`
(`setup_training.sh` calls it). They are shuffled with a fixed seed, because
`subsample_traj` keeps a prefix of the manifest. DROID needs no manifest. For data that is
already on disk in another layout:

```bash
python tools/data/build_manifests.py --out $LACWM_DATA --egodex-root ... --abc-root ...
```

DROID is read through one of two loaders that take the same arguments, selected by
`DROID_LOADER` / `DROID_DATA_DIR`. The default is the TFDS release; `setup_training.sh`
downloads the LeRobot copy and selects its loader in `.lacwm_env`.

**Short episodes.** When an episode is shorter than the 125-step clip, the window is shifted
to start before frame 0: the missing history frames repeat frame 0 (as at the start of a
rollout) while every supervised future frame comes from a real frame.

## Pretraining recipe

`configs/experiment/pretrain.yaml`: ABC + AgiBot + DROID + EgoDex, 60k steps, dense DiT
(lr 3e-5), latent action on both conditioning pathways, motion decoder on, no camera-view
dropping. Arguments after the GPU count in `train.sbatch` are passed to Hydra, so a source
can be left out, for example ABC:

```bash
sbatch ... tools/launch/train.sbatch pretrain 8 \
  '~dataset.datasets.ABC' '~val_dataset.datasets.ABC' '~viz_dataset.datasets.ABC'
```

- **Cross-augmentation.** The inverse model reads an independently augmented copy of the
  clip while the forward model is supervised on the original, so the latent action cannot
  smuggle future appearance through as a shortcut.
- **Human data** (EgoDex) gets a stronger photometric augmentation than the robot data.
  Augmentations are photometric only and shared across all frames of a clip.
- **Batch size.** The trainer has no gradient accumulation, so the effective batch is
  exactly GPUs × `data_loader.batch_size`.

## Adaptation to an unseen embodiment

Three stages on LIBERO (agentview + wrist), each initialised from the previous one:

```bash
export LACWM_PRETRAIN_CKPT=<pretraining run dir>/snapshot.pt
sbatch ... tools/launch/train.sbatch libero_stage1 <n>   # IDM + FDM
sbatch ... tools/launch/train.sbatch libero_stage2 <n>   # projector, FDM frozen
sbatch ... tools/launch/train.sbatch libero_stage3 <n>   # projector + FDM
```

| stage | model | trains | why |
|---|---|---|---|
| 1 | latent action | inverse + forward model (LoRA) | adapt to the new visuals while still using latent actions. **No motion decoder**: its job was to anchor the latent space during pretraining, and a freshly initialised head would only pull the latent away from that space. |
| 2 | explicit action | action projector only (forward model frozen) | force the projector to map raw actions into the latent space the world model already understands |
| 3 | explicit action | projector + forward model (LoRA) | joint refinement; at inference only the projector and forward model are used |

Stages 2 and 3 read the previous stage's snapshot from `$LACWM_RUNS/libero_stage{1,2}/snapshot.pt`,
and every stage refuses to start (`require_load_path`) if its checkpoint is missing.
LIBERO is expected at `$LACWM_DATA/libero/manifest.csv`.

## Smoke tests

Every recipe has a `*_smoke` variant that inherits it: 12 iterations at batch 1 with
validation and visualization, no checkpoints written, and a timestamped run directory so
smokes never resume from one another. The adaptation smokes run without the earlier
stage's checkpoint, so they check wiring, not learning.

```bash
sbatch ... --gres=gpu:1 tools/launch/train.sbatch pretrain_smoke 1
sbatch ... --gres=gpu:1 tools/launch/train.sbatch libero_stage2_smoke 1
```

## Gotchas worth knowing

- **A crashed run still exits with Slurm `COMPLETED 0:0`**: the trainer catches the
  exception. Check the log for `ERROR __main__`; the job state tells you nothing.
- **`flow loss is fully masked`** in the log means the clip does not fit the episodes and the
  batch trained on nothing.
- **Resuming** works because real runs use a fixed run directory; a preempted job that is
  resubmitted continues from `snapshot.pt`. Don't add a timestamp to it.
- Submit from the **repo root**: the launchers locate `tools/launch/env.sh` from the
  submission directory, and put this checkout first on `PYTHONPATH`.

## Layout

```
projects/latent_action_models/
  train.py                              Hydra entry point
  lam/latent_action_dit_model.py        LatentActionDiTModel
  lam/explicit_action_dit_model.py      ExplicitActionDiTModel (action projector)
  configs/experiment/                   pretrain, libero_stage{1,2,3}, and their *_smoke
robot_wm/
  modeling/networks/wan_forward_model.py   WanForwardModel: Wan DiT (dense or LoRA),
                                           ActionToControl, ActionToTokens
  modeling/networks/st_inverse_model.py    inverse model
  modeling/networks/action_encoder.py      action projector MLP
  modeling/tokenizers/rgb/wan_vae.py       Wan VAE tokenizer (fp32)
  datasets/multi_dataset.py                dataset mixing, camera re-referencing, augmentation
  datasets/utils.py                        sparse_history_inds (clip sampling)
  datasets/{abc,agibot,droid,egodex,libero}/   per-source loaders
tools/
  data/fetch_data.py, data/build_manifests.py, data/verify_pretrain_data.py
  launch/env.sh, train.sbatch, verify_data.sbatch, batch_sweep.sbatch,
  launch/gen_null_prompt.py   (builds the cached null-prompt embedding the model needs)
setup_training.sh             provisions a new machine end to end
```

## Citation

```bibtex
@article{huang2026crossembodiment,
  title   = {Cross-Embodiment Robot Foundation World Models with Latent Actions},
  author  = {Huang, Huang and Yenamandra, Sriram and Majumdar, Arjun and Aljalbout, Elie and
             Nagarajan, Tushar and Yang, Tsung-Yen and Rai, Akshara and Rabbat, Michael and
             Fei-Fei, Li and Wu, Jiajun and Wu, Tingfan and Meier, Franziska},
  journal = {arXiv preprint arXiv:2610.10846},
  year    = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE). Modules adapted from
[torchtitan](https://github.com/pytorch/torchtitan) and
[PyTorch examples](https://github.com/pytorch/examples) (BSD-3-Clause) say so in their
headers. The Wan2.1-Fun weights, VideoX-Fun and the four datasets are not part of this
repository and come under their own licenses and terms of use (AgiBot World, for example,
is non-commercial); check them before using a trained model.
