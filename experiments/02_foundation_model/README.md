# 02_foundation_model — FM-based reconstruction

Two-step pipeline:

1. **Embed** a dataset with a pre-trained FM (SE / scGPT / scConcept /
   SCimilarity) → one zarr / h5ad of cell embeddings.
2. **Decode** those embeddings back to gene expression with a downstream
   MLP, Transformer or KNN decoder.

The FM weights are frozen — no fine-tuning. Matches the
`tutorials/fm.ipynb` recipe.

## Layout

```
02_foundation_model/
├── configs/
│   ├── base_pretrained_emb.yaml      # root for embed-side scripts
│   ├── base_pretrained_dec_hvg.yaml  # root for decoder-side scripts
│   ├── pretrained/
│   │   ├── SE.yaml
│   │   ├── scGPT.yaml
│   │   ├── scConcept.yaml
│   │   └── scimilarity.yaml
│   ├── decoder/
│   │   ├── MLP.yaml
│   │   ├── Transformer.yaml
│   │   └── KNN.yaml
│   ├── model/eval/
│   │   └── AE.yaml                   # eval-side AE reference
│   └── data/
│       ├── pbmc.yaml
│       ├── tahoe.yaml
│       └── luca.yaml
├── codes/
│   ├── SE_emb.py                     # FM-specific embed (zarr loop)
│   ├── SE_emb_adata.py               # FM-specific embed (in-memory h5ad)
│   ├── scGPT_emb.py
│   ├── scConcept_emb.py
│   ├── scimilarity_emb.py
│   ├── decoderonly_hvg.py            # decoder w/ Hydra
│   ├── decoderonly_hvg_tsfm.py       # transformer decoder variant
│   ├── eval_decoder_statistical.py   # statistical scoring of decoder outputs
│   └── eval_decoder_biological.py    # biological scoring of decoder outputs
└── submit/
    └── decoderonly_grid.sbatch       # Hydra --multirun decoder sweep
```

## Eval drivers

Once you have a trained decoder checkpoint from `decoderonly_hvg.py` /
`decoderonly_hvg_tsfm.py`, score it with:

```bash
python experiments/02_foundation_model/codes/eval_decoder_statistical.py \
  pretrained=SE decoder=MLP data=pbmc +metric=decode_statistical
```

Both eval drivers use Hydra root `base_eval_decode` and rely on the
metric subgroups under `configs/metric/`.

## Step 1: embed

Each FM has its own driver + Hydra config:

```bash
# SE (uses the STATE package)
STATE_SRC=/path/to/state/src conda activate cstm_scvi_env
python experiments/02_foundation_model/codes/SE_emb.py \
  total_parts=1 parts=1 \
  data_args.output_dir=/path/to/SE_emb.zarr

# scGPT
conda activate scgpt
python experiments/02_foundation_model/codes/scGPT_emb.py \
  total_parts=1 parts=1

# scConcept
conda activate scconcept_env
python experiments/02_foundation_model/codes/scConcept_emb.py \
  total_parts=1 parts=1

# SCimilarity
conda activate scimilarity_env
python experiments/02_foundation_model/codes/scimilarity_emb.py \
  total_parts=1 parts=1
```

`total_parts` × `parts` shards the combination grid so multiple workers
can split the corpus.

## Step 2: decode

The decoder reads embeddings from disk and trains an MLP / Transformer /
KNN head to predict gene expression. Hydra-driven with subgroup defaults:

```bash
python experiments/02_foundation_model/codes/decoderonly_hvg.py \
  pretrained=<SE|scGPT|scConcept|scimilarity> \
  decoder=<MLP|Transformer|KNN> \
  data=<pbmc|tahoe|luca>
```

For the transformer-specific driver:

```bash
python experiments/02_foundation_model/codes/decoderonly_hvg_tsfm.py \
  pretrained=SE decoder=Transformer data=pbmc
```

## Sweep

`submit/decoderonly_grid.sbatch` is a Hydra `--multirun` sweep over
decoder hidden sizes and number of layers. Edit the override list at the
top of the file before submitting.

## Quick smoke test

```bash
conda activate cstm_scvi_env
RECONEVAL_OUT=/tmp/reconeval_smoke STATE_SRC=/lustre/groups/ml01/code/xiaotong.fu/state/src \
python experiments/02_foundation_model/codes/decoderonly_hvg.py \
  pretrained=SE decoder=MLP data=pbmc \
  decoder.max_epochs=1 decoder.min_epochs=1
```

Expected output: an MLP decoder checkpoint under
`${RECONEVAL_OUT}/weights/pbmc/SE_MLP/.../<filename>.ckpt`.

## Conda envs

| Env | Drivers |
|---|---|
| `cstm_scvi_env` | `SE_emb*`, `decoderonly_*` (needs the `state` package) |
| `scgpt` | `scGPT_emb.py` |
| `scconcept_env` | `scConcept_emb.py` |
| `scimilarity_env` | `scimilarity_emb.py` |
