Installation
============

ReconEval supports two install paths.

Metrics only (lightweight pip install)
--------------------------------------

If you just want to score your own ``(true, reconstructed)`` AnnData
pair with the metrics API and run ``tutorials/metrics.ipynb``, a vanilla
pip install is enough:

.. code-block:: bash

    git clone https://github.com/theislab/ReconEval.git
    cd ReconEval
    python -m venv .venv && source .venv/bin/activate
    pip install -e .

Verify:

.. code-block:: bash

    python -c "from sc_reconstruction.metrics import compute_all_metrics, funky_heatmap; print('OK')"

Full benchmark (conda env)
--------------------------

To run the other three tutorials (``end_to_end.ipynb``, ``fm.ipynb``,
``latent_shift.ipynb``) or any of the training and eval drivers under
``experiments/``, use the conda environment file. It pins compatible
versions of torch, scvi-tools, lightning, scvi, dask, and the STATE
foundation-model package.

.. code-block:: bash

    conda env create -f envs/cstm_scvi_env.yaml
    conda activate reconeval

Optional environment variables
------------------------------

The training and eval drivers read a small set of env vars:

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Variable
      - What it controls
    * - ``RECONEVAL_OUT``
      - Output root for weights, checkpoints, Hydra outputs (default
        ``~/reconeval_outputs/``).
    * - ``CELLFLOW_SRC``
      - Path to a cellflow source clone for ``03_latent_shift``. Skipped
        if unset.
    * - ``STATE_SRC``
      - Path to a STATE source clone for SE embedding. Skipped if unset.
