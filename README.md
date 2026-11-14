# FilaNet: Solar Filament Segmentation Challenge 2026

Our solution to the Solar Filament Segmentation Challenge 2026 (Kaggle, MAGFiLO dataset). FilaNet is an
EdgeAttNet-style U-Net that uses deformable convolutions throughout and edge-guided self-attention at the
bottleneck. It segments whole 2048×2048 GONG H-α frames. Hysteresis thresholding and fragment merging then turn
the probability map into individual filaments.

`pipeline.ipynb` runs the whole pipeline with the trained weights in `weights/`, from the raw test images to
`submission.csv`, and checks the result against the file we submitted. Nothing is retrained.

## Layout

| Path | Contents |
|---|---|
| `pipeline.ipynb` | End-to-end notebook (executed, with outputs) |
| `weights/` | Trained FilaNet weights in four files under 50 MB each, with SHA-256 sums in `manifest.json` |
| `submitted/submission_filanet7.csv` | The predictions we submitted |
| `src/img_prepro_v2.py` | Preprocessing: limb detection, limb-darkening and cloud flattening, fixed intensity window, CLAHE |
| `src/model.py` | FilaNet |
| `src/release.py` | Checks and joins the weight files |
| `src/inference.py` | Probability maps with 8-fold test-time augmentation |
| `src/submission.py` | Hysteresis thresholding, fragment merging, RLE submission rows |
| `src/metrics.py` | Local Panoptic Quality, ported from the organisers' notebook |
| `src/dataset.py`, `src/losses.py`, `src/train.py` | Training: data, loss (Tversky + clDice + BCE, centreline head), loop |
| `src/sweep.py` | Offline search for the post-processing operating point on validation |
| `src/predict.py` | Command-line version of the notebook's inference steps |
| `report/` | Technical report (LaTeX source and PDF) |

## Reproducing the submission

```bash
pip install -r requirements.txt
kaggle competitions download -c filament-segmentation-2026   # unzip to data/MAGFiLO_1.0_Kaggle_2026
jupyter nbconvert --to notebook --execute pipeline.ipynb      # or open it and run all cells
```

Or, from the command line:

```bash
python src/img_prepro_v2.py data/MAGFiLO_1.0_Kaggle_2026/test/test_images work/processed_test
python src/predict.py --checkpoint <see note> --tta --point 0.99,300,0.7 --merge-gap 10 \
    --images work/processed_test --out submission.csv
```

`predict.py` takes training checkpoints. To use it with the released weights, first write them out as one checkpoint:
`python -c "import sys; sys.path.insert(0,'src'); import torch; from release import load_state_dict; torch.save({'model': load_state_dict('weights')}, 'filanet7.pt')"`

A 16 GB GPU runs the test set in about 40 minutes with test-time augmentation.
