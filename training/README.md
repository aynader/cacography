# Training the handwriting model

The renderer writes every word with a generative model trained on real pen recordings, then reads it back to catch bad ones. This folder is the code that trained the shipped model, from raw datasets to the files in `models/`.

## The model

- **Writer** (`handmodel.py`): a 3-layer LSTM (512 units) with a mixture-density output over pen steps, conditioned on a per-writer style vector. Instead of learning where it is in the text by soft attention, it is *told* which character it is writing and predicts when to move on. That removes the classic failure where attention slips and letters get skipped or repeated.
- **Checker** (`train_reader.py`, `train_checker.py`): a CTC line reader that reads generated words back. A word that doesn't read as its text is written again.
- **Shape check** (`shapecheck.py`): flags scribbles the reader still manages to read, using limits measured on real writing.
- **Symbol bank** (`build_symbol_bank.py`): real handwritten samples of about 200 math symbols. They are used as they are, with no model involved.

## Data

| Dataset | What it is | Used for | Terms |
|---|---|---|---|
| [BRUSH](https://github.com/brownvc/decoupled-style-descriptors#brush-dataset) ([Kotani et al., ECCV 2020](https://arxiv.org/abs/2011.07813)) | 27,649 online samples from 170 writers, labelled per character | everyday words from its careful writers; digits and punctuation from all | non-commercial research |
| [IAM-OnDB](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database) (Liwicki & Bunke, ICDAR 2005) | 13,049 whiteboard text lines from 221 writers | everyday words; the writers whose styles documents get | non-commercial research, free registration |
| [MathWriting](https://github.com/google-research/google-research/tree/master/mathwriting) ([Gervais et al., 2024](https://arxiv.org/abs/2404.10690)) | 230k human-written math expressions | the symbol bank (isolated symbols, human inks only) | CC BY-NC-SA 4.0 |

Put them under `data/`, which is never committed:

```
data/brush_zip/BRUSH.zip
data/iamondb/ascii-all.tar.gz, lineStrokes-all.tar.gz
data/mathwriting/mathwriting-2024.tgz
```

## Recipe

From this folder, with PyTorch on a CUDA GPU (one consumer GPU is enough; the whole run takes a few hours):

```bash
# 1. Pack the pen recordings (late i-dots and t-bars move next to their letters)
python extract_brush.py data/brush_zip/BRUSH.zip data/brush
python prep_brush.py data/brush/BRUSH data/cache/brush2.npz
python prep_iam.py data/iamondb data/cache/iam.npz

# 2. IAM-OnDB has text per line, not per letter: train a CTC reader, then force-align
python train_align.py --epochs 14 --out runs/align
python align_iam.py

# 3. Cut lines into baseline-relative words, then choose what the model may learn from
python build_words.py --min-score -0.05
python curate_words.py

# 4. The checker, and the shape limits
python train_checker.py
python shapecheck.py

# 5. The writer (40 epochs from scratch; checkpoints scored by writing unseen words)
python train_words.py

# 6. Judge it against the installed model, then install it and undo the sampling shrink
python gate_models.py runs/hand/best.pt
python install_model.py --run runs/hand --checker runs/checker/best.pt
python calibrate_size.py

# 7. The symbol bank
python prep_symbols.py data/mathwriting/mathwriting-2024.tgz
python build_symbol_bank.py
```

Writers and words held out of training are listed in `writers.json` and `wordsdata.py`. Every score is reported on writers *and* words the model never saw, so memorised vocabulary doesn't count. `defects.py` measures what a reader of the page would complain about: dry-pen breaks, missing i-dots, squished or stacked letters, stray marks. Those rates are compared with real writing, which has some of each.

## What we learned

- **Learn only from real handwriting.** Training on repaired or synthetic words teaches the repairs. Fine-tuning on the model's own words that passed every check scored well and looked less human.
- **Curate the writers.** Many BRUSH writers leave most i's undotted, and the model copied them faithfully. Their everyday words are left out, but they still teach the characters the rest of the data is short of.
- **Losses don't pick the model.** Heavier regularisation gave better losses and blended the writers into one average hand. A model ships only if `gate_models.py` finds no defect rate worse than the installed model's, *and* it wins a blind side-by-side comparison by eye.
- **Result:** in a blind test of three-line blocks, generated handwriting was judged real as often as real handwriting was.

The trained weights inherit the datasets' non-commercial terms, so they are a separate download and not part of this repository.
