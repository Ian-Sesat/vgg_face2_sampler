# vggfaces-sampler

Extracts and samples 500,000 images uniformly from VGG Faces v2 across all identities. Deletes tar.gz files and temp extraction folder after sampling to save disk space.

## Usage

Set `VGG_DIR` to the dataset path and run:

```bash
python extract_vggfaces.py
```

## Output

```
vggfaces_sampled/
    n000001/   ← ~54 images per identity
    n000002/
    ...
```

Sampling uses zone-based selection to guarantee coverage across all image variations per identity.

## Requirements

```bash
pip install tqdm
```
