"""
extract_vggfaces.py
Extracts VGG Faces v2 tar.gz files and samples 500,000 images
uniformly across all identities.
Deletes tar.gz files after extraction to save disk space.
"""

import os
import random
import shutil
import tarfile
from pathlib import Path
from tqdm import tqdm

BASE_DIR     = '/your/path'
VGG_DIR      = os.path.join(BASE_DIR, 'VGG-Face2', 'data')
EXTRACT_DIR  = os.path.join(BASE_DIR, 'vggfaces_sampled')
TARGET_TOTAL = 500000
RANDOM_SEED  = 42
IMAGE_EXTS   = {'.jpg', '.jpeg', '.png'}

random.seed(RANDOM_SEED)
os.makedirs(EXTRACT_DIR, exist_ok=True)

TRAIN_TAR = os.path.join(VGG_DIR, 'vggface2_train.tar.gz')
TEST_TAR  = os.path.join(VGG_DIR, 'vggface2_test.tar.gz')
TEMP_DIR  = os.path.join(BASE_DIR, 'vggfaces_temp')
os.makedirs(TEMP_DIR, exist_ok=True)


def extract_tar(tar_path, extract_to):
    print(f"\nExtracting : {os.path.basename(tar_path)}")
    print(f"This may take a while (~37GB for train)...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        members = tar.getmembers()
        for m in tqdm(members, desc="Extracting"):
            tar.extract(m, extract_to)
    print(f"Extracted → {extract_to}")
    os.remove(tar_path)
    print(f"Deleted   → {tar_path} ")


# extract train tar
if os.path.exists(TRAIN_TAR):
    extract_tar(TRAIN_TAR, TEMP_DIR)
else:
    print(f"Train tar not found: {TRAIN_TAR}")

# extract test tar
if os.path.exists(TEST_TAR):
    extract_tar(TEST_TAR, TEMP_DIR)
else:
    print(f"Test tar not found: {TEST_TAR}")

# find extracted root
vgg_root = TEMP_DIR
for sub in ['train', 'data/train']:
    candidate = os.path.join(TEMP_DIR, sub)
    if os.path.exists(candidate):
        vgg_root = candidate
        break

print(f"\nVGG Faces root : {vgg_root}")

# discover all identity folders
identities = sorted([
    d for d in os.listdir(vgg_root)
    if os.path.isdir(os.path.join(vgg_root, d))
])
print(f"Total identities : {len(identities):,}")

# count images per identity
print("Counting images per identity...")
identity_images = {}
for identity in tqdm(identities, desc="Scanning"):
    identity_path = os.path.join(vgg_root, identity)
    images = [
        os.path.join(identity_path, f)
        for f in os.listdir(identity_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]
    if images:
        identity_images[identity] = images

total_images = sum(len(imgs) for imgs in identity_images.values())
print(f"Total images available : {total_images:,}")
print(f"Target                 : {TARGET_TOTAL:,}")

# compute images per identity (uniform sampling)
num_identities    = len(identity_images)
base_per_identity = TARGET_TOTAL // num_identities
remainder         = TARGET_TOTAL  % num_identities

print(f"Identities with images : {num_identities:,}")
print(f"Base per identity      : {base_per_identity}")

# shuffle identities for random remainder assignment
identity_list = list(identity_images.keys())
random.shuffle(identity_list)

def zone_select(images, n):
    """Pick n images uniformly across sorted list using zone-based sampling."""
    images    = sorted(images)
    total     = len(images)
    n         = min(n, total)
    zone_size = total / n
    return [
        random.choice(images[int(z * zone_size):int((z + 1) * zone_size)])
        for z in range(n)
    ]


# sample and copy
print(f"\nSampling {TARGET_TOTAL:,} images uniformly...")
total_extracted = 0

for i, identity in enumerate(tqdm(identity_list, desc="Sampling")):
    images = identity_images[identity]
    n      = base_per_identity + (1 if i < remainder else 0)
    n      = min(n, len(images))

    sampled  = zone_select(images, n)
    dest_dir = os.path.join(EXTRACT_DIR, identity)
    os.makedirs(dest_dir, exist_ok=True)

    for src in sampled:
        shutil.copy2(src, os.path.join(dest_dir, os.path.basename(src)))

    total_extracted += n

# delete temp extracted folder to save space
print(f"\nDeleting temp folder {TEMP_DIR}...")
shutil.rmtree(TEMP_DIR)
print("Deleted ")

print(f"\nDone!")
print(f"Total sampled  : {total_extracted:,}")
print(f"Saved to       : {EXTRACT_DIR}")

identity_counts = [
    len(list(Path(os.path.join(EXTRACT_DIR, d)).glob('*')))
    for d in os.listdir(EXTRACT_DIR)
    if os.path.isdir(os.path.join(EXTRACT_DIR, d))
]
print(f"Identities     : {len(identity_counts):,}")
print(f"Min per identity: {min(identity_counts)}")
print(f"Max per identity: {max(identity_counts)}")
print(f"Avg per identity: {sum(identity_counts)/len(identity_counts):.1f}")
