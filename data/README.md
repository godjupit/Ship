# Dataset Installation

The public SuperShip dataset is distributed separately through Tsinghua Cloud.

From the repository root, either run:

```bash
SUPERSHIP_DATA_URL='<THU_CLOUD_DIRECT_DOWNLOAD_URL>' bash download_data.sh
```

or install an archive already downloaded from the cloud drive:

```bash
bash download_data.sh /path/to/SuperShip-public-data.tar.zst
```

The archive extracts into `data/SuperShip/`. Verify the installation with:

```bash
python3 supership/validate_release.py
```
