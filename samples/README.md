# Sample files

## Camera raws (`raws/`)

Real public-domain camera raws from [raw.pixls.us](https://raw.pixls.us) (CC0 where marked on that archive). Useful for testing ingest → develop → print without your own files on hand.

Fetch / refresh:

```bash
bash samples/fetch_raws.sh
```

| Local file | Camera | Source path on raw.pixls.us |
|------------|--------|-----------------------------|
| `nikon_d40_DSC_1842.NEF` | Nikon D40 | `Nikon/D40/DSC_1842.NEF` |
| `nikon_d90_00001.NEF` | Nikon D90 | `Nikon/D90/00001.NEF` |
| `canon_40d_MG_0153.CR2` | Canon EOS 40D | `Canon/EOS 40D/_MG_0153.CR2` |
| `canon_550d_IMG_4047.CR2` | Canon EOS 550D | `Canon/EOS 550D/IMG_4047.CR2` |
| `sony_a6000_DSC01542.ARW` | Sony α6000 | `Sony/ILCE-6000/DSC01542.ARW` |

Try one:

```bash
python scripts/run_spike.py samples/raws/nikon_d40_DSC_1842.NEF --film hp5-plus-v1
python scripts/run_darkroom_ui.py
# then upload a file from samples/raws/
```

These binaries are gitignored (large). Re-run the fetch script after clone.
