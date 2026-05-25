#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_flame.sh — Download FLAME 2020 and place it in all required locations.
#
# Prerequisites
# ------------
# 1. Register a free account at https://flame.is.tue.mpg.de/
#    (accepts the FLAME license on that site)
# 2. Run:   bash setup_flame.sh
# ---------------------------------------------------------------------------
set -e

DECA_DATA="/mnt/d/GitHub/cv_project/third_party/DECA/data"
FLAME_PYT="/mnt/d/GitHub/cv_project/Aura3D/third_party/flame_pytorch/data/FLAME2020"
TMP_ZIP="/tmp/FLAME2020.zip"
TMP_DIR="/tmp/FLAME2020_extracted"

# ---- collect credentials -------------------------------------------------
echo ""
echo "Register at https://flame.is.tue.mpg.de/ if you have not already."
read -rp "FLAME username (email): " USERNAME
read -rsp "FLAME password: "         PASSWORD
echo ""

urlencode() {
    python3 -c "import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1]))" "$1"
}
U=$(urlencode "$USERNAME")
P=$(urlencode "$PASSWORD")

# ---- download FLAME 2020 -------------------------------------------------
echo "Downloading FLAME2020.zip ..."
wget --post-data "username=${U}&password=${P}" \
     'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' \
     -O "$TMP_ZIP" --no-check-certificate --continue

echo "Extracting ..."
mkdir -p "$TMP_DIR"
unzip -o "$TMP_ZIP" -d "$TMP_DIR"

# Locate generic_model.pkl inside the zip (some versions nest it)
PKL=$(find "$TMP_DIR" -name "generic_model.pkl" | head -1)
if [[ -z "$PKL" ]]; then
    echo "ERROR: generic_model.pkl not found inside FLAME2020.zip" >&2
    exit 1
fi
echo "Found: $PKL"

# ---- copy to both locations -----------------------------------------------
echo "Placing in DECA data dir ..."
cp "$PKL" "$DECA_DATA/generic_model.pkl"

echo "Placing in flame_pytorch data dir ..."
mkdir -p "$FLAME_PYT"
cp "$PKL" "$FLAME_PYT/generic_model.pkl"

# Also copy flame_static_embedding.pkl if present (activates eye vertex indices)
STATIC_EMB=$(find "$TMP_DIR" -name "flame_static_embedding.pkl" | head -1)
if [[ -n "$STATIC_EMB" ]]; then
    echo "Copying flame_static_embedding.pkl ..."
    cp "$STATIC_EMB" "$FLAME_PYT/flame_static_embedding.pkl"
fi

# ---- done -----------------------------------------------------------------
echo ""
echo "FLAME 2020 setup complete."
echo "  $DECA_DATA/generic_model.pkl"
echo "  $FLAME_PYT/generic_model.pkl"
[[ -n "$STATIC_EMB" ]] && echo "  $FLAME_PYT/flame_static_embedding.pkl"
echo ""
echo "You can now run FLAME tracking with:"
echo "  bash run_flame_tracking.sh"
