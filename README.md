# Goose Means

A small unsupervised learning experiment implemented to test the [Goose programming language](https://aardappel.github.io/goose/).
It compares Lloyd's $k$-means method with the exponential moving average (EMA) approach from the recently published [e-means paper](https://arxiv.org/abs/2611.09501).

## Usage

### Goose Installation

Download the Goose archive for your system from the [v2026.4 release](https://github.com/aardappel/goose/releases/tag/v2026.4).
From the repo root, create `.tools/` and save the archive there.

### Linux

Requires `glibc 2.35` or newer and C library headers (`libc6-dev` on Debian or Ubuntu, `glibc-devel` on Fedora).

```bash
tar -xzf .tools/goose-v2026.4-linux-x64.tar.gz -C .tools
export GOOSE_HOME="$PWD/.tools/goose-home"
./.tools/goose-v2026.4-linux-x64/goose main.goose
```

### Windows

```powershell
Expand-Archive -LiteralPath '.tools\goose-v2026.4-windows-x64.zip' -DestinationPath '.tools'
$env:GOOSE_HOME = Join-Path $PWD '.tools\goose-home'
& '.\.tools\goose-v2026.4-windows-x64\goose.exe' main.goose
```

## Comparison Description

We use a fixed seed `42` to generate `1,000 points` in five overlapping clouds, and reuse that dataset for all `20` runs.
Each run picks `5` points as starting centroids and gives the same positions to both methods.

- **Lloyd's method** assigns all `1,000` points to their nearest centroids, then moves each centroid to the mean of its assigned points. Empty clusters keep their previous centroids.
- **EMA method** shuffles the points before each pass and updates the centroids after every `50` points. Using a fixed step size, it updates an EMA of each cluster's point sum and count, then divides the sum by the count.

The code doesn't implement the paper's automatic parameter selection, step-size schedule, cluster revival, and finalization.

## Results

The program reports median mean squared error (MSE) across `20` runs before training and after each epoch.
After `10` epochs, median MSE was `10.6833` for Lloyd and `10.4291` for EMA method. The EMA variant achieved lower error in **15 of 20** runs.
