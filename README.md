[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10_%7C_3.11-blue?logo=python&logoColor=white)](https://docs.python.org/3/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
![Tests](https://github.com/luisherrmann/tsconcat_private/workflows/Tests/badge.svg?branch=master)
![Build](https://github.com/luisherrmann/tsconcat_private/workflows/Build/badge.svg?branch=master)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=luisherrmann_tsconcat_private&metric=coverage&token=0d40140507fe7315b266b70b845b1abe3de69f8e)](https://sonarcloud.io/summary/new_code?id=luisherrmann_tsconcat_private)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=luisherrmann_tsconcat_private&metric=reliability_rating&token=0d40140507fe7315b266b70b845b1abe3de69f8e)](https://sonarcloud.io/summary/new_code?id=luisherrmann_tsconcat_private)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# tsconcat
---

This is a neat little tool that supports the concatenation of `tensorstore` datasets using the `n5`/`zarr` driver along a given axis to obtain concatenated datasets, without having to write a new dataset which would imply doubling the storage used on disk or using a python `for`-loop to iterate over all datasets.

**NOTE**: This tool works under the assumption your **individual datasets are homogeneous**, i.e. they all use the same block structure, datatype, and compression scheme.

## Installation
You can install the package from PyPI

`pip install tsconcat`

or directly from GitHub by running

`pip install git+https://github.com/luisherrmann/tsconcat`

## How it works
---
The tool exploits the file hierarchy of `n5`/ `zarr` driver datasets to create concatenated datasets. Suppose you have two tensorstore datasets `ds1`. `ds2` with shape `[2, 2, 2]` and `[2, 3, 2]`, respectively. Furthermore, let's say the datasets use a block size of `[1, 2, 1]`. Then, the respective nested file hierarchies would be:

### *ds1*
```bash
.
├── metadata
├── 0
│   └── 0
│       ├── 0
│       └── 1
└── 1
    └── 0
        ├── 0
        └── 1
```

and

### *ds2*
```bash
.
├── metadata
├── 0
│   ├── 0
│   │   ├── 0
│   │   └── 1
│   └── 1
│       ├── 0
│       └── 1
└── 1
    ├── 0
    │   ├── 0
    │   └── 1
    └── 1
        ├── 0
        └── 1
```

, where directory nesting level 0 corresponds to dimension 0, nesting level 1 corresponds to dimension 1 and so on.

Let's say we want to **concatenate along dimension 1** to obtain a dataset `ds12`. By assumption of homogeneity (same block size), the file hierarchy for a concatenated dataset must look exactly **the same on all nesting levels above the concatenation level**. So, all we have to do is link all directories from nesting level 1 into common directories. For the above example:

1. `ds1/0/0 -> ds12/0/0`
2. `ds2/0/0 -> ds12/0/1`
3. `ds2/0/1 -> ds12/0/2`
4. `ds1/1/0 -> ds12/1/0`
5. `ds2/1/0 -> ds12/1/1`
6. `ds2/1/1 -> ds12/1/1`

This leading to the following file hierarchy in the concatenated dataset:

### *ds12*
```bash
.
├── metadata
├── 0
│   ├── 0 <---- 'ds1/0/0'
│   │   ├── 0
│   │   └── 1
│   ├── 1 <---- 'ds2/0/0'
│   │   ├── 0
│   │   └── 1
│   └── 2 <---- 'ds2/0/1'
│       ├── 0
│       └── 1
└── 1
    ├── 0 <---- 'ds1/1/0'
    │   ├── 0
    │   └── 1
    ├── 1 <---- 'ds2/1/0'
    │   ├── 0
    │   └── 1
    └── 2 <---- 'ds2/1/1'
        ├── 0
        └── 1
```

The metadata object of `*ds12*` is written to match the concatenated dataset.

**NOTE**: The `n5`and `zarr` driver specifications require all blocks to be the same size (except for termination blocks). The specification is respected in the above example where all blocks have size `[1, 2, 1]`, except for termination blocks `0/2/1` and `1/2/1` at a block size of `[1, 1, 1]`. But what if we change the concatenation order and build a dataset `ds21` from concatenating `ds2` with `ds1`?

In that case, all blocks would have size `[1, 2, 1]` except for termination blocks`0/1/1` and `1/1/1`. But these are NOT termination blocks! However, we can pretend they are blocks of the correct size `[1, 2, 1]` and mask out the excess.

The `zarr` driver specification additionally allows for flat directory structures using `.` as a dimension separator. E.g. `ds1` in the above example would look like this:
### *ds1*
```bash
. .. metadata  0.0.0  0.0.1  1.0.0  1.0.1
```
Since the number of blocks and thus of files to be linked does not change, scenarios with flat or nested directory structures are equivalent with the exception of the dimension separator.

## Usage
You can perform the concatenation by running
```bash
tsconcat <CONCAT_PATH> <TS_PATH1> <TS_PATH2> ... <CAT_DIM> [-d <DRIVER>] [-s <DIMSEP>] [-p]
```
, where `CONCAT_PATH` is the path of the target directory where to concatenate the datasets, and `TS_PATH1`, ... are the paths to the tensorstore datasets to be concatenated. `CAT_DIM` is the dimension along which to concatenate. As optional arguments, you can provide the `<DRIVER>` used in your tensorstores (`'n5'` or `'zarr'`) and the dimension separator `<DIMSEP>` to use (where `'.'` is only supported for the zarr driver). Using the `-p` flag enables a progress bar.

You can read from the datasets in python as follows:
```python
from tsconcat import ConcatDataset
ds = ConcatDataset(concat_path)
data = ds[:].read().result()
```
