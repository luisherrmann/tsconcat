import pytest
import itertools as it
import numpy as np
from tempfile import TemporaryFile, TemporaryDirectory
from typing import Tuple, List
from addict import Dict
import tensorstore as ts
import os
import json

from tsconcat.tsconcat import ConcatDataset, get_link_cnt, tsconcat


def get_ts_write_n5config(
    dimensions: Tuple[int],
    block_size: Tuple[int],
    path: str = ".",
    cname: str = "lz4",
    clevel: int = 9,
    dtype: str = "uint8",
    **kwargs,
) -> Dict:
    assert len(dimensions) == len(block_size)
    config = Dict(
        {
            "driver": "n5",
            "kvstore": {
                "driver": "file",
                "path": path,
            },
            "metadata": {
                "compression": {
                    "type": "blosc",
                    "cname": cname,
                    "clevel": clevel,
                    "shuffle": 0,
                },
                "dataType": dtype,
                "dimensions": dimensions,
                "blockSize": block_size,
            },
            "create": True,
            "delete_existing": True,
        }
    )
    return config


def get_ts_write_zarrconfig(
    shape: Tuple[int],
    chunks: Tuple[int],
    zarr_format: int = 2,
    path: str = ".",
    cname: str = "lz4",
    clevel: int = 9,
    dimsep: str = "/",
    dtype: str = "<u1",
    **kwargs,
) -> Dict:
    assert len(shape) == len(chunks)
    config = Dict(
        {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": path,
            },
            "metadata": {
                "zarr_format": zarr_format,
                "compressor": {
                    "id": "blosc",
                    "cname": cname,
                    "clevel": clevel,
                    "shuffle": 0,
                },
                "dimension_separator": dimsep,
                "dtype": dtype,
                "shape": shape,
                "chunks": chunks,
            },
            "create": True,
            "delete_existing": True,
        }
    )
    return config


@pytest.fixture
def temp_file():
    with TemporaryFile() as temp_file:
        yield temp_file


@pytest.fixture
def temp_dir():
    with TemporaryDirectory() as temp_dir:
        yield temp_dir


def create_ts(config, shape):
    data = np.random.randint(255, size=shape, dtype=np.uint8)
    ds = ts.open(config).result()
    ds.write(data).result()
    return data, ds


TSCONCAT_TEST_CASES = [
    (
        [
            get_ts_write_n5config(
                **{
                    "dimensions": (1,),
                    "block_size": (1,),
                }
            ),
            get_ts_write_n5config(
                **{
                    "dimensions": (1,),
                    "block_size": (1,),
                }
            ),
        ],
        0,
        "n5",
        "/",
        {"0": "A/0", "1": "B/0"},
        {
            "dimensions": [2],
            "custom": {
                "catdim": 0,
                "padded_catlens": [1, 1],
                "virtual_catlens": [1, 1],
            },
        },
        None,
    ),
    (
        [
            get_ts_write_zarrconfig(
                **{
                    "shape": (1,),
                    "chunks": (1,),
                }
            ),
            get_ts_write_zarrconfig(
                **{
                    "shape": (1,),
                    "chunks": (1,),
                }
            ),
        ],
        0,
        "zarr",
        "/",
        {"0": "A/0", "1": "B/0"},
        {
            "shape": [2],
            "dimension_separator": "/",
            "custom": {
                "catdim": 0,
                "padded_catlens": [1, 1],
                "virtual_catlens": [1, 1],
            },
        },
        None,
    ),
    (
        [
            get_ts_write_n5config(
                **{
                    "dimensions": (1, 1),
                    "block_size": (1, 1),
                }
            ),
            get_ts_write_n5config(
                **{
                    "dimensions": (1, 1),
                    "block_size": (1, 1),
                }
            ),
        ],
        0,
        "n5",
        "/",
        {"0": "A/0", "1": "B/0"},
        {
            "dimensions": [2, 1],
            "custom": {
                "catdim": 0,
                "padded_catlens": [1, 1],
                "virtual_catlens": [1, 1],
            },
        },
        None,
    ),
    *[
        (
            [
                get_ts_write_zarrconfig(
                    **{"shape": (1, 1), "chunks": (1, 1), "dimsep": src_dimsep0}
                ),
                get_ts_write_zarrconfig(
                    **{"shape": (1, 1), "chunks": (1, 1), "dimsep": src_dimsep1}
                ),
            ],
            0,
            "zarr",
            tgt_dimsep,
            {
                **{
                    True: {"0": "A/0"},
                    False: {
                        tgt_dimsep.join(["0", "0"]): src_dimsep0.join(["A", "0", "0"])
                    },
                }[(src_dimsep0, tgt_dimsep) == ("/", "/")],
                **{
                    True: {"1": "B/0"},
                    False: {
                        tgt_dimsep.join(["1", "0"]): src_dimsep1.join(["B", "0", "0"])
                    },
                }[(src_dimsep1, tgt_dimsep) == ("/", "/")],
            },
            {
                "shape": [2, 1],
                "dimension_separator": tgt_dimsep,
                "custom": {
                    "catdim": 0,
                    "padded_catlens": [1, 1],
                    "virtual_catlens": [1, 1],
                },
            },
            None,
        )
        for (src_dimsep0, src_dimsep1, tgt_dimsep) in it.product(*[("/", ".")] * 3)
    ],
    (
        [
            get_ts_write_n5config(
                **{
                    "dimensions": (1, 1),
                    "block_size": (1, 1),
                }
            ),
            get_ts_write_n5config(
                **{
                    "dimensions": (1, 1),
                    "block_size": (1, 1),
                }
            ),
        ],
        1,
        "n5",
        "/",
        {"0/0": "A/0/0", "0/1": "B/0/0"},
        {
            "dimensions": [1, 2],
            "custom": {
                "catdim": 1,
                "padded_catlens": [1, 1],
                "virtual_catlens": [1, 1],
            },
        },
        None,
    ),
    *[
        (
            [
                get_ts_write_zarrconfig(
                    **{"shape": (1, 1), "chunks": (1, 1), "dimsep": src_dimsep0}
                ),
                get_ts_write_zarrconfig(
                    **{"shape": (1, 1), "chunks": (1, 1), "dimsep": src_dimsep1}
                ),
            ],
            1,
            "zarr",
            tgt_dimsep,
            {
                tgt_dimsep.join(["0", "0"]): src_dimsep0.join(["A", "0", "0"]),
                tgt_dimsep.join(["0", "1"]): src_dimsep1.join(["B", "0", "0"]),
            },
            {
                "shape": [1, 2],
                "dimension_separator": tgt_dimsep,
                "custom": {
                    "catdim": 1,
                    "padded_catlens": [1, 1],
                    "virtual_catlens": [1, 1],
                },
            },
            None,
        )
        for (src_dimsep0, src_dimsep1, tgt_dimsep) in it.product(*[("/", ".")] * 3)
    ],
    (
        [
            get_ts_write_n5config(
                **{
                    "dimensions": (2, 1),
                    "block_size": (1, 1),
                }
            ),
            get_ts_write_n5config(
                **{
                    "dimensions": (3, 1),
                    "block_size": (1, 1),
                }
            ),
        ],
        0,
        "n5",
        "/",
        {"0": "A/0", "1": "A/1", "2": "B/0", "3": "B/1", "4": "B/2"},
        {
            "dimensions": [5, 1],
            "custom": {
                "catdim": 0,
                "padded_catlens": [2, 3],
                "virtual_catlens": [2, 3],
            },
        },
        None,
    ),
    *[
        (
            [
                get_ts_write_zarrconfig(
                    **{
                        "shape": (2, 1),
                        "chunks": (1, 1),
                        "dimsep": src_dimsep0,
                    }
                ),
                get_ts_write_zarrconfig(
                    **{"shape": (3, 1), "chunks": (1, 1), "dimsep": src_dimsep1}
                ),
            ],
            0,
            "zarr",
            tgt_dimsep,
            {
                **{
                    True: {
                        "0": "A/0",
                        "1": "A/1",
                    },
                    False: {
                        tgt_dimsep.join(["0", "0"]): src_dimsep0.join(["A", "0", "0"]),
                        tgt_dimsep.join(["1", "0"]): src_dimsep0.join(["A", "1", "0"]),
                    },
                }[(src_dimsep0, tgt_dimsep) == ("/", "/")],
                **{
                    True: {"2": "B/0", "3": "B/1", "4": "B/2"},
                    False: {
                        tgt_dimsep.join(["2", "0"]): src_dimsep1.join(["B", "0", "0"]),
                        tgt_dimsep.join(["3", "0"]): src_dimsep1.join(["B", "1", "0"]),
                        tgt_dimsep.join(["4", "0"]): src_dimsep1.join(["B", "2", "0"]),
                    },
                }[(src_dimsep1, tgt_dimsep) == ("/", "/")],
            },
            {
                "shape": [5, 1],
                "dimension_separator": tgt_dimsep,
                "custom": {
                    "catdim": 0,
                    "padded_catlens": [2, 3],
                    "virtual_catlens": [2, 3],
                },
            },
            None,
        )
        for (src_dimsep0, src_dimsep1, tgt_dimsep) in it.product(*[("/", ".")] * 3)
    ],
    (
        [
            get_ts_write_n5config(
                **{
                    "dimensions": (1, 3),
                    "block_size": (1, 1),
                }
            ),
            get_ts_write_n5config(
                **{
                    "dimensions": (1, 2),
                    "block_size": (1, 1),
                }
            ),
        ],
        1,
        "n5",
        "/",
        {
            "0/0": "A/0/0",
            "0/1": "A/0/1",
            "0/2": "A/0/2",
            "0/3": "B/0/0",
            "0/4": "B/0/1",
        },
        {
            "dimensions": [1, 5],
            "custom": {
                "catdim": 1,
                "padded_catlens": [3, 2],
                "virtual_catlens": [3, 2],
            },
        },
        None,
    ),
    *[
        (
            [
                get_ts_write_zarrconfig(
                    **{"shape": (1, 3), "chunks": (1, 1), "dimsep": src_dimsep0}
                ),
                get_ts_write_zarrconfig(
                    **{"shape": (1, 2), "chunks": (1, 1), "dimsep": src_dimsep1}
                ),
            ],
            1,
            "zarr",
            tgt_dimsep,
            {
                tgt_dimsep.join(["0", "0"]): src_dimsep0.join(["A", "0", "0"]),
                tgt_dimsep.join(["0", "1"]): src_dimsep0.join(["A", "0", "1"]),
                tgt_dimsep.join(["0", "2"]): src_dimsep0.join(["A", "0", "2"]),
                tgt_dimsep.join(["0", "3"]): src_dimsep1.join(["B", "0", "0"]),
                tgt_dimsep.join(["0", "4"]): src_dimsep1.join(["B", "0", "1"]),
            },
            {
                "shape": [1, 5],
                "dimension_separator": tgt_dimsep,
                "custom": {
                    "catdim": 1,
                    "padded_catlens": [3, 2],
                    "virtual_catlens": [3, 2],
                },
            },
            None,
        )
        for (src_dimsep0, src_dimsep1, tgt_dimsep) in it.product(*[("/", ".")] * 3)
    ],
    (
        [
            get_ts_write_n5config(
                **{
                    "dimensions": (1, 3),
                    "block_size": (1, 2),
                }
            ),
            get_ts_write_n5config(
                **{
                    "dimensions": (1, 4),
                    "block_size": (1, 2),
                }
            ),
        ],
        1,
        "n5",
        "/",
        {
            "0/0": "A/0/0",
            "0/1": "A/0/1",
            "0/2": "B/0/0",
            "0/3": "B/0/1",
        },
        {
            "dimensions": [1, 8],
            "custom": {
                "catdim": 1,
                "padded_catlens": [4, 4],
                "virtual_catlens": [3, 4],
            },
        },
        None,
    ),
    (
        [
            get_ts_write_zarrconfig(
                **{
                    "shape": (1, 3),
                    "chunks": (1, 2),
                }
            ),
            get_ts_write_zarrconfig(
                **{
                    "shape": (1, 4),
                    "chunks": (1, 2),
                }
            ),
        ],
        1,
        "zarr",
        "/",
        {
            "0/0": "A/0/0",
            "0/1": "A/0/1",
            "0/2": "B/0/0",
            "0/3": "B/0/1",
        },
        {
            "shape": [1, 8],
            "dimension_separator": "/",
            "custom": {
                "catdim": 1,
                "padded_catlens": [4, 4],
                "virtual_catlens": [3, 4],
            },
        },
        None,
    ),
    (
        [
            get_ts_write_n5config(
                **{
                    "dimensions": (2, 3, 4),
                    "block_size": (1, 1, 2),
                }
            ),
            get_ts_write_n5config(
                **{
                    "dimensions": (2, 3, 5),
                    "block_size": (1, 1, 2),
                }
            ),
            get_ts_write_n5config(
                **{
                    "dimensions": (2, 3, 3),
                    "block_size": (1, 1, 2),
                }
            ),
        ],
        2,
        "n5",
        "/",
        {
            **{
                f"{i}/{j}/{k}": f"A/{i}/{j}/{k}"
                for i, j, k in it.product(range(2), range(3), range(2))
            },
            **{
                f"{i}/{j}/{k + 2}": f"B/{i}/{j}/{k}"
                for i, j, k in it.product(range(2), range(3), range(3))
            },
            **{
                f"{i}/{j}/{k + 5}": f"C/{i}/{j}/{k}"
                for i, j, k in it.product(range(2), range(3), range(2))
            },
        },
        {
            "dimensions": [2, 3, 13],
            "custom": {
                "catdim": 2,
                "padded_catlens": [4, 6, 3],
                "virtual_catlens": [4, 5, 3],
            },
        },
        None,
    ),
    *[
        (
            [
                get_ts_write_zarrconfig(
                    **{
                        "shape": (2, 3, 4),
                        "chunks": (1, 1, 2),
                        "dimsep": src_dimsep0,
                    }
                ),
                get_ts_write_zarrconfig(
                    **{
                        "shape": (2, 3, 5),
                        "chunks": (1, 1, 2),
                        "dimsep": src_dimsep1,
                    }
                ),
                get_ts_write_zarrconfig(
                    **{
                        "shape": (2, 3, 3),
                        "chunks": (1, 1, 2),
                        "dimsep": src_dimsep2,
                    }
                ),
            ],
            2,
            "zarr",
            tgt_dimsep,
            {
                **{
                    tgt_dimsep.join([f"{i}", f"{j}", f"{k}"]): src_dimsep0.join(
                        ["A", f"{i}", f"{j}", f"{k}"]
                    )
                    for i, j, k in it.product(range(2), range(3), range(2))
                },
                **{
                    tgt_dimsep.join([f"{i}", f"{j}", f"{k + 2}"]): src_dimsep1.join(
                        ["B", f"{i}", f"{j}", f"{k}"]
                    )
                    for i, j, k in it.product(range(2), range(3), range(3))
                },
                **{
                    tgt_dimsep.join([f"{i}", f"{j}", f"{k + 5}"]): src_dimsep2.join(
                        ["C", f"{i}", f"{j}", f"{k}"]
                    )
                    for i, j, k in it.product(range(2), range(3), range(2))
                },
            },
            {
                "shape": [2, 3, 13],
                "dimension_separator": tgt_dimsep,
                "custom": {
                    "catdim": 2,
                    "padded_catlens": [4, 6, 3],
                    "virtual_catlens": [4, 5, 3],
                },
            },
            None,
        )
        for (src_dimsep0, src_dimsep1, src_dimsep2, tgt_dimsep) in it.product(
            *[("/", ".")] * 4
        )
    ],
    (
        [
            get_ts_write_n5config(
                **{
                    "dimensions": (3, 5, 4),
                    "block_size": (2, 2, 2),
                }
            ),
            get_ts_write_n5config(
                **{
                    "dimensions": (3, 4, 4),
                    "block_size": (2, 2, 2),
                }
            ),
            get_ts_write_n5config(
                **{
                    "dimensions": (3, 3, 4),
                    "block_size": (2, 2, 2),
                }
            ),
        ],
        1,
        "n5",
        "/",
        {
            **{f"{i}/{j}": f"A/{i}/{j}" for i, j in it.product(range(2), range(3))},
            **{f"{i}/{j + 3}": f"B/{i}/{j}" for i, j in it.product(range(2), range(2))},
            **{f"{i}/{j + 5}": f"C/{i}/{j}" for i, j in it.product(range(2), range(2))},
        },
        {
            "dimensions": [3, 13, 4],
            "custom": {
                "catdim": 1,
                "padded_catlens": [6, 4, 3],
                "virtual_catlens": [5, 4, 3],
            },
        },
        None,
    ),
    *[
        (
            [
                get_ts_write_zarrconfig(
                    **{"shape": (3, 5, 4), "chunks": (2, 2, 2), "dimsep": src_dimsep0}
                ),
                get_ts_write_zarrconfig(
                    **{"shape": (3, 3, 4), "chunks": (2, 2, 2), "dimsep": src_dimsep1}
                ),
                get_ts_write_zarrconfig(
                    **{"shape": (3, 3, 4), "chunks": (2, 2, 2), "dimsep": src_dimsep2}
                ),
            ],
            1,
            "zarr",
            tgt_dimsep,
            {
                **{
                    False: {
                        tgt_dimsep.join([f"{i}", f"{j}", f"{k}"]): src_dimsep0.join(
                            ["A", f"{i}", f"{j}", f"{k}"]
                        )
                        for i, j, k in it.product(range(2), range(3), range(2))
                    },
                    True: {
                        f"{i}/{j}": f"A/{i}/{j}"
                        for i, j in it.product(range(2), range(3))
                    },
                }[(src_dimsep0, tgt_dimsep) == ("/", "/")],
                **{
                    False: {
                        tgt_dimsep.join([f"{i}", f"{j + 3}", f"{k}"]): src_dimsep1.join(
                            ["B", f"{i}", f"{j}", f"{k}"]
                        )
                        for i, j, k in it.product(range(2), range(2), range(2))
                    },
                    True: {
                        f"{i}/{j + 3}": f"B/{i}/{j}"
                        for i, j in it.product(range(2), range(2))
                    },
                }[(src_dimsep1, tgt_dimsep) == ("/", "/")],
                **{
                    False: {
                        tgt_dimsep.join([f"{i}", f"{j + 5}", f"{k}"]): src_dimsep2.join(
                            ["C", f"{i}", f"{j}", f"{k}"]
                        )
                        for i, j, k in it.product(range(2), range(2), range(2))
                    },
                    True: {
                        f"{i}/{j + 5}": f"C/{i}/{j}"
                        for i, j in it.product(range(2), range(2))
                    },
                }[(src_dimsep2, tgt_dimsep) == ("/", "/")],
            },
            {
                "shape": [3, 13, 4],
                "dimension_separator": tgt_dimsep,
                "custom": {
                    "catdim": 1,
                    "padded_catlens": [6, 4, 3],
                    "virtual_catlens": [5, 3, 3],
                },
            },
            None,
        )
        for (src_dimsep0, src_dimsep1, src_dimsep2, tgt_dimsep) in it.product(
            *[("/", ".")] * 4
        )
    ],
    (
        [
            get_ts_write_n5config(
                **{
                    "dimensions": (3, 3, 4),
                    "block_size": (2, 2, 2),
                }
            ),
            get_ts_write_n5config(
                **{
                    "dimensions": (4, 3, 4),
                    "block_size": (2, 2, 2),
                }
            ),
            get_ts_write_n5config(
                **{
                    "dimensions": (3, 3, 4),
                    "block_size": (2, 2, 2),
                }
            ),
        ],
        0,
        "n5",
        "/",
        {
            **{f"{i}": f"A/{i}" for i in range(2)},
            **{f"{i + 2}": f"B/{i}" for i in range(2)},
            **{f"{i + 4}": f"C/{i}" for i in range(2)},
        },
        {
            "dimensions": [11, 3, 4],
            "custom": {
                "catdim": 0,
                "padded_catlens": [4, 4, 3],
                "virtual_catlens": [3, 4, 3],
            },
        },
        None,
    ),
    *[
        (
            [
                get_ts_write_zarrconfig(
                    **{"shape": (5, 3, 4), "chunks": (2, 2, 2), "dimsep": src_dimsep0}
                ),
                get_ts_write_zarrconfig(
                    **{"shape": (3, 3, 4), "chunks": (2, 2, 2), "dimsep": src_dimsep1}
                ),
                get_ts_write_zarrconfig(
                    **{"shape": (3, 3, 4), "chunks": (2, 2, 2), "dimsep": src_dimsep2}
                ),
            ],
            0,
            "zarr",
            tgt_dimsep,
            {
                **{
                    False: {
                        tgt_dimsep.join([f"{i}", f"{j}", f"{k}"]): src_dimsep0.join(
                            ["A", f"{i}", f"{j}", f"{k}"]
                        )
                        for i, j, k in it.product(range(3), range(2), range(2))
                    },
                    True: {f"{i}": f"A/{i}" for i in range(3)},
                }[(src_dimsep0, tgt_dimsep) == ("/", "/")],
                **{
                    False: {
                        tgt_dimsep.join([f"{i + 3}", f"{j}", f"{k}"]): src_dimsep1.join(
                            ["B", f"{i}", f"{j}", f"{k}"]
                        )
                        for i, j, k in it.product(range(2), range(2), range(2))
                    },
                    True: {f"{i + 3}": f"B/{i}" for i in range(2)},
                }[(src_dimsep1, tgt_dimsep) == ("/", "/")],
                **{
                    False: {
                        tgt_dimsep.join([f"{i + 5}", f"{j}", f"{k}"]): src_dimsep2.join(
                            ["C", f"{i}", f"{j}", f"{k}"]
                        )
                        for i, j, k in it.product(range(2), range(2), range(2))
                    },
                    True: {f"{i + 5}": f"C/{i}" for i in range(2)},
                }[(src_dimsep2, tgt_dimsep) == ("/", "/")],
            },
            {
                "shape": [13, 3, 4],
                "dimension_separator": tgt_dimsep,
                "custom": {
                    "catdim": 0,
                    "padded_catlens": [6, 4, 3],
                    "virtual_catlens": [5, 3, 3],
                },
            },
            None,
        )
        for (src_dimsep0, src_dimsep1, src_dimsep2, tgt_dimsep) in it.product(
            *[("/", ".")] * 4
        )
    ],
]


@pytest.mark.parametrize(
    "configs, catdim, driver, dimsep, exp_links, exp_metadata, error",
    [
        *TSCONCAT_TEST_CASES,
        # Check for dimension mismatch
        (
            [
                get_ts_write_n5config(
                    **{
                        "dimensions": (1, 3),
                        "block_size": (1, 1),
                    }
                ),
                get_ts_write_n5config(
                    **{
                        "dimensions": (1, 2),
                        "block_size": (1, 1),
                    }
                ),
            ],
            0,
            "n5",
            "/",
            {},
            {},
            AssertionError,
        ),
        (
            [
                get_ts_write_zarrconfig(
                    **{
                        "shape": (1, 3),
                        "chunks": (1, 1),
                    }
                ),
                get_ts_write_zarrconfig(
                    **{
                        "shape": (1, 2),
                        "chunks": (1, 1),
                    }
                ),
            ],
            0,
            "zarr",
            "/",
            {},
            {},
            AssertionError,
        ),
        # Check for block_size mismatch
        (
            [
                get_ts_write_n5config(
                    **{
                        "dimensions": (3, 1),
                        "block_size": (2, 1),
                    }
                ),
                get_ts_write_n5config(
                    **{
                        "dimensions": (2, 1),
                        "block_size": (1, 1),
                    }
                ),
            ],
            0,
            "n5",
            "/",
            {},
            {},
            AssertionError,
        ),
        (
            [
                get_ts_write_zarrconfig(
                    **{
                        "shape": (3, 1),
                        "chunks": (2, 1),
                    }
                ),
                get_ts_write_zarrconfig(
                    **{
                        "shape": (2, 1),
                        "chunks": (1, 1),
                    }
                ),
            ],
            0,
            "zarr",
            "/",
            {},
            {},
            AssertionError,
        ),
        (
            [
                get_ts_write_n5config(
                    **{
                        "dimensions": (3, 2),
                        "block_size": (1, 2),
                    }
                ),
                get_ts_write_n5config(
                    **{
                        "dimensions": (2, 2),
                        "block_size": (1, 1),
                    }
                ),
            ],
            0,
            "n5",
            "/",
            {},
            {},
            AssertionError,
        ),
        (
            [
                get_ts_write_zarrconfig(
                    **{
                        "shape": (3, 2),
                        "chunks": (1, 2),
                    }
                ),
                get_ts_write_zarrconfig(
                    **{
                        "shape": (2, 2),
                        "chunks": (1, 1),
                    }
                ),
            ],
            0,
            "zarr",
            "/",
            {},
            {},
            AssertionError,
        ),
        (
            [
                get_ts_write_n5config(
                    **{
                        "dimensions": (3, 1, 1),
                        "block_size": (1, 1, 1),
                    }
                ),
                get_ts_write_n5config(
                    **{
                        "dimensions": (2, 1),
                        "block_size": (1, 1),
                    }
                ),
            ],
            0,
            "n5",
            "/",
            {},
            {},
            AssertionError,
        ),
        (
            [
                get_ts_write_zarrconfig(
                    **{
                        "shape": (3, 1, 1),
                        "chunks": (1, 1, 1),
                    }
                ),
                get_ts_write_zarrconfig(
                    **{
                        "shape": (2, 1),
                        "chunks": (1, 1),
                    }
                ),
            ],
            0,
            "zarr",
            "/",
            {},
            {},
            AssertionError,
        ),
        # Check for compression mismatch
        (
            [
                get_ts_write_n5config(
                    **{
                        "dimensions": (3, 1),
                        "block_size": (1, 1),
                        "cname": "zlib",
                        "clevel": 9,
                    }
                ),
                get_ts_write_n5config(
                    **{
                        "dimensions": (2, 1),
                        "block_size": (1, 1),
                        "cname": "lz4",
                        "clevel": 9,
                    }
                ),
            ],
            0,
            "n5",
            "/",
            {},
            {},
            AssertionError,
        ),
        (
            [
                get_ts_write_zarrconfig(
                    **{
                        "shape": (3, 1),
                        "chunks": (1, 1),
                        "cname": "zlib",
                        "clevel": 9,
                    }
                ),
                get_ts_write_zarrconfig(
                    **{
                        "shape": (2, 1),
                        "chunks": (1, 1),
                        "cname": "lz4",
                        "clevel": 9,
                    }
                ),
            ],
            0,
            "zarr",
            "/",
            {},
            {},
            AssertionError,
        ),
        (
            [
                get_ts_write_n5config(
                    **{
                        "dimensions": (3, 1),
                        "block_size": (1, 1),
                        "cname": "zlib",
                        "clevel": 9,
                    }
                ),
                get_ts_write_n5config(
                    **{
                        "dimensions": (2, 1),
                        "block_size": (1, 1),
                        "cname": "zlib",
                        "clevel": 5,
                    }
                ),
            ],
            0,
            "n5",
            "/",
            {},
            {},
            AssertionError,
        ),
        (
            [
                get_ts_write_zarrconfig(
                    **{
                        "shape": (3, 1),
                        "chunks": (1, 1),
                        "cname": "zlib",
                        "clevel": 9,
                    }
                ),
                get_ts_write_zarrconfig(
                    **{
                        "shape": (2, 1),
                        "chunks": (1, 1),
                        "cname": "zlib",
                        "clevel": 5,
                    }
                ),
            ],
            0,
            "zarr",
            "/",
            {},
            {},
            AssertionError,
        ),
        # Check for dtype mismatch
        (
            [
                get_ts_write_n5config(
                    **{
                        "dimensions": (3, 1),
                        "block_size": (1, 1),
                        "dtype": "uint8",
                    }
                ),
                get_ts_write_n5config(
                    **{
                        "dimensions": (2, 1),
                        "block_size": (1, 1),
                        "dtype": "float32",
                    }
                ),
            ],
            0,
            "n5",
            "/",
            {},
            {},
            AssertionError,
        ),
        (
            [
                get_ts_write_zarrconfig(
                    **{
                        "shape": (3, 1),
                        "chunks": (1, 1),
                        "dtype": "<u1",
                    }
                ),
                get_ts_write_zarrconfig(
                    **{
                        "shape": (2, 1),
                        "chunks": (1, 1),
                        "dtype": "<f4",
                    }
                ),
            ],
            0,
            "zarr",
            "/",
            {},
            {},
            AssertionError,
        ),
        ([], 2, "n5", ".", {}, {}, ValueError),
        (
            [
                get_ts_write_n5config(
                    **{
                        "dimensions": (2, 3, 4),
                        "block_size": (1, 1, 2),
                    }
                )
            ],
            2,
            "n5",
            ".",
            {},
            {},
            ValueError,
        ),
        (
            [
                get_ts_write_n5config(
                    **{
                        "dimensions": (2, 3, 4),
                        "block_size": (1, 1, 2),
                    }
                ),
                get_ts_write_n5config(
                    **{
                        "dimensions": (2, 3, 5),
                        "block_size": (1, 1, 2),
                    }
                ),
            ],
            2,
            "illegal",
            ".",
            {},
            {},
            ValueError,
        ),
        (
            [
                get_ts_write_n5config(
                    **{
                        "dimensions": (2, 3, 4),
                        "block_size": (1, 1, 2),
                    }
                ),
                get_ts_write_n5config(
                    **{
                        "dimensions": (2, 3, 5),
                        "block_size": (1, 1, 2),
                    }
                ),
            ],
            2,
            "n5",
            ".",
            {},
            {},
            ValueError,
        ),
        *[
            (
                [
                    get_ts_write_zarrconfig(
                        **{
                            "shape": (2, 3, 4),
                            "chunks": (1, 1, 2),
                            "dimsep": "/",
                        }
                    ),
                    get_ts_write_zarrconfig(
                        **{
                            "shape": (2, 3, 5),
                            "chunks": (1, 1, 2),
                            "dimsep": "/",
                        }
                    ),
                ],
                2,
                "zarr",
                tgt_dimsep,
                {},
                {},
                ValueError,
            )
            for tgt_dimsep in ["", ",", ":", "a", "1", "a1", "ab", "a.b"]
        ],
    ],
)
def test_tsconcat(configs, catdim, driver, dimsep, exp_links, exp_metadata, error):
    data_list = []
    store_tmpdirs = [TemporaryDirectory() for cfg in enumerate(configs)]
    temp_dir = TemporaryDirectory()
    store_paths = [p.name for p in store_tmpdirs]
    for p, cfg in zip(store_tmpdirs, configs):
        cfg.kvstore.path = p.name
        if cfg["driver"] == "n5":
            shape = cfg["metadata"]["dimensions"]
        elif cfg["driver"] == "zarr":
            shape = cfg["metadata"]["shape"]
        data, _ = create_ts(cfg, shape)
        data_list.append(data)

    if error is None:
        tsconcat(
            temp_dir.name,
            store_paths,
            catdim,
            driver=driver,
            dimsep=dimsep,
            progress=True,
        )

        # assert links match
        links = find_links(temp_dir.name)

        def replace_letter(path, dimseps):
            letter = path[0]
            idx = ord(letter.lower()) - 97
            root_path = store_paths[idx]
            dimsep = dimseps[idx]
            return path.replace(f"{letter}{dimsep}", f"{root_path}/")

        if driver == "zarr":
            dimseps = [cfg["metadata"]["dimension_separator"] for cfg in configs]
            exp_links = {
                src: replace_letter(tgt, dimseps) for src, tgt in exp_links.items()
            }
        else:
            dimseps = ["/"] * len(configs)
            exp_links = {
                src: replace_letter(tgt, dimseps) for src, tgt in exp_links.items()
            }
        # We use issubset here because tensorstore may decide to ommit some blocks for optimization purposes
        assert set(links.items()).issubset(set(exp_links.items()))

        # assert metadata match
        if driver == "n5":
            metadata_name = "attributes.json"
        elif driver == "zarr":
            metadata_name = ".zarray"
        with open(os.path.join(temp_dir.name, metadata_name)) as f:
            metadata = json.load(f)
        with open(os.path.join(p.name, metadata_name)) as f:
            metadata_orig = json.load(f)
        for key in metadata:
            if key in exp_metadata:
                assert metadata[key] == exp_metadata[key]
            else:
                assert metadata[key] == metadata_orig[key]

    else:
        with pytest.raises(error):
            tsconcat(temp_dir.name, store_paths, catdim, driver=driver, dimsep=dimsep)

    for p in store_tmpdirs:
        p.cleanup()
    temp_dir.cleanup()


class TestConcatDataset:
    @pytest.mark.parametrize(
        "index, padded_mask, catdim, exp_index",
        [
            ([], np.array([1, 0], dtype=bool), 0, (np.array([0, 0], dtype=bool),)),
            (
                slice(None),
                np.array([1, 0], dtype=bool),
                0,
                (np.array([1, 0], dtype=bool),),
            ),
            (
                np.array([0], dtype=int),
                np.array([1, 0], dtype=bool),
                0,
                (np.array([1, 0], dtype=bool),),
            ),
            (
                np.array([0], dtype=bool),
                np.array([1, 0], dtype=bool),
                0,
                (np.array([0, 0], dtype=bool),),
            ),
            (
                np.array([1], dtype=bool),
                np.array([1, 0], dtype=bool),
                0,
                (np.array([1, 0], dtype=bool),),
            ),
            (
                np.array([0], dtype=int),
                np.array([0, 1], dtype=bool),
                0,
                (np.array([0, 1], dtype=bool),),
            ),
            (
                np.array([0], dtype=bool),
                np.array([0, 1], dtype=bool),
                0,
                (np.array([0, 0], dtype=bool),),
            ),
            (
                np.array([1], dtype=bool),
                np.array([0, 1], dtype=bool),
                0,
                (np.array([0, 1], dtype=bool),),
            ),
            (
                np.array([1], dtype=int),
                np.array([1, 1], dtype=bool),
                0,
                (np.array([0, 1], dtype=bool),),
            ),
            (
                np.array([1], dtype=int),
                np.array([1, 1], dtype=bool),
                0,
                (np.array([0, 1], dtype=bool),),
            ),
            # Check case with multiple dimensions
            (
                (slice(None), np.array([0], dtype=int)),
                np.array([1, 0], dtype=bool),
                1,
                (slice(None), np.array([1, 0], dtype=bool)),
            ),
            (
                (slice(None), np.array([0], dtype=int)),
                np.array([1, 0], dtype=bool),
                2,
                (slice(None), np.array([0], dtype=int), np.array([1, 0], dtype=bool)),
            ),
            (
                (slice(None), np.array([0], dtype=int)),
                np.array([1, 0], dtype=bool),
                3,
                (
                    slice(None),
                    np.array([0], dtype=int),
                    slice(None),
                    np.array([1, 0], dtype=bool),
                ),
            ),
            # Check higher-dimensional masks
            (
                np.array([1, 3], dtype=int),
                np.array([1, 1, 0, 1, 1, 1], dtype=bool),
                0,
                (np.array([0, 1, 0, 0, 1, 0], dtype=bool),),
            ),
            (
                np.array([1, 1, 0, 0, 1], dtype=bool),
                np.array([1, 1, 0, 1, 1, 1], dtype=bool),
                0,
                (np.array([1, 1, 0, 0, 0, 1], dtype=bool),),
            ),
        ],
    )
    def test_remap_index(self, index, padded_mask, catdim, exp_index):
        masksize = np.sum(padded_mask)
        index_ = ConcatDataset._remap_index(index, masksize, padded_mask, catdim)
        assert isinstance(index_, tuple)
        assert len(index_) == len(exp_index)
        for elem, exp_elem in zip(index_, exp_index):
            assert type(elem) == type(exp_elem)
            if isinstance(elem, np.ndarray):
                assert np.all(elem == exp_elem)
            else:
                assert elem == exp_elem

    @pytest.mark.parametrize(
        "configs, catdim, driver, dimsep",
        [params[:4] for params in TSCONCAT_TEST_CASES],
    )
    def test__init__(self, configs, catdim, driver, dimsep):
        data_list = []
        store_tmpdirs = [TemporaryDirectory() for cfg in enumerate(configs)]
        temp_dir = TemporaryDirectory()
        store_paths = [p.name for p in store_tmpdirs]
        for p, cfg in zip(store_tmpdirs, configs):
            cfg.kvstore.path = p.name
            if cfg["driver"] == "n5":
                shape = cfg["metadata"]["dimensions"]
            elif cfg["driver"] == "zarr":
                shape = cfg["metadata"]["shape"]
            data, _ = create_ts(cfg, shape)
            data_list.append(data)

        tsconcat(
            temp_dir.name,
            store_paths,
            catdim,
            driver=driver,
            dimsep=dimsep,
            progress=True,
        )
        # TODO: Move this to a separate test
        ds = ConcatDataset(temp_dir.name, driver=driver)

        for p in store_tmpdirs:
            p.cleanup()
        temp_dir.cleanup()

    @pytest.mark.parametrize(
        "configs, catdim, driver, dimsep",
        [params[:4] for params in TSCONCAT_TEST_CASES],
    )
    def test__getitem__(self, configs, catdim, driver, dimsep):
        data_list = []
        store_tmpdirs = [TemporaryDirectory() for cfg in enumerate(configs)]
        temp_dir = TemporaryDirectory()
        store_paths = [p.name for p in store_tmpdirs]
        for p, cfg in zip(store_tmpdirs, configs):
            cfg.kvstore.path = p.name
            if cfg["driver"] == "n5":
                shape = cfg["metadata"]["dimensions"]
            elif cfg["driver"] == "zarr":
                shape = cfg["metadata"]["shape"]
            data, _ = create_ts(cfg, shape)
            data_list.append(data)

        tsconcat(
            temp_dir.name,
            store_paths,
            catdim,
            driver=driver,
            dimsep=dimsep,
            progress=True,
        )
        data_ = np.concatenate(data_list, axis=catdim)
        ds = ConcatDataset(temp_dir.name, driver=driver)
        data = ds[:].read().result()
        assert np.all(data == data_)

        for p in store_tmpdirs:
            p.cleanup()
        temp_dir.cleanup()

    @pytest.mark.parametrize(
        "configs, catdim, driver, dimsep",
        [params[:4] for params in TSCONCAT_TEST_CASES],
    )
    def test__setitem__(self, configs, catdim, driver, dimsep):
        store_tmpdirs = [TemporaryDirectory() for cfg in enumerate(configs)]
        temp_dir = TemporaryDirectory()
        store_paths = [p.name for p in store_tmpdirs]
        for p, cfg in zip(store_tmpdirs, configs):
            cfg.kvstore.path = p.name
            if cfg["driver"] == "n5":
                shape = cfg["metadata"]["dimensions"]
            elif cfg["driver"] == "zarr":
                shape = cfg["metadata"]["shape"]
            data, _ = create_ts(cfg, shape)

        tsconcat(
            temp_dir.name,
            store_paths,
            catdim,
            driver=driver,
            dimsep=dimsep,
            progress=True,
        )
        ds = ConcatDataset(temp_dir.name, driver=driver, mode="w")
        if driver == "n5":
            shape = ds.metadata["dimensions"]
            dtype = ds.metadata["dataType"]
        else:
            shape = ds.metadata["shape"]
            dtype = ds.metadata["dtype"]
        data_ = np.random.randint(low=0, high=256, size=ds[:].shape, dtype=dtype)
        ds[:].write(data_).result()

        ds_ = ConcatDataset(temp_dir.name, driver=driver)
        data = ds_[:].read().result()
        assert np.all(data == data_)

        for p in store_tmpdirs:
            p.cleanup()
        temp_dir.cleanup()


def find_links(root_path: str) -> List[str]:
    """Recursively find all symbolic links in a directory tree and returns a list of paths."""
    import os

    links = {}
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            if os.path.islink(path):
                links[os.path.relpath(path, root_path)] = os.readlink(path)
        for dirname in dirnames:
            path = os.path.join(dirpath, dirname)
            if os.path.islink(path):
                links[os.path.relpath(path, root_path)] = os.readlink(path)
        dirnames[:] = [
            d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))
        ]
    return links


@pytest.mark.parametrize(
    "dimensions, block_size, catdim, src_dimsep, tgt_dimsep, exp_link_cnt",
    [
        *[
            ((1,), (1,), 0, src_dimsep, tgt_dimsep, 1)
            for src_dimsep, tgt_dimsep in it.product(["/", "."], ["/", "."])
        ],
        ((4,), (2,), 0, "/", ".", 2),
        ((5,), (2,), 0, "/", ".", 3),
        *[
            ((4, 4), (2, 2), 0, src_dimsep, tgt_dimsep, exp_result)
            for (src_dimsep, tgt_dimsep), exp_result in zip(
                it.product(["/", "."], ["/", "."]), [2, 4, 4, 4]
            )
        ],
        *[
            ((5, 4), (2, 2), 0, src_dimsep, tgt_dimsep, exp_result)
            for (src_dimsep, tgt_dimsep), exp_result in zip(
                it.product(["/", "."], ["/", "."]), [3, 6, 6, 6]
            )
        ],
        *[
            ((4, 4), (2, 2), 1, src_dimsep, tgt_dimsep, exp_result)
            for (src_dimsep, tgt_dimsep), exp_result in zip(
                it.product(["/", "."], ["/", "."]), [4, 4, 4, 4]
            )
        ],
        *[
            ((4, 5), (2, 2), 1, src_dimsep, tgt_dimsep, exp_result)
            for (src_dimsep, tgt_dimsep), exp_result in zip(
                it.product(["/", "."], ["/", "."]), [6, 6, 6, 6]
            )
        ],
    ],
)
def test_get_link_cnt(
    dimensions, block_size, catdim, src_dimsep, tgt_dimsep, exp_link_cnt
):
    assert (
        get_link_cnt(dimensions, block_size, catdim, src_dimsep, tgt_dimsep)
        == exp_link_cnt
    )
