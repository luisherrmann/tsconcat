import os
from os.path import basename, dirname
import glob
import json
import tqdm
import argparse
from typing import Any, Dict, List, Tuple, Iterable
import itertools as it
import numpy as np
import tensorstore as ts

from .utils import get_ts_read_config

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("concat_path", type=str, help="Path to concatenated tensorstore")
parser.add_argument(
    "paths", type=str, nargs="+", help="Paths to tensorstores to concatenate"
)
parser.add_argument(
    "catdim", type=int, default=0, help="Dimension to concatenate along"
)
parser.add_argument(
    "-d",
    "--driver",
    type=str,
    default="n5",
    help="The driver of the tensorstores. Either 'n5' or 'zarr'.",
)
parser.add_argument(
    "-s",
    "--dimsep",
    type=str,
    default=".",
    help="The dimension separator to use in the concatenated dataset. Either '.' or '/'.",
)
parser.add_argument("-p", "--progress", action="store_true", help="Show progress bar")


def zarr_format_match(zarr_format1: str, zarr_format2: str):
    assert (
        zarr_format1 == zarr_format2
    ), f"Mismatch in zarr_format: {zarr_format1} != {zarr_format2}"


def blockSize_match(block_size1: Iterable[int], block_size2: Iterable[int]):
    block_size1 = np.array([*block_size1])
    block_size2 = np.array([*block_size2])
    len1, len2 = len(block_size1), len(block_size2)
    assert len1 == len2, f"Mismatch in blockSize dimensionalities: {len1} != {len2}"
    assert np.all(
        block_size1 == block_size2
    ), f"Mismatch in blockSizes: {block_size1} != {block_size2}"


def compression_match(compression1: Dict[str, Any], compression2: Dict[str, Any]):
    assert set(compression1.keys()) == set(compression2.keys())
    values_nmatch = [compression1[k] != compression2[k] for k in compression1.keys()]
    if any(values_nmatch):
        items = zip(compression1.keys(), compression1.values(), compression2.values())
        items_nmatch = list(it.compress(items, values_nmatch))
        mismatch_str = ", ".join([f"{k}: {v1} != {v2}" for k, v1, v2 in items_nmatch])
        mismatch_str += "." if len(items_nmatch) < 10 else ", ..."
        raise AssertionError(f"Mismatch in compression parameters: {mismatch_str}")


def dataType_match(data_type1: str, data_type2: str):
    assert (
        data_type1 == data_type2
    ), f"Mismatch in dataTypes: {data_type1} != {data_type2}"


def dimensions_match(
    dimensions1: Iterable[int], dimensions2: Iterable[int], catdim: int
):
    dimensions1 = [*dimensions1]
    dimensions2 = [*dimensions2]
    dimensions1.pop(catdim)
    dimensions2.pop(catdim)
    dimensions1 = np.array([*dimensions1])
    dimensions2 = np.array([*dimensions2])
    len1, len2 = len(dimensions1), len(dimensions2)
    assert len1 == len2, f"Mismatch in dimensionalities: {len1} != {len2}"
    assert np.all(
        dimensions1 == dimensions2
    ), f"Mismatch in non-concatenation dimensions: {dimensions1} != {dimensions2}"


def reraise_err(path1: str, path2: str, msg: str):
    err_msg = f"Compatibility error for paths \n{path1} and \n{path2} \n{msg}"
    raise AssertionError(err_msg)


def check_n5_metadata(paths, metadata, catdim):
    for p in paths[1:]:
        try:
            blockSize_match(metadata[paths[0]]["blockSize"], metadata[p]["blockSize"])
        except AssertionError as e:
            reraise_err(paths[0], p, str(e))
        try:
            compression_match(
                metadata[paths[0]]["compression"], metadata[p]["compression"]
            )
        except AssertionError as e:
            reraise_err(paths[0], p, str(e))
        try:
            dataType_match(metadata[paths[0]]["dataType"], metadata[p]["dataType"])
        except AssertionError as e:
            reraise_err(paths[0], p, str(e))
        try:
            dimensions_match(
                metadata[paths[0]]["dimensions"], metadata[p]["dimensions"], catdim
            )
        except AssertionError as e:
            reraise_err(paths[0], p, str(e))
    path0_ndim = len(metadata[paths[0]]["dimensions"])
    assert catdim < path0_ndim, ValueError(f"catdim must be < {path0_ndim}")


def check_zarr_metadata(paths, metadata, catdim):
    for p in paths[1:]:
        try:
            zarr_format_match(
                metadata[paths[0]]["zarr_format"], metadata[p]["zarr_format"]
            )
        except AssertionError as e:
            reraise_err(paths[0], p, str(e))
        try:
            blockSize_match(metadata[paths[0]]["chunks"], metadata[p]["chunks"])
        except AssertionError as e:
            reraise_err(paths[0], p, str(e))
        try:
            compression_match(
                metadata[paths[0]]["compressor"], metadata[p]["compressor"]
            )
        except AssertionError as e:
            reraise_err(paths[0], p, str(e))
        try:
            dataType_match(metadata[paths[0]]["dtype"], metadata[p]["dtype"])
        except AssertionError as e:
            reraise_err(paths[0], p, str(e))
        try:
            dimensions_match(metadata[paths[0]]["shape"], metadata[p]["shape"], catdim)
        except AssertionError as e:
            reraise_err(paths[0], p, str(e))
    path0_ndim = len(metadata[paths[0]]["shape"])
    assert catdim < path0_ndim, ValueError(f"catdim must be < {path0_ndim}")


def check_stores(paths: Iterable[str], catdim: int, driver: str) -> Dict[str, Any]:
    # Check that all paths provided exist
    paths = [*paths]
    if len(paths) < 2:
        raise ValueError("You must provide at least 2 paths to concatenate!")
    paths_nexist = [not (os.path.exists(p)) for p in paths]
    if any(paths_nexist):
        invalid_paths = it.compress(paths, paths_nexist)
        invalid_str = ", ".join(invalid_paths)
        invalid_str += "." if len(invalid_paths) < 10 else ", ..."
        raise ValueError(f"The following paths do not exist: {invalid_str}")
    # Check that all paths are valid tensorstores
    for p in paths:
        try:
            cfg = get_ts_read_config(p, driver)
            store = ts.open(cfg, read=True)
        except:
            raise RuntimeError(f"Failed to open tensorstore: {p}")
    # Load metadata for all tensorstores
    metadata = {}
    for p in paths:
        if driver == "n5":
            metadata_path = os.path.join(p, "attributes.json")
        elif driver == "zarr":
            metadata_path = os.path.join(p, ".zarray")
        with open(metadata_path, "r") as f:
            metadata[p] = json.load(f)
    if driver == "n5":
        check_n5_metadata(paths, metadata, catdim)
    elif driver == "zarr":
        check_zarr_metadata(paths, metadata, catdim)
    else:
        raise ValueError(f"Driver '{driver}' is not one of 'n5' | 'zarr'!")
    return metadata


def write_n5_metadata(concat_path: str, paths: List[str], catdim: int):
    # Load metadata for all tensorstores
    metadata = {}
    for p in paths:
        with open(os.path.join(p, "attributes.json"), "r") as f:
            metadata[p] = json.load(f)
    # Write metadata for concatenated tensorstore
    concat_metadata = metadata[paths[0]]
    # Dimensions of the individual tensorstores along the concatenation dimension,
    # assuming all blocks except for the last are full or zero-padded.
    def _catdim_ceil(dim: int, block_size: int):
        return int(np.ceil(dim / block_size)) * block_size

    padded_catlens = [
        _catdim_ceil(
            metadata[p]["dimensions"][catdim], metadata[p]["blockSize"][catdim]
        )
        for p in paths[:-1]
    ]
    padded_catlens += [metadata[paths[-1]]["dimensions"][catdim]]
    virtual_catlens = [metadata[p]["dimensions"][catdim] for p in paths]
    # Add custom metadata for later usage by the interface
    concat_metadata["dimensions"][catdim] = sum(padded_catlens)

    concat_metadata["custom"] = {
        "catdim": catdim,
        "padded_catlens": padded_catlens,
        "virtual_catlens": virtual_catlens,
    }
    with open(os.path.join(concat_path, "attributes.json"), "w") as f:
        json.dump(concat_metadata, f)


def write_zarr_metadata(concat_path: str, paths: List[str], catdim: int, dimsep: str):
    # Load metadata for all tensorstores
    metadata = {}
    for p in paths:
        with open(os.path.join(p, ".zarray"), "r") as f:
            metadata[p] = json.load(f)
    # Write metadata for concatenated tensorstore
    concat_metadata = metadata[paths[0]]
    # Dimensions of the individual tensorstores along the concatenation dimension,
    # assuming all blocks except for the last are full or zero-padded.
    def _catdim_ceil(dim: int, block_size: int):
        return int(np.ceil(dim / block_size)) * block_size

    padded_catlens = [
        _catdim_ceil(metadata[p]["shape"][catdim], metadata[p]["chunks"][catdim])
        for p in paths[:-1]
    ]
    padded_catlens += [metadata[paths[-1]]["shape"][catdim]]
    virtual_catlens = [metadata[p]["shape"][catdim] for p in paths]
    # Add custom metadata for later usage by the interface
    concat_metadata["shape"][catdim] = sum(padded_catlens)
    concat_metadata["dimension_separator"] = dimsep

    concat_metadata["custom"] = {
        "catdim": catdim,
        "padded_catlens": padded_catlens,
        "virtual_catlens": virtual_catlens,
    }
    with open(os.path.join(concat_path, ".zarray"), "w") as f:
        json.dump(concat_metadata, f)


def list_paths(path: str, prefix: Tuple[str], sep: str = "/"):
    path_base = os.path.join(path, sep.join(prefix))
    if sep == "/":
        paths = [p for p in os.listdir(path_base) if p.isnumeric()]
    elif sep == ".":
        level = len(prefix)
        sep = "." if level > 0 else ""
        pattern = f"{path_base}{sep}[0-9]*"
        paths = set(basename(p).split(".")[level] for p in glob.glob(pattern))
    else:
        raise ValueError(f"Separator '{sep}' is not one of '/' | '.'!")
    return paths


def _tsconcat(
    concat_path: str,
    paths: List[str],
    dimensions: Iterable[Tuple[int]],
    block_sizes: Iterable[Tuple[int]],
    catdim: int,
    src_dimseps: List[str],
    tgt_dimsep: str,
    level: int = 0,
    prefix: Tuple[str] = (),
    progress=None,
):
    if level == catdim:
        # Symlink all blocks in the concatenation dimension
        block_cnt = 0
        for p, src_sep, p_dims, p_bsizes in zip(
            paths, src_dimseps, dimensions, block_sizes
        ):
            if src_sep == "/":
                # Go down path branch to the level of the concatenation dimension
                p_branch = os.path.join(p, src_sep.join(prefix))
                if tgt_dimsep == ".":
                    for path, dirs, files in os.walk(p_branch):
                        blocks = [fn for fn in files if fn.isnumeric()]
                        for b in blocks:
                            pb_src = os.path.join(path, b)
                            pb = pb_src[len(p) + 1 :].split("/")
                            pb[level] = str(int(pb[level]) + block_cnt)
                            pb_tgt = os.path.join(concat_path, tgt_dimsep.join(pb))
                            os.symlink(pb_src, pb_tgt, target_is_directory=False)
                            if progress is not None:
                                progress.update(1)

                elif tgt_dimsep == "/":
                    blocks = list_paths(p, prefix, sep=src_sep)
                    for b in blocks:
                        pb_src = p_branch + (src_sep if level > 0 else "") + b
                        pb_linkname = str(int(b) + block_cnt)
                        pb_tgt = tgt_dimsep.join(
                            (concat_path,) + prefix + (pb_linkname,)
                        )
                        source_is_directory = os.path.isdir(pb_src)
                        os.symlink(
                            pb_src, pb_tgt, target_is_directory=source_is_directory
                        )
                        if progress is not None:
                            progress.update(1)

            elif src_sep == ".":
                p_branch = os.path.join(p, src_sep.join(prefix))
                sep = "." if level > 0 else ""
                pattern = f"{p_branch}{sep}[0-9]*"
                blocks = [basename(p).split(".") for p in glob.glob(pattern)]
                # chunks_at_catdim = np.unique([b[level] for b in blocks])
                for b in blocks:
                    pb_src = os.path.join(p, src_sep.join(b))
                    b[level] = str(int(b[level]) + block_cnt)
                    pb_linkname = tgt_dimsep.join(b)
                    pb_tgt = os.path.join(concat_path, pb_linkname)
                    if tgt_dimsep == "/":
                        os.makedirs(dirname(pb_tgt), exist_ok=True)
                    os.symlink(pb_src, pb_tgt, target_is_directory=False)
                    if progress is not None:
                        progress.update(1)

            block_cnt += int(np.ceil(p_dims[catdim] / p_bsizes[catdim]))

    else:
        # Create new directories for each block and do recursion step
        blocks = list_paths(paths[0], prefix, sep=src_dimseps[0])
        for b in blocks:
            prefix_ = prefix + (b,)
            dir_path = tgt_dimsep.join((concat_path,) + prefix_)
            os.mkdir(dir_path)
            _tsconcat(
                concat_path,
                paths,
                dimensions,
                block_sizes,
                catdim,
                src_dimseps,
                tgt_dimsep,
                level=level + 1,
                prefix=prefix_,
                progress=progress,
            )


def get_link_cnt(
    dimensions: Tuple[int],
    block_size: Tuple[int],
    catdim: int,
    src_dimsep: str,
    tgt_dimsep: str,
):
    assert len(dimensions) == len(block_size)
    assert src_dimsep in ["/", "."], ValueError(f"src_dimsep must be one of '/' | '.'!")
    assert tgt_dimsep in ["/", "."], ValueError(f"tgt_dimsep must be one of '/' | '.'!")

    def _blocks_pp(dimensions, block_size, catdim):
        return int(
            np.prod([np.ceil(dimensions[d] / block_size[d]) for d in range(catdim + 1)])
        )

    if tgt_dimsep == ".":
        # All source blocks need to be linked
        link_cnt = _blocks_pp(dimensions, block_size, len(dimensions) - 1)
    elif tgt_dimsep == "/":
        if src_dimsep == ".":
            link_cnt = _blocks_pp(dimensions, block_size, len(dimensions) - 1)
        elif src_dimsep == "/":
            link_cnt = _blocks_pp(dimensions, block_size, catdim)
    return link_cnt


def get_total_link_cnt(
    dimensions: Iterable[Tuple[int]],
    block_sizes: Iterable[Tuple[int]],
    catdim: int,
    src_dimseps: Iterable[str],
    tgt_dimsep: str,
):
    return np.sum(
        [
            get_link_cnt(dim, bsize, catdim, src_dimsep, tgt_dimsep)
            for dim, bsize, src_dimsep in zip(dimensions, block_sizes, src_dimseps)
        ]
    )


def tsconcat(
    concat_path: str,
    paths: Iterable[str],
    catdim: int = 0,
    driver: str = "n5",
    dimsep: str = "/",
    progress: bool = False,
) -> None:
    """
    Concatenate a list of paths into a single TensorStore.

    Parameters
    ----------
    concat_path : str
        The path to concatenate into.
    paths : Iterable[str]
        A list of paths to concatenate.
    catdim : int, optional
        The dimension to concatenate along, by default 0.
    driver : str, optional
        The driver used on the stores to be concatenated, by default "n5".
    dimsep : str, optional
        The dimension separator to use in the concatenated dataset. Either '.' or '/',
        by default '/'.
        NOTE: Option '.' is only supported for 'zarr' driver.
    progress : bool, optional
        If True, show a progress bar, by default False.

    Returns
    -------
    None
    """
    if driver not in ["n5", "zarr"]:
        raise ValueError(f"Driver '{driver}' is not one of 'n5' | 'zarr'!")
    if dimsep not in ["/", "."]:
        raise ValueError(f"Dimension separator '{dimsep}' is not one of '/' | '.'!")
    if dimsep == "." and driver != "zarr":
        raise ValueError(
            f"Dimension separator '{dimsep}' is only supported for 'zarr' driver!"
        )

    metadata = check_stores(paths, catdim, driver)

    if driver == "n5":
        dimensions = [m["dimensions"] for m in metadata.values()]
        block_sizes = [m["blockSize"] for m in metadata.values()]
        src_dimseps = ["/"] * len(paths)
    elif driver == "zarr":
        dimensions = [m["shape"] for m in metadata.values()]
        block_sizes = [m["chunks"] for m in metadata.values()]
        src_dimseps = [m["dimension_separator"] for m in metadata.values()]

    if os.path.exists(concat_path):
        if len(os.listdir(concat_path)) > 0:
            raise RuntimeError(f"Path {concat_path} already exists!")
    else:
        os.mkdir(concat_path)
    total_link_cnt = get_total_link_cnt(
        dimensions, block_sizes, catdim, src_dimseps, dimsep
    )
    progress = tqdm.tqdm(total=total_link_cnt) if progress else None
    _tsconcat(
        concat_path,
        paths,
        dimensions,
        block_sizes,
        catdim,
        src_dimseps=src_dimseps,
        tgt_dimsep=dimsep,
        progress=progress,
    )
    if driver == "n5":
        write_n5_metadata(concat_path, paths, catdim)
    elif driver == "zarr":
        write_zarr_metadata(concat_path, paths, catdim, dimsep)


class ConcatDataset(object):
    def __init__(self, path: str, driver: str = "n5", dimsep: str = ".", mode="r"):
        """A TensorStore dataset created with the tsconcat tool.

        Parameters
        ----------
        path : str
            The path to the concatenated dataset.
        driver : str, optional
            The driver to use for opening the dataset. Default is "n5".
        dimsep : str, optional
            The separator used for dimensions in the concatenated dataset. Default is ".".
        mode : str, optional
            The mode to open the dataset in. Default is "r". Either "r" or "w".
        """
        self.path = path
        write_mode = mode == "w"
        self.ts_dataset = ts.open(
            get_ts_read_config(path, driver), read=True, write=write_mode
        ).result()
        if driver == "n5":
            metadata_path = os.path.join(path, "attributes.json")
        elif driver == "zarr":
            metadata_path = os.path.join(path, ".zarray")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        assert "custom" in self.metadata, "No metadata for concat dataset found!"
        assert (
            "catdim" in self.metadata["custom"]
        ), "No metadata for concat dataset found!"
        self.catdim = self.metadata["custom"]["catdim"]
        assert (
            "virtual_catlens" in self.metadata["custom"]
        ), "No metadata for concat dataset found!"
        self.virtual_catlens = self.metadata["custom"]["virtual_catlens"]
        self.total_virtual_catlen = sum(self.virtual_catlens)
        assert (
            "padded_catlens" in self.metadata["custom"]
        ), "No metadata for concat dataset found!"
        self.padded_catlens = self.metadata["custom"]["padded_catlens"]
        self.total_padded_catlen = sum(self.padded_catlens)
        self.mask = np.zeros(shape=sum(self.padded_catlens), dtype=bool)
        # Create mask for virtual concatenation dimension
        low = 0
        for plen, vlen in zip(self.padded_catlens, self.virtual_catlens):
            high = low + vlen
            self.mask[low:high] = 1
            low += plen

    def remap_index(self, index):

        return ConcatDataset._remap_index(
            index, self.total_virtual_catlen, self.mask, self.catdim
        )

    @staticmethod
    def _remap_index(index, masksize, padded_mask, catdim):
        if isinstance(index, Tuple):
            index = [*index]
            if len(index) > catdim:
                # Replace selection along concatenation dimension with joint mask
                mask = np.zeros(masksize, dtype=bool)
                index_catdim = index[catdim]
                mask[index_catdim] = 1
                index_ = index[:catdim] + [mask]
                if len(index) > catdim:
                    index_ += index[catdim + 1 :]
            else:
                index_ = index + [slice(None)] * (catdim + 1 - len(index))
        else:
            # Slice along 0-th dimension
            if catdim == 0:
                mask = np.zeros(masksize, dtype=bool)
                mask[index] = 1
                index_ = [mask]
            else:
                index_ = [index] + [slice(None)] * catdim

        # Transform catdim into higher-dim padded mask
        index_catdim = index_[catdim]
        if isinstance(index_catdim, slice) and index_catdim == slice(None):
            index_[catdim] = padded_mask
        else:
            index_padded = np.zeros(len(padded_mask), dtype=bool)
            index_padded[padded_mask] = index_[catdim]
            index_[catdim] = index_padded
        index_ = tuple(index_)
        return index_

    def __getitem__(self, index):
        index_ = self.remap_index(index)
        return self.ts_dataset.__getitem__(index_)

    def __setitem__(self, index, source):
        index_ = self.remap_index(index)
        return self.ts_dataset.__setitem__(index_, source)


if __name__ == "__main__":
    args = parser.parse_args()
    print("\n".join(f"{k}={v}" for k, v in vars(args).items()))
    args_dict = vars(args)
    tsconcat(**args_dict)
