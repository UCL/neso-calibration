"""Functions for extracting and processing outputs from NESO models."""

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np


def read_hdf5_datasets(
    hdf5_file_path: Path,
    dataset_paths: dict[str, str | Sequence[str]],
) -> dict[str, np.ndarray]:
    """Read one or more datasets at specified paths from a HDF5 file.

    Args:
        hdf5_file_path: Path to HDF5 file to read dataset(s) from.
        dataset_paths: Map from string keys to one or more HDF5 paths to datasets within
           the HDF5 file to read. If a key is associated with a single HDF5 path then
           the returned dictionary will contain an entry for the key corresponding to
           the dataset specified by that path as a NumPy array. If a key is associated
           with a sequence of paths, then it is assumed all paths refer to HDF5 datasets
           of the same shape such that they can be stacked in an array with number of
           dimensions one more than each datasets number of dimensions, with the
           returned dictionary then containing an entry for the key corresponding to
           this stacked array, with first axis indexing across the dataset paths.

    Returns:
        Dictionary of read datasets with keys corresponding to keys of `dataset_paths`
        argument, and values either the associated dataset as a NumPy array (if the
        key in `dataset_paths` mapped to a single path) or a the associated stacked
        datasets as a NumPy array (if the key in `dataset_paths` mapped to a sequence
        of paths), with the first axis of the array corresponding to the indexing
        through the sequence of paths associated with the key.
    """
    with h5py.File(hdf5_file_path, "r") as hdf5_file:
        return {
            key: np.asarray(hdf5_file[path_or_paths])
            if isinstance(path_or_paths, str)
            else np.stack([hdf5_file[path] for path in path_or_paths])
            for key, path_or_paths in dataset_paths.items()
        }
