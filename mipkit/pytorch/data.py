from abc import ABC, abstractmethod
from pandas import DataFrame
import os
import pandas as pd
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Union
from torch.utils.data import Dataset as _TorchDataset
import sys
import threading
import warnings
from tqdm import tqdm
from multiprocessing.pool import ThreadPool


def apply_transform(transform: Callable, data, map_items: bool = True):
    """
    Transform `data` with `transform`.
    If `data` is a list or tuple and `map_data` is True, each item of `data` will be transformed
    and this method returns a list of outcomes.
    otherwise transform will be applied once with `data` as the argument.
    Args:
        transform: a callable to be used to transform `data`
        data: an object to be transformed.
        map_items: whether to apply transform to each item in `data`,
            if `data` is a list or tuple. Defaults to True.
    Raises:
        Exception: When ``transform`` raises an exception.
    """
    try:
        if isinstance(data, (list, tuple)) and map_items:
            return [transform(item) for item in data]
        return transform(data)
    except Exception as e:
        raise type(e)(f"Applying transform {transform}.").with_traceback(
            e.__traceback__)


class DataReader():
    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class CSVReader(DataReader):
    def __init__(self, csv_file):
        if not os.path.isfile(csv_file):
            raise FileNotFoundError('Cannot find file {}'.format(csv_file))
        self.data = pd.read_csv(csv_file)

    def iloc(self, row_idx):
        columns = self.data.columns
        data_dict = {}
        [data_dict.update({col: self.data[col][row_idx]}) for col in columns]
        return data_dict

    def __getitem__(self, index: int):
        return self.iloc(index)

    def __len__(self) -> int:
        return len(self.data)


class Randomizable(ABC):
    def __call__(self, data):
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement this method.")


class Transform(ABC):
    def __call__(self, data):
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement this method.")


class Dataset(_TorchDataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    For example, typical input data can be a list of dictionaries::
        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(self, data: DataReader, transform: Optional[Callable] = None) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.
        """

        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        data = self.data[index]
        if self.transform is not None:
            data = apply_transform(self.transform, data)
        return data


class CacheDataset(Dataset):
    """
    Dataset with cache mechanism that can load data and cache deterministic transforms' result during training.
    By caching the results of non-random preprocessing transforms, it accelerates the training data pipeline.
    If the requested data is not in the cache, all transforms will run normally
    (see also :py:class:`monai.data.dataset.Dataset`).
    Users can set the cache rate or number of items to cache.
    It is recommended to experiment with different `cache_num` or `cache_rate` to identify the best training speed.
    To improve the caching efficiency, please always put as many as possible non-random transforms
    before the randomized ones when composing the chain of transforms.
    when `transforms` is used in a multi-epoch training pipeline, before the first training epoch,
    this dataset will cache the results up to ``ScaleIntensityRanged``, as
    all non-random transforms `LoadNiftid`, `AddChanneld`, `Spacingd`, `Orientationd`, `ScaleIntensityRanged`
    can be cached. During training, the dataset will load the cached results and run
    ``RandCropByPosNegLabeld`` and ``ToTensord``, as ``RandCropByPosNegLabeld`` is a randomized transform
    and the outcome not cached.
    """

    def __init__(
        self,
        data: Sequence,
        transform: Union[Sequence[Callable], Callable],
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int = 0,
    ) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: transforms to execute operations on input data.
            cache_num: number of items to be cached. Default is `sys.maxsize`.
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            cache_rate: percentage of cached data in total, default is 1.0 (cache all).
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            num_workers: the number of worker threads to use.
                If 0 a single thread will be used. Default is 0.
        """
        super().__init__(data, transform)
        self.cache_num = min(cache_num, int(len(data) * cache_rate), len(data))
        if self.cache_num > 0:
            self._cache = [None] * self.cache_num
            pbar = tqdm(total=self.cache_num,
                        desc="Load and cache transformed data")

            if num_workers > 0:
                self._item_processed = 0
                self._thread_lock = threading.Lock()
                with ThreadPool(num_workers) as p:
                    p.map(
                        self._load_cache_item_thread,
                        [(i, data[i], transform.transforms, pbar)
                         for i in range(self.cache_num)],
                    )
            else:
                for i in range(self.cache_num):
                    self._cache[i] = self._load_cache_item(
                        data[i], transform.transforms)
                    if pbar is not None:
                        pbar.update(1)
            if pbar is not None:
                pbar.close()

    def _load_cache_item(self, item: Any, transforms: Sequence[Callable]):
        """
        Args:
            item: input item to load and transform to generate dataset for model.
            transforms: transforms to execute operations on input item.
        """
        for _transform in transforms:
            # execute all the deterministic transforms
            if isinstance(_transform, Randomizable) or not isinstance(_transform, Transform):
                break
            item = apply_transform(_transform, item)
        return item

    def _load_cache_item_thread(self, args: Any) -> None:
        """
        Args:
            args: tuple with contents (i, item, transforms, pbar).
                i: the index to load the cached item to.
                item: input item to load and transform to generate dataset for model.
                transforms: transforms to execute operations on input item.
                pbar: tqdm progress bar
        """
        i, item, transforms, pbar = args
        self._cache[i] = self._load_cache_item(item, transforms)
        if pbar is not None:
            with self._thread_lock:
                pbar.update(1)

    def __getitem__(self, index):
        if index < self.cache_num:
            # load data from cache and execute from the first random transform
            start_run = False
            data = self._cache[index]
            for _transform in self.transform.transforms:  # pytype: disable=attribute-error
                if not start_run and not isinstance(_transform, Randomizable) and isinstance(_transform, Transform):
                    continue
                else:
                    start_run = True
                data = apply_transform(_transform, data)
        else:
            # no cache for this data, execute all the transforms directly
            data = super().__getitem__(index)
        return data
