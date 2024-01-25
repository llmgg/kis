import gc
from typing import (
    Dict,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
)
from datasets.download import DownloadConfig, DownloadMode
from datasets.features import Features
from datasets.splits import Split
from datasets.utils import Version, VerificationMode
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from kis.utils.log import kis_logger as logger


class KisDataSet(object):
    def __init__(
            self,
            path: Optional[str] = None,
            name: Optional[str] = None,
            data_dir: Optional[str] = None,
            data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
            split: Optional[Union[str, Split]] = None,
            cache_dir: Optional[str] = None,
            features: Optional[Features] = None,
            download_config: Optional[DownloadConfig] = None,
            download_mode: Optional[Union[DownloadMode, str]] = None,
            verification_mode: Optional[Union[VerificationMode, str]] = None,
            ignore_verifications="deprecated",
            keep_in_memory: Optional[bool] = None,
            save_infos: bool = False,
            revision: Optional[Union[str, Version]] = None,
            token: Optional[Union[bool, str]] = None,
            use_auth_token="deprecated",
            task="deprecated",
            streaming: bool = False,
            num_proc: Optional[int] = None,
            storage_options: Optional[Dict] = None,
            **kwargs,
    ):
        self.path = path
        self.name = name
        self.data_dir = data_dir
        self.data_files = data_files
        self.split = split
        self.cache_dir = cache_dir
        self.features = features
        self.download_config = download_config
        self.download_mode = download_mode
        self.verification_mode = verification_mode
        self.ignore_verifications = ignore_verifications
        self.keep_in_memory = keep_in_memory
        self.save_infos = save_infos
        self.revision = revision
        self.token = token
        self.use_auth_token = use_auth_token
        self.task = task
        self.streaming = streaming
        self.num_proc = num_proc
        self.storage_options = storage_options

        # New defined parameters
        self.kwargs = kwargs
        self.dataset = None
        self.dataloader = None
        self._split_iterator = None
        self._split_iterated = None
        self._batch_size = 1
        self._num_samples_read = 0
        self._epoch_split_iterated = 0
        self._num_split_rows = None
        self._num_batch_read = 0
        self._log_step = 1000

    def set_dataset(self, new_dataset):
        del self.dataset
        gc.collect()
        logger.info("Set the dataset by user.")
        self.dataset = new_dataset

    def load_local_json_files(self, data_dir=None, data_files=None, split_iterated=None):
        self._load_local_files("json", data_dir, data_files, split_iterated)

    def load_local_parquet_files(self, data_dir=None, data_files=None, split_iterated=None):
        self._load_local_files("parquet", data_dir, data_files, split_iterated)

    def _load_local_files(self, data_format, data_dir=None, data_files=None, split_iterated=None):
        self.path = data_format
        self.data_dir = data_dir
        self.data_files = data_files
        self._load_datasets()
        if split_iterated is not None:
            self._set_split_iterator(split_iterated)

    def split_to_dataloader(
            self, split_name, batch_size=1, shuffle=None, sampler=None,
            batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0,
            worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2,
            persistent_workers=False, pin_memory_device=""
    ):
        logger.info(f"Note: create the dataloader will double the memory of the dataset.")
        self._set_split_iterator(split_name)
        self.dataloader = DataLoader(
            [s for s in self._split_iterator], batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn, multiprocessing_context=multiprocessing_context,
            generator=generator, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device
        )

    def split_to_generator(self, split_name, batch_size=1, batch_num=-1, cols=None, log_step=1000):
        self._batch_size = batch_size
        self._log_step = log_step
        self._num_batch_read = 0
        self._set_split_iterator(split_name)

        if self.streaming is False and batch_num <= 0:
            batch_num = self._num_split_rows // self._batch_size
        cols = [col.strip() for col in cols.split(",")] if isinstance(cols, str) else cols

        if isinstance(cols, list):
            def get_cols(batch):
                return ["\n\n".join(s[col] for col in cols) for s in batch]

            if batch_num > 0:
                for _ in tqdm(range(batch_num)):
                    yield get_cols(self._next_batch())
            else:
                epoch_split_iterated_now = self._epoch_split_iterated
                while epoch_split_iterated_now == self._epoch_split_iterated:
                    yield get_cols(self._next_batch())
        else:
            if batch_num > 0:
                for _ in tqdm(range(batch_num)):
                    yield self._next_batch()
            else:
                epoch_split_iterated_now = self._epoch_split_iterated
                while epoch_split_iterated_now == self._epoch_split_iterated:
                    yield self._next_batch()
        logger.info(f"Total number of batch read: {self._num_batch_read}")

    def _set_split_iterator(self, split_iterated):
        self._epoch_split_iterated += 1
        logger.info(f"Epoch No: {self._epoch_split_iterated}")
        if split_iterated != self._split_iterated:
            logger.info(f"Set the iterator with split: '{split_iterated}'")
        self._split_iterated = split_iterated
        logger.info(f"split '{self._split_iterated}' has been set as iterator.")
        self._split_iterator = iter(self.dataset[self._split_iterated])
        if self.streaming is False:
            self._num_split_rows = self.dataset.num_rows[self._split_iterated]
            logger.info(f"Number of samples in the iterator: {self._num_split_rows}")

    def shuffle(self, **kwargs):
        dataset_shuffled = None
        if isinstance(self.dataset, DatasetDict):
            dataset_shuffled = self.dataset.shuffle(
                seeds=kwargs.get("seeds", None),
                seed=kwargs.get("seed", None),
                generators=kwargs.get("generators", None),
                keep_in_memory=kwargs.get("keep_in_memory", False),
                load_from_cache_file=kwargs.get("load_from_cache_file", None),
                indices_cache_file_names=kwargs.get("indices_cache_file_names", None),
                writer_batch_size=kwargs.get("writer_batch_size", 1000)
            )
        elif isinstance(self.dataset, IterableDatasetDict):
            dataset_shuffled = self.dataset.shuffle(
                seed=kwargs.get("seed", None),
                generator=kwargs.get("generator", None),
                buffer_size=kwargs.get("buffer_size", 1000)
            )
        del self.dataset
        gc.collect()
        self.dataset = dataset_shuffled

    def _next_batch(self):
        self._num_batch_read += 1
        if self._num_batch_read % self._log_step == 0:
            logger.info(f"Number of batch read: {self._num_batch_read}")
        return [self._next_sample() for _ in range(self._batch_size)]

    def _next_sample(self):
        self._num_samples_read += 1
        try:
            return next(self._split_iterator)
        except StopIteration:
            logger.warning("Reach the end of the iterator. Start a new iteration.")
            logger.info(f"Reset the iterator with split: '{self._split_iterated}'")
            self.restart_iterator()
            return next(self._split_iterator)

    def restart_iterator(self):
        self._set_split_iterator(self._split_iterated)

    @property
    def split_iterated(self):
        return self._split_iterated

    @property
    def num_batch_read(self):
        return self._num_batch_read

    @property
    def num_samples_read(self):
        return self._num_samples_read

    @property
    def epoch_split_iterated(self):
        return self._epoch_split_iterated

    @property
    def num_split_rows(self):
        return self._num_split_rows

    def _load_datasets(self):
        del self.dataset
        gc.collect()
        logger.info("args info used to load dataset:")
        logger.info("*******************************************************************")
        self.print_args_info()
        logger.info("===================================================================")
        try:
            self.dataset = load_dataset(
                path=self.path, name=self.name, data_dir=self.data_dir, data_files=self.data_files, split=self.split,
                cache_dir=self.cache_dir, features=self.features, download_config=self.download_config,
                download_mode=self.download_mode, verification_mode=self.verification_mode,
                ignore_verifications=self.ignore_verifications, keep_in_memory=self.keep_in_memory,
                save_infos=self.save_infos, revision=self.revision, token=self.token,
                use_auth_token=self.use_auth_token,
                task=self.task, streaming=self.streaming, num_proc=self.num_proc, storage_options=self.storage_options
            )
        except Exception as e:
            logger.info(e)
            self.dataset = load_dataset(
                path=self.path, name=self.name, data_dir=self.data_dir, data_files=self.data_files, split=self.split,
                cache_dir=self.cache_dir, features=self.features, download_config=self.download_config,
                download_mode=self.download_mode, verification_mode=self.verification_mode,
                ignore_verifications=self.ignore_verifications, keep_in_memory=self.keep_in_memory,
                save_infos=self.save_infos, revision=self.revision, use_auth_token=self.use_auth_token,
                task=self.task, streaming=self.streaming, num_proc=self.num_proc, storage_options=self.storage_options
            )
        if isinstance(self.dataset, Dataset):
            logger.info(f"Transform the Dataset to DatasetDict")
            self.split = "Dataset" if self.split is None else self.split
            self.dataset = DatasetDict(
                {self.split: self.dataset}
            )
        elif isinstance(self.dataset, IterableDataset):
            logger.info(f"Transform the IterableDataset to IterableDatasetDict")
            self.split = "Dataset" if self.split is None else self.split
            self.dataset = IterableDatasetDict(
                {self.split: self.dataset}
            )
        logger.info(f"Type of the returned KisDataSet: {type(self.dataset)}")
        logger.info(f"splits in the dataset: {self.dataset.keys()}")

    def print_args_info(self):
        logger.info(f"path: {self.path}")
        logger.info(f"name: {self.name}")
        logger.info(f"data_dir: {self.data_dir}")
        logger.info(f"data_files: {self.data_files}")
        logger.info(f"split: {self.split}")
        logger.info(f"cache_dir: {self.cache_dir}")
        logger.info(f"features: {self.features}")
        logger.info(f"download_config: {self.download_config}")
        logger.info(f"download_mode: {self.download_mode}")
        logger.info(f"verfication_mode: {self.verification_mode}")
        logger.info(f"ignore_verifications: {self.ignore_verifications}")
        logger.info(f"keep_in_memory: {self.keep_in_memory}")
        logger.info(f"save_infos: {self.save_infos}")
        logger.info(f"revision: {self.revision}")
        logger.info(f"token: {self.token}")
        logger.info(f"use_auth_token: {self.use_auth_token}")
        logger.info(f"task: {self.task}")
        logger.info(f"streaming: {self.streaming}")
        logger.info(f"num_proc: {self.num_proc}")
        logger.info(f"storage_options: {self.storage_options}")
