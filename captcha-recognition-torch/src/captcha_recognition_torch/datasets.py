from kedro.io import AbstractDataset
import zipfile

from typing import Any, Dict, Type, Union, Optional


class ZipFileDataSet(AbstractDataset):
    def __init__(self, filepath):
        self._filepath = filepath

    def _load(self) -> dict:
        with zipfile.ZipFile(self._filepath, 'r') as zip_ref:
            return {name: zip_ref.namelist() for name in zip_ref.namelist()}
            # return {name: zip_ref.read(name) for name in zip_ref.namelist()}

    def _save(self, data: dict) -> None:
        raise NotImplementedError("Save operation is not supported for ZipFileDataSet.")
    
    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
        )


# import os
# import tempfile
# from copy import deepcopy
# from typing import Any, Dict, Type, Union, Optional
# from warnings import warn

# from kedro.io import AbstractDataset, DatasetError
# from kedro.io.core import parse_dataset_definition, VERSION_KEY, VERSIONED_FLAG_KEY


# class ZipFileDataSet(AbstractDataset):
#     """
#     ZipFileDataSet decompresses and extracts files from zip files.
#     Expects to return a single file from the unzipped dataset,
#     and supports multiple methods for filtering sets of files.
#     """

#     DEFAULT_DATASET = {
#         "type": "text.TextDataset",
#         "fs_args": {
#             "open_args_load": {
#                 "mode": "rb",
#             }
#         }
#     }

#     def __init__(
#             self,
#             filepath: str,
#             zipped_filename: str = None,
#             zipped_filename_suffix: str = None,
#             ignored_prefixes: str = None,
#             ignored_suffixes: str = None,
#             credentials: Dict[str, str] = None,
#             dataset: Optional[Union[str, Type[AbstractDataset], Dict[str, Any]]] = None,
#             filepath_arg: str = 'filepath',
#     ):

#         if dataset is None:
#             dataset = ZipFileDataSet.DEFAULT_DATASET

#         dataset = dataset if isinstance(dataset, dict) else {"type": dataset}
#         self._dataset_type, self._dataset_config = parse_dataset_definition(dataset)
#         if VERSION_KEY in self._dataset_config:
#             raise DatasetError(
#                 "`{}` does not support versioning of the underlying dataset. "
#                 "Please remove `{}` flag from the dataset definition.".format(
#                     self.__class__.__name__, VERSIONED_FLAG_KEY
#                 )
#             )

#         self._filepath_arg = filepath_arg
#         if self._filepath_arg in self._dataset_config:
#             warn(
#                 "`{}` key must not be specified in the dataset definition as it "
#                 "will be overwritten by partition path".format(self._filepath_arg)
#             )

#         self._filepath = filepath
#         self._zipped_filename = zipped_filename
#         self._zipped_filename_suffix = zipped_filename_suffix
#         self._ignored_prefixes = ignored_prefixes or ['_', '.']
#         self._ignored_suffixes = (ignored_suffixes or []) + ['/']
#         credentials = credentials or {}
#         self._password = credentials.get('password', credentials.get('pwd'))

#     def _is_ignored(self, name):
#         for ignored_prefix in self._ignored_prefixes:
#             if name.startswith(ignored_prefix):
#                 return True
#         for ignored_suffix in self._ignored_suffixes:
#             if name.endswith(ignored_suffix):
#                 return True
#         return False

#     def _load(self) -> bytes:
#         import zipfile
#         with zipfile.ZipFile(self._filepath) as zipped:
#             namelist = zipped.namelist()
#             if self._zipped_filename_suffix is not None:
#                 namelist = [
#                     name for name in namelist if name.lower().endswith(self._zipped_filename_suffix)
#                 ]
#             namelist = [
#                 name
#                 for name in namelist if not self._is_ignored(name)
#             ]
#             if len(namelist) > 1 and self._zipped_filename is None:
#                 raise DatasetError(f'Multiple files found! Please specify which file to extract: {namelist}')
#             if len(namelist) <= 0:
#                 raise DatasetError(f'No files found in the archive!')

#             target_filename = namelist[0]
#             if self._zipped_filename is not None:
#                 target_filename = self._zipped_filename

#             with zipped.open(target_filename, pwd=self._password) as zipped_file:
#                 temp_unzipped_dir = tempfile.mkdtemp()
#                 temp_unzipped_filepath = os.path.join(temp_unzipped_dir, "temp_file")
#                 with open(temp_unzipped_filepath, "wb") as temp_unzipped_file:
#                     temp_unzipped_file.write(zipped_file.read())

#                 kwargs = deepcopy(self._dataset_config)
#                 kwargs[self._filepath_arg] = temp_unzipped_filepath
#                 dataset = self._dataset_type(**kwargs)
#                 data = dataset.load()
#                 os.remove(temp_unzipped_filepath)
#                 return data

#     def _save(self, data: Any) -> None:
#         raise DatasetError(f'Saving is unsupported')

#     def _describe(self) -> Dict[str, Any]:
#         return dict(
#             filepath=self._filepath,
#             zipped_filename=self._zipped_filename,
#             zipped_filename_suffix=self._zipped_filename_suffix,
#             ignored_prefixes=self._ignored_prefixes,
#             ignored_suffxies=self._ignored_suffixes,
#         )