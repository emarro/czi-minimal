import glob
import logging
import os
import pathlib
import shutil
import time
from typing import Any, Literal

from composer import TimeUnit
from composer.callbacks import CheckpointSaver
from composer.core import Callback, State, Time, Timestamp
from composer.loggers import Logger
from composer.utils import PartialFilePath, dist, get_save_filename

import hashlib
import inspect
import re
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from types import MethodType

import fsspec
from huggingface_hub import HfApi, file_exists, repo_exists
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)


def fsspec_exists(filename):
    """Check if a file exists using fsspec."""
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def fsspec_listdir(dirname):
    """Listdir in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):
    """Mkdirs in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    fs.makedirs(dirname, exist_ok=exist_ok)


def snapshot_repo_to_tmp_dir(
    run_id: str | None = None,
    tmp_dir_exists_ok: bool = False,
) -> str:
    """Snapshot a repo to a local (tmp) directory.

    Args:
        run_id (optional: str): Run ID (e.g., wandb uuid), to be used in creating hash
            for the local (tmp) directory
            If None, timestamp is used.
        tmp_dir_exists_ok (bool): Whether to throw an error (False) if tmp dir exists
            already or re-use existing (True).
    """

    def _snapshot_files(src_path: Path, dest_path: Path, ignore: list[str]) -> None:
        """Helper method that recursively copies files from src_path to dest_path.
        Ignores files matching the patterns in ignore (list).
        """
        # print(f"Src path: {src_path}")
        # print(f"Ignoreing files: {ignore}")
        # print(
        #    [
        #        str(ignore_file)
        #        for ignore_file in ignore
        #        if re.search(re.escape(ignore_file), str(src_path))
        #    ]
        # )
        # if any([re.search(ignore_file, str(src_path)) for ignore_file in ignore]):
        if any(
            [
                Path(src_path).match(ignore_file)
                for ignore_file in ignore + [".git/*", "^tmp*"]
            ]
        ):
            return
        # print(f"Searching through {src_path}")
        # print(f"Escaped from .gitignore")
        if os.path.isdir(src_path):
            dest_path.mkdir(parents=True, exist_ok=True)
            for sp in fsspec_listdir(src_path):
                _snapshot_files(
                    src_path / sp, dest_path / Path(sp).resolve().name, ignore
                )
        if os.path.isdir(src_path):
            return
        # else: src_path is a file
        shutil.copy2(src_path, dest_path)

    # Get .gitignore list
    project_root = Path(__file__).resolve().parent.parent.parent
    with open(project_root / ".gitignore", "r", encoding="utf-8") as gf:
        ignore_list = [line.strip() for line in gf.readlines() if len(line.strip()) > 0]
    ignore_list.extend(
        [ignore_file[:-1] for ignore_file in ignore_list if ignore_file.endswith("/")]
    )

    # Construct a unique ID for temporary directory
    root = gettempdir()
    log.debug(root)
    hash_key = hashlib.blake2s(
        (run_id if run_id is not None else str(int(time.time()))).encode("utf-8"),
        digest_size=16,
    ).hexdigest()
    tmp_dir = os.path.join(root, f"tmp{hash_key}")
    if fsspec_exists(tmp_dir):
        if tmp_dir_exists_ok:
            return tmp_dir
        else:
            raise ValueError(
                f"Cannot create snapshot. Temporary directory {tmp_dir} already exists."
                " Please remove it or use a different run_id."
            )
    fsspec_mkdirs(tmp_dir)
    log.debug(tmp_dir)
    _snapshot_files(project_root, Path(tmp_dir).resolve(), ignore_list)
    log.debug(f"Snapshot saved to {tmp_dir}")
    return tmp_dir


def _flatten_and_copy(src_path: Path, dest_path: Path, ignore: list[str]) -> None:
    """Copy file contents and flatten relative imports.

    Ignores __init__.py files.
    Recursively applies to directories and flattens file names from `/` to `_`
    """

    def _copy_file_contents_and_flatten_relative_imports(src, dest):
        """Helper method that copies file contents and flattens relative imports.

        All instances of `src.` are replaced with `.` and all `.` (after the first one)
            in relative imports are replaced with `_`.
        """
        # print(f"Flattening imports in {src}")
        with open(src, "r", encoding="utf-8") as f:
            lines = f.readlines()
        modified_lines = []
        for line in lines:
            # Match lines starting with "import ."
            if re.match(r"^\s*import\s+\.(\S+)\s*$", line):
                # Replace all remaining '.' with '_'
                modified_line = re.sub(
                    r"import \.([\w.]+)",
                    lambda m: f"import {m.group(1).replace('.', '_')}",
                    line,
                )
                assert modified_line[0] != "_", f"Line {modified_line} invalid import"

            # Match lines starting with "from ."
            elif re.match(r"^\s*from\s+\.(\S+)\s+import", line):
                # Replace all remaining '.' with '_'
                modified_line = re.sub(
                    r"from \.([\w.]+)",
                    lambda m: f"from {m.group(1).replace('.', '_')}",
                    line,
                )
                if modified_line[0] == "_":
                    modified_line = modified_line.replace("_", ".", 1)
                assert modified_line[0] != "_", f"Line {modified_line} invalid import"

            # Match lines starting with "import hnet."
            elif re.match(r"^\s*import\s+hnet\.(\S+)\s*$", line):
                # Replace 'import hnet.' with 'import .'
                modified_line = re.sub(r"^\s*import\s+hnet\.hnet\.", "import .", line)
                # Replace all remaining '.' with '_'
                modified_line = re.sub(
                    r"^import \.([\w.]+)",
                    lambda m: f"import .{m.group(1).replace('.', '_')}",
                    modified_line,
                )
                # assert False, f"subbed {line} -> {modified_line}"
                if modified_line.encode("utf-8")[0] == "_":
                    modified_line = modified_line.replace("_", ".", 1)
                assert modified_line[0] != "_", f"Line {modified_line} invalid import"
                assert "." not in modified_line, f"Line {modified_line} invalid import"

            # Match lines starting with "from hnet."
            elif re.match(r"^\s*from\s+hnet\.(\S+)\s+import", line):
                # Replace 'from hnet.' with 'from .'
                modified_line = re.sub(r"^\s*from\s+hnet\.hnet\.", "from .", line)
                # Replace all remaining '.' with '_'
                modified_line = re.sub(
                    r"^from \.([\w.]+)",
                    lambda m: f"from .{m.group(1).replace('.', '_')}",
                    modified_line,
                )  # .replace("_", ".", 1)
                # assert False, f"subbed {line} -> {modified_line}"
                if modified_line.encode("utf-8")[0] == "_":
                    modified_line = modified_line.replace("_", ".", 1)
                assert modified_line[0] != "_", f"Line {modified_line} invalid import"
                # assert "_" not in modified_line[0:2], (
                #    f"Line {modified_line} invalid import"
                # )

            # Match lines starting with "import caduceus."
            elif re.match(r"^\s*import\s+caduceus\.(\S+)\s*$", line):
                # Replace 'import caduceus.' with 'import .'
                modified_line = re.sub(
                    r"^\s*import\s+caduceus\.caduceus\.", "import .caduceus.", line
                )
                # Replace all remaining '.' with '_'
                modified_line = re.sub(
                    r"^import \.([\w.]+)",
                    lambda m: f"import .{m.group(1).replace('.', '_')}",
                    modified_line,
                )
                # assert False, f"subbed {line} -> {modified_line}"
                if modified_line.encode("utf-8")[0] == "_":
                    modified_line = modified_line.replace("_", ".", 1)
                assert modified_line[0] != "_", f"Line {modified_line} invalid import"
                assert "." not in modified_line, f"Line {modified_line} invalid import"

            # Match lines starting with "from caduceus."
            elif re.match(r"^\s*from\s+caduceus\.(\S+)\s+import", line):
                # Replace 'from caduceus.' with 'from .'
                modified_line = re.sub(
                    r"^\s*from\s+caduceus\.caduceus\.", "from .caduceus.", line
                )
                # Replace all remaining '.' with '_'
                modified_line = re.sub(
                    r"^from \.([\w.]+)",
                    lambda m: f"from .{m.group(1).replace('.', '_')}",
                    modified_line,
                )  # .replace("_", ".", 1)
                # assert False, f"subbed {line} -> {modified_line}"
                if modified_line.encode("utf-8")[0] == "_":
                    modified_line = modified_line.replace("_", ".", 1)
                assert modified_line[0] != "_", f"Line {modified_line} invalid import"
                # assert "_" not in modified_line[0:2], (
                #    f"Line {modified_line} invalid import"
                # )

            else:
                modified_line = line

            modified_lines.append(modified_line.encode("utf-8"))
        with open(
            dest,
            "wb",
        ) as f:
            f.writelines(modified_lines)

    if any([Path(src_path).match(ignore_file) for ignore_file in ignore]):
        log.debug("Skipping:", src_path)
        # print("Skipping:", src_path)
        return
    if os.path.isdir(src_path):
        for sp in fsspec_listdir(src_path):
            # print(
            # f"Copying and flattening {sp} to -> {Path(f'{str(dest_path)}_{Path(sp).resolve().name}')}"
            # )
            _flatten_and_copy(
                src_path / sp,
                Path(f"{str(dest_path)}_{Path(sp).resolve().name}"),
                ignore,
            )
    if os.path.isdir(src_path):
        return
    # Copy contents; replace `.` in relative imports with `_` and remove `src.` prefix
    # (e.g. `src.backbone.dit` -> `.backbone_dit`)
    log.debug(f"Copying {src_path} to {dest_path}")
    _copy_file_contents_and_flatten_relative_imports(src_path, dest_path)


def _state_dict_no_buffers(self, *args, **kwargs):
    # Copy original state_dict
    filtered_state_dict = {
        k: v for k, v in super(self.__class__, self).state_dict(*args, **kwargs).items()
    }
    # Skip explicitly listed parameter names
    skip_params_for_push = getattr(self, "skip_params_for_push", [])
    for skip_name in skip_params_for_push:
        filtered_state_dict.pop(skip_name, None)

    return filtered_state_dict


def save_pretrained_or_push_to_hub(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    repo_id: str = "emarro/dna-dev",
    commit_message: str = "Add model and code",
    local: bool = False,
    private: bool = True,
    project_root: str | None = None,
) -> None:
    """Push / Save model and code to hub / local directory.

    Enables model loading using `AutoModel.from_pretrained` paradigm.

    Args:
        model (PreTrainedModel): Model to push / save.
        tokenizer (PreTrainedTokenizer): Tokenizer to push / save.
        repo_id (str) Repository ID on Hugging Face Hub / Local directory.
        commit_message (str): Commit message.
        local (bool): If True, push to local directory instead of Hugging Face Hub.
        private (bool): Whether remote hub repo is private.
        project_root (optional: str): Path to the project root directory. If None, uses
            the parent of __file__ path.
            Use this parameter if, for example, pushing from a tmp copy of the repo.
    """
    # Register config and model classes
    model.config.auto_map = model.config.auto_map
    model_cls_path = (  # e.g.:
        inspect.getfile(model.__class__)  # <project_path>/src/denoiser/diffusion.py
        .split(str(Path(__file__).resolve().parent.parent))[-1]
        .replace("/", ".")  # .src.denoiser.diffusion.py
        .split(".py")[0][1:]  # src.denoiser.diffusion
    )
    # print(f"Importing {model_cls_path} from {type(model).__name__}")
    exec(f"from {model_cls_path} import {type(model).__name__}")
    exec(f"from {model_cls_path} import {type(model.config).__name__}")
    exec(f"{type(model.config).__name__}.register_for_auto_class()")
    for automodel in ["AutoModel", "AutoModelForCausalLM", "AutoModelForMaskedLM"]:
        if automodel in model.config.__class__.auto_map.keys():
            exec(f'{type(model).__name__}.register_for_auto_class("{automodel}")')

    # Update model config paths to remove `src` and flatten (replace `.` with `_`)
    # in `_target_` (e.g. `src.backbone.dit` -> `backbone_dit`)
    # if re.match(r"^src\.", model.config.backbone_config["_target_"]):
    #    model.config.backbone_config["_target_"] = re.sub(
    #        r"^([\w.]+)\.",
    #        lambda m: f"{m.group(1).replace('.', '_')}.",
    #        re.sub(r"^src\.", "", model.config.backbone_config["_target_"]),
    #    )
    # if re.match(r"^src\.", model.config.noise_config["_target_"]):
    #    model.config.noise_config["_target_"] = re.sub(
    #        r"^([\w.]+)\.",
    #        lambda m: f"{m.group(1).replace('.', '_')}.",
    #        re.sub(r"^src\.", "", model.config.noise_config["_target_"]),
    #    )
    # log.debug("Updated model.config:")
    # log.debug(model.config)

    # Set up destination

    tmp_dir = TemporaryDirectory() if not local else None
    dest_path = Path(repo_id) if local else Path(tmp_dir.name)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Temporarily override state_dict() to remove buffers
    model.state_dict = MethodType(_state_dict_no_buffers, model)
    # Save/push model and tokenizer
    # print(f"Destination path: {dest_path}")
    # print(model)
    log.debug(f"{'Saving' if local else 'Pushing'} model to {repo_id}")
    if local:
        model.save_pretrained(dest_path, safe_serialization=False)
        tokenizer.save_pretrained(dest_path)
    else:
        if not repo_id:
            raise ValueError("Argument `repo_id` is required for push_to_hub.")
        if not repo_exists(repo_id) or not file_exists(repo_id, "tokenizer.json"):
            tokenizer.push_to_hub(
                repo_id, private=private, commit_message="Upload tokenizer"
            )

        model.push_to_hub(
            repo_id,
            private=private,
            commit_message="Update pytorch.bin; " + commit_message,
            safe_serialization=False,
        )

    # Copy source files
    # print(f"Project root: {project_root}")
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    else:
        project_root = Path(project_root).resolve()
    # print(f"ls {project_root}: {os.listdir(project_root)}")
    with open(project_root / ".gitignore", "r", encoding="utf-8") as gf:
        ignore = [line.strip() for line in gf.readlines() if len(line.strip()) > 0]
    ignore.extend(
        [ignore_file[:-1] for ignore_file in ignore if ignore_file.endswith("/")]
    )
    ignore.append("__init__.py")
    model_file_path = inspect.getfile(model.__class__).split(
        str(Path(__file__).resolve().parent.parent.parent)
    )[-1][1:]
    ## TODO: Remove debugging prints
    # print(f"Model: {model}")
    # print(f"Model file: {model_file_path}")
    # upload each necessary file individually to HF
    paths_to_copy = {
        project_root / ".gitignore": ".gitignore",
        # project_root / "src/denoiser/base.py": "denoiser_base.py",
        project_root / model_file_path: model_file_path.split("/")[-1],
        project_root / "src/hnet/hnet/modules": "modules",
        project_root / "src/hnet/hnet/models": "models",
        project_root / "src/caduceus/caduceus": "caduceus",
        # project_root / "src/noise_schedule": "noise_schedule",
    }
    for src_path, dest_name in paths_to_copy.items():
        dest = dest_path / dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        _flatten_and_copy(src_path, dest, ignore)
    # Add __init__.py
    (dest_path / "__init__.py").touch()
    # Upload to hub if not local
    if not local:
        api = HfApi()
        log.debug(f"Creating repo or fetching URL for {repo_id}")
        url = api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
        log.debug(f"Found repo at {url}")

        log.debug(f"Uploading files to {repo_id} from {dest_path}")
        commit_info = api.upload_folder(
            folder_path=dest_path,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        log.debug(f"Commit info: {commit_info}")
        log.debug(f"Removing temporary directory {tmp_dir.name}")
        tmp_dir.cleanup()
    log.debug("Done")


class HuggingFaceCompatibleCheckpointing(CheckpointSaver):
    """A checkpoint callback that saves models in a manner in which one can
    `AutoModel.from_pretrained(<ckpt_path>)`.

    """

    def __init__(
        self,
        disable_hf: bool = False,
        save_local: bool = True,
        save_to_hub: bool = False,
        hub_repo_id: str | None = None,
        private: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_to_hub = save_to_hub and not disable_hf
        self.hub_repo_id = hub_repo_id
        if self.save_to_hub and hub_repo_id is None:
            raise ValueError("Saving to hub requires a hub repo id be provided.")
        self.save_local = save_local and not disable_hf
        self.disable_hf = disable_hf or not (self.save_to_hub or self.save_local)
        self.private = private
        self.project_root = None
        self.hf_filename = PartialFilePath(
            f"HF_{self.filename.filename.split('.pt')[0]}", self.filename.folder
        )
        if self.latest_filename is not None:
            self.latest_hf_filename = PartialFilePath(
                f"HF_{self.latest_filename.filename.split('.pt')[0]}",
                self.latest_filename.folder,
            )
        self.saved_hf_checkpoints: list[str] = []
        self.all_saved_hf_checkpoints_to_timestamp: dict[str, Timestamp] = {}
        # TODO: Leads to OSError device is busy when using tmpdir in /share/kuleshov

    def fit_start(self, state: State, logger: Logger) -> None:
        super().fit_start(state, logger)
        if dist.get_global_rank() == 0 and not self.disable_hf:
            self.project_root = snapshot_repo_to_tmp_dir(tmp_dir_exists_ok=True)
            log.info(f"Created tmp repo for HF checkpointing at {self.project_root}")
        dist.barrier()  # Holds all the ranks until repo snapshot is done

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()

        all_hf_checkpoints = []
        for (
            save_filename,
            timestamp,
        ) in self.all_saved_hf_checkpoints_to_timestamp.items():
            all_hf_checkpoints.append((save_filename, timestamp.state_dict()))
        state_dict["all_saved_hf_checkpoints_to_timestamp"] = all_hf_checkpoints
        return state_dict

    def load_state_dict(self, state: dict[str, Any]):
        super().load_state_dict(state)
        if "all_saved_hf_checkpoints_to_timestamp" in state:
            for save_filename, timestamp_state in state[
                "all_saved_hf_checkpoints_to_timestamp"
            ]:
                load_timestamp = Timestamp()
                load_timestamp.load_state_dict(timestamp_state)
                self.all_saved_hf_checkpoints_to_timestamp[save_filename] = (
                    load_timestamp
                )

    def _save_checkpoint(self, state: State, logger: Logger) -> None:
        """
        Copied / adapted from composer.callbacks.CheckpointSaver._save_checkpoint
            for HF compatibility.
        """
        # TODO: Check that HF saving works with state.fsdp_sharded_state_dict_enabled
        #  (or if we can ignore this scenario).
        # TODO: Do we need to implement HF uploading for remote uploading too?
        # Hacky try/catch to traige errors with removing already deleted ckpts (concurrency bug?)
        try:
            super()._save_checkpoint(state, logger)  # Perform standard checkpointing
        except FileNotFoundError as e:
            pass
        if self.disable_hf:
            # super()._save_checkpoint(state, logger)  # Perform standard checkpointing
            # Exit and don't upload to hf
            return

        hf_filename_with_placeholders = self.hf_filename.format(
            state, keep_placeholders=True
        )
        save_hf_filename = get_save_filename(state, hf_filename_with_placeholders)
        self.all_saved_hf_checkpoints_to_timestamp[save_hf_filename] = state.timestamp

        # Adapting `checkpoint.save_checkpoint / ._save_checkpoint` for HF
        saved_hf_path = None
        if dist.get_global_rank() == 0:
            if self.save_local:
                save_pretrained_or_push_to_hub(
                    model=state.model.module.model
                    if hasattr(state.model, "module")
                    else state.model.model,
                    tokenizer=state.model.module.tokenizer
                    if hasattr(state.model, "module")
                    else state.model.tokenizer,
                    repo_id=save_hf_filename,
                    local=True,
                    project_root=self.project_root,
                    private=self.private,
                )
                saved_hf_path = save_hf_filename
                log.debug(f"HF checkpoint locally saved to {saved_hf_path}")
            if self.save_to_hub:
                metrics_str = "Train metrics:\n\t" + "\n\t".join(
                    [
                        f"{k}={v.item():0.4f}"
                        for k, v in state.train_metric_values.items()
                    ]
                )
                if hasattr(state, "eval_metric_values"):
                    metrics_str += "\n\nVal metrics:\n\t" + "\n\t".join(
                        [
                            f"{k}={v.item():0.4f}"
                            for k, v in state.eval_metric_values.items()
                        ]
                    )
                flop_counter_callbacks = [
                    callback
                    for callback in state.callbacks
                    if "total_train_flops" in dir(callback)
                ]
                total_train_flops = -1
                if len(flop_counter_callbacks) > 0:
                    total_train_flops = flop_counter_callbacks[0].total_train_flops
                    # print(f"total_train_flops: {total_train_flops:e}")
                commit_message = (
                    f"Checkpoint @ Epoch {state.timestamp.epoch.value}, "
                    f"Batch {state.timestamp.batch.value}\n\n"
                    f"{metrics_str}\n\n"
                    f"Timestamp:\n"
                    f"\titeration={state.timestamp.iteration.value}\n"
                    f"\tepoch={state.timestamp.epoch.value}\n"
                    f"\tbatch={state.timestamp.batch.value}\n"
                    f"\tsample={state.timestamp.sample.value}\n"
                    f"\ttoken={state.timestamp.token.value}\n"
                    f"\ttrain_flops={total_train_flops:e}\n"
                    f"\tepoch_in_iteration={state.timestamp.epoch_in_iteration.value}\n"
                    f"\ttoken_in_iteration={state.timestamp.token_in_iteration.value}\n"
                    f"\tbatch_in_epoch={state.timestamp.batch_in_epoch.value}\n"
                    f"\tsample_in_epoch={state.timestamp.sample_in_epoch.value}\n"
                    f"\ttoken_in_epoch={state.timestamp.token_in_epoch.value}"
                )
                save_pretrained_or_push_to_hub(
                    model=state.model.module.model
                    if hasattr(state.model, "module")
                    else state.model.model,
                    tokenizer=state.model.module.tokenizer
                    if hasattr(state.model, "module")
                    else state.model.tokenizer,
                    repo_id=self.hub_repo_id,
                    local=False,
                    project_root=self.project_root,
                    commit_message=commit_message,
                    private=self.private,
                )
            log.debug(f"HF checkpoint pushed to {self.hub_repo_id}")

            # if not saved_hf_path:  # not all ranks save
            # super()._save_checkpoint(state, logger)  # Perform standard checkpointing
            # return

        self.rank_saves_symlinks = (
            dist.get_global_rank() == 0 or not state.fsdp_sharded_state_dict_enabled
        )
        if self.latest_hf_filename is not None and self.num_checkpoints_to_keep != 0:
            symlink = self.latest_hf_filename.format(state)
            os.makedirs(os.path.dirname(symlink), exist_ok=True)
            try:
                os.remove(symlink)
            except FileNotFoundError:
                pass
            # Sharded checkpoints for torch >2.0 use directories not files for
            # load_paths
            if state.fsdp_sharded_state_dict_enabled:
                src_path = str(pathlib.Path(saved_hf_path).parent)
            else:
                src_path = saved_hf_path
            if self.rank_saves_symlinks:
                os.symlink(os.path.relpath(src_path, os.path.dirname(symlink)), symlink)
        self.saved_hf_checkpoints.append(saved_hf_path)
        # Debug: Load from HF
        print(f"Loading model saved at: {str(saved_hf_path)}")
        AutoModel.from_pretrained(saved_hf_path, trust_remote_code=True)
        print(f"Loading Tokenizer saved at: {str(saved_hf_path)}")
        AutoTokenizer.from_pretrained(saved_hf_path, trust_remote_code=True)

        if self.num_checkpoints_to_keep >= 0:
            # Adapting `super().__rotate_checkpoints` for HF
            while len(self.saved_hf_checkpoints) > self.num_checkpoints_to_keep:
                checkpoint_to_delete = self.saved_hf_checkpoints.pop(0)
                prefix_dir = str(pathlib.Path(checkpoint_to_delete).parent)
                if not state.fsdp_sharded_state_dict_enabled:
                    shutil.rmtree(checkpoint_to_delete)
                else:
                    if dist.get_global_rank() == 0:
                        shutil.rmtree(prefix_dir)

    def close(self, state: State, logger: Logger) -> None:
        """Clean up tmp repo snapshot"""
        if dist.get_global_rank() == 0:
            # Only clean up if project_root was initialized (not empty string)
            if self.project_root and fsspec_exists(self.project_root):
                shutil.rmtree(self.project_root)
        dist.barrier()
        super().close(state, logger)
