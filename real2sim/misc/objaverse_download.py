import glob
import gzip
import json
import multiprocessing
import os
import urllib.request
import warnings
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

__version__ = "0.0.7"
ROOT_PATH = os.path.join("/home/xuanlin/objaverse")


def load_annotations(uids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Load the full metadata of all objects in the dataset.

    Args:
        uids: A list of uids with which to load metadata. If None, it loads
        the metadata for all uids.
    """
    metadata_path = os.path.join(ROOT_PATH, "metadata")
    object_paths = _load_object_paths()
    dir_ids = (
        set([object_paths[uid].split("/")[1] for uid in uids])
        if uids is not None
        else [f"{i // 1000:03d}-{i % 1000:03d}" for i in range(160)]
    )
    if len(dir_ids) > 10:
        dir_ids = tqdm(dir_ids)
    out = {}
    for i_id in dir_ids:
        json_file = f"{i_id}.json.gz"
        local_path = os.path.join(metadata_path, json_file)
        if not os.path.exists(local_path) or os.path.getsize(local_path) < 1024:
            hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/metadata/{i_id}.json.gz"
            # wget the file and put it in local_path
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            urllib.request.urlretrieve(hf_url, local_path)
        with gzip.open(local_path, "rb") as f:
            data = json.load(f)
        if uids is not None:
            data = {uid: data[uid] for uid in uids if uid in data}
        out.update(data)
        if uids is not None and len(out) == len(uids):
            break
    return out


def _load_object_paths() -> Dict[str, str]:
    """Load the object paths from the dataset.

    The object paths specify the location of where the object is located
    in the Hugging Face repo.

    Returns:
        A dictionary mapping the uid to the object path.
    """
    object_paths_file = "object-paths.json.gz"
    local_path = os.path.join(ROOT_PATH, object_paths_file)
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 1024:
        hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_paths_file}"
        # wget the file and put it in local_path
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        urllib.request.urlretrieve(hf_url, local_path)
    with gzip.open(local_path, "rb") as f:
        object_paths = json.load(f)
    return object_paths


def load_uids() -> List[str]:
    """Load the uids from the dataset.

    Returns:
        A list of uids.
    """
    return list(_load_object_paths().keys())


def _download_object(
    uid: str,
    object_path: str,
) -> Tuple[str, str]:
    """Download the object for the given uid.

    Args:
        uid: The uid of the object to load.
        object_path: The path to the object in the Hugging Face repo.

    Returns:
        The local path of where the object was downloaded.
    """
    # print(f"downloading {uid}")
    local_path = os.path.join(ROOT_PATH, object_path)
    tmp_local_path = os.path.join(ROOT_PATH, object_path + ".tmp")
    hf_url = (
        f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_path}"
    )
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(hf_url, tmp_local_path)

    os.rename(tmp_local_path, local_path)

    print(
        "Downloaded", uid
    )

    return uid, local_path


def load_objects(uids: List[str], download_processes: int = 1) -> Dict[str, str]:
    """Return the path to the object files for the given uids.

    If the object is not already downloaded, it will be downloaded.

    Args:
        uids: A list of uids.
        download_processes: The number of processes to use to download the objects.

    Returns:
        A dictionary mapping the object uid to the local path of where the object
        downloaded.
    """
    object_paths = _load_object_paths()
    out = {}
    if download_processes == 1:
        uids_to_download = []
        idx = 0
        for uid in uids:
            idx += 1
            if idx % 1000 == 0:
                print(f"processed {idx} uids")
            if uid.endswith(".glb"):
                uid = uid[:-4]
            if uid not in object_paths:
                warnings.warn(f"Could not find object with uid {uid}. Skipping it.")
                continue
            object_path = object_paths[uid]
            local_path = os.path.join(ROOT_PATH, object_path)
            if os.path.exists(local_path) and os.path.getsize(local_path) > 1024: # the incomplete files from git clone are < 1kb, so we need to overwrite them
                out[uid] = local_path
                continue
            uids_to_download.append((uid, object_path))
        if len(uids_to_download) == 0:
            return out
        for uid, object_path in uids_to_download:
            uid, local_path = _download_object(
                uid, object_path
            )
            out[uid] = local_path
    else:
        args = []
        idx = 0
        for uid in uids:
            idx += 1
            if idx % 1000 == 0:
                print(f"processed {idx} uids")
            if uid.endswith(".glb"):
                uid = uid[:-4]
            if uid not in object_paths:
                warnings.warn(f"Could not find object with uid {uid}. Skipping it.")
                continue
            object_path = object_paths[uid]
            local_path = os.path.join(ROOT_PATH, object_path)
            if not os.path.exists(local_path) or os.path.getsize(local_path) < 1024:
                args.append((uid, object_paths[uid]))
            else:
                out[uid] = local_path
        if len(args) == 0:
            return out
        print(
            f"starting download of {len(args)} objects with {download_processes} processes"
        )
        args = [(*arg,) for arg in args]
        with multiprocessing.Pool(download_processes) as pool:
            pool.starmap(_download_object, args, chunksize=max(10, len(args) // download_processes))
            # for uid, local_path in r:
            #     out[uid] = local_path
    return out


def load_lvis_annotations() -> Dict[str, List[str]]:
    """Load the LVIS annotations.

    If the annotations are not already downloaded, they will be downloaded.

    Returns:
        A dictionary mapping the LVIS category to the list of uids in that category.
    """
    hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/lvis-annotations.json.gz"
    local_path = os.path.join(ROOT_PATH, "lvis-annotations.json.gz")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(hf_url, local_path)
    with gzip.open(local_path, "rb") as f:
        lvis_annotations = json.load(f)
    return lvis_annotations


if __name__ == "__main__":
    # object_paths = _load_object_paths()
    # uids = [k for k, v in object_paths.items() if v.startswith("glbs/000-000")][
    #     500:1000
    # ]
    # uids = [k for k, v in object_paths.items()]
    
    # cabinets
    uids = ['e77dc1c101c3483a834b2a9dbe2b9754',
            '5680a17f941743f3a56fabfbd8dea3c7',
            'df46dde07f7540b7a23bedba35a469c7',
            '66e57d8c325f4bf98abf62f6de6176ed',
            'e783d6d64f8c453ab534bdde715b210d',
            'af295ff1fde6413ea41ef767a8c3e51c' # whole kitchen scene
    ]
    uids = ['715f375555634780890d84a00a2007ec', # coke can
            '30178d8ee92949499854f6edaac8574f', # soft drinks
            'eeecdb30541249ce8be4414d1971f095', # red bull
            'da0cfd4c6a4e41c5aaf06797b463235b', # apple
            'c0cef813b85947fdaaa8254867342687', # apple
    ]
    annotations = load_annotations(uids)
    final_uids = []
    for k in uids:
        # if k in annotations and annotations[k]['license'] == 'by' and annotations[k]['archives']['glb']['size'] < 2 * 1024 * 1024: # we don't want to download too big files
        if k in annotations and annotations[k]['license'] == 'by':
            final_uids.append(k)
    uids = final_uids
    print(f"Loaded {len(uids)} uids")
    objects = load_objects(uids, download_processes=min(10, len(uids)))
    print(f"Loaded {len(objects)} objects")