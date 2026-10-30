"""Build an nnU-Net junction landmark (heatmap regression) training dataset from
the synthetic DNA images, in the same format as the segmentation-probability-only
dataset built from real data (create_nnunet_heatmap_dataset.py followed by the
segprob-only variant of create_nnunet_dataset_variants.py, e.g.
Dataset012_JunctionDetection_v1_SegProbOnly):

    <output-dir>/
        dataset.json
        imagesTr/<case>_0000.tif   float32 1024x1024 segmentation foreground probability
        labelsTr/<case>.tif        uint8 1024x1024 class-id blob map
                                   (1 = Normal Fork, 2 = Reversed Fork / Crossing)
        case_mapping.json          case -> HF source image, class, fiber range, junction

Full 4096x4096 images are sampled from the train split of the HuggingFace repo
(jhehli/synthetic-tem-images), balanced over the fork classes and fiber ranges
(reproducible via --seed). Each image is segmented with a trained nnU-Net
segmentation model in 4x4 patches of 1024x1024, exactly like the real-data
probability maps (AnnotationTool/backend/pipeline/backfill_segmentation_probabilities.py);
the stitched probability map is resized to 1024x1024. The single manifest junction
per fork image becomes the landmark label (replication_fork -> Normal Fork,
reversed_fork -> Reversed Fork / Crossing); negative images get an empty label map.

Images the segmentation model was trained on can be excluded with
--exclude-case-mapping (the case_mapping.json written by
Segmentation/PreProcessing/nnUNet/generate_nnUNet_synthetic_dataset.py), so their
probability maps aren't unrealistically clean.

With --hf-repo, the finished dataset folder is uploaded to that (private, created
if missing) HuggingFace dataset repo, under <output-dir name>/.

Run from the repo root in the nnU-Net environment (GPU), e.g. via
JunctionDetection/PreProcessing/synthetic_nnunet_landmark_dataset_job.sh:
    python -m JunctionDetection.PreProcessing.create_synthetic_nnunet_landmark_dataset \\
        --model-dir <nnUNet_results>/Dataset021_SegmentationSynthetic/<trainer>__nnUNetPlans__2d \\
        --output-dir <nnUNet_raw>/Dataset022_JunctionDetectionSynthetic_SegProbOnly \\
        [--exclude-case-mapping <nnUNet_raw>/Dataset021_SegmentationSynthetic/case_mapping.json] \\
        [--n-images 126] [--seed 42] [--hf-repo <user>/<repo>]
"""
import argparse
import json
import random
import tempfile
from pathlib import Path

import pandas as pd
import tifffile
import torch
import torchvision.transforms.functional as TF

from JunctionDetection.PreProcessing.create_nnunet_heatmap_dataset import (
    FILE_ENDING,
    NEGATIVE_LABEL,
    NNUNET_IMAGES_DIR,
    NNUNET_LABELS_DIR,
    NORMAL_FORK_LABEL,
    RAW_CHANNEL_ID,
    REVERSED_FORK_CROSSING_COMBINED_LABEL,
    _resize_array,
    _sanitize_case_id,
    create_labels,
    get_label_mapping,
)
from Segmentation.PostProcessing.segmentation_postprocessing import stitch_mask_tiles
from Segmentation.PreProcessing.General.preprocessing_util import (
    create_patches_from_img,
    init_dir,
)
from Segmentation.PreProcessing.nnUNet.generate_nnUNet_synthetic_dataset import (
    FORK_TYPES,
    HF_REPO_ID,
    fetch,
    load_manifest,
)
from Segmentation.Util.nnunet_util import (
    NNUNET_DEFAULT_CHECKPOINT,
    NNUNET_DEFAULT_FOLDS,
    initialize_nnunet_predictor,
    run_nnunet_predict_from_patches,
)
from Segmentation.Util.patch_grid_util import (
    GRID_SIZE,
    PATCH_SIZE,
    load_probability_pred_patches,
    nnunet_input_patch_filename,
)

SPLIT = "train"
# same size as Dataset012_JunctionDetection_v1_SegProbOnly
N_IMAGES = 126
SEED = 42
IMAGE_SIZE = 4096
IMAGE_RESIZE = 1024

CLASS_TO_LABEL = {
    "replication_fork": NORMAL_FORK_LABEL,
    "reversed_fork": REVERSED_FORK_CROSSING_COMBINED_LABEL,
    "negative": NEGATIVE_LABEL,
}


def balanced_cell_counts(
    cells: dict[tuple[str, int], list[dict]],
    n_images: int,
    rng: random.Random,
) -> dict[tuple[str, int], int]:
    """Number of images to draw per (class, fiber_min) cell.

    Images are handed out one at a time to the cell with the fewest images so
    far, breaking ties by the running class total, then the running fiber range
    total, then randomly. This yields equal per-cell counts with the remainder
    spread evenly over classes and fiber ranges, and respects cell sizes."""
    n_available = sum(len(rows) for rows in cells.values())
    if n_images > n_available:
        raise ValueError(
            f"Requested {n_images} images but only {n_available} are available")

    counts = {cell: 0 for cell in cells}
    class_totals = {cls: 0 for cls, _ in cells}
    fiber_totals = {fiber: 0 for _, fiber in cells}
    tie_breaks = {cell: rng.random() for cell in cells}
    for _ in range(n_images):
        cell = min(
            (c for c in cells if counts[c] < len(cells[c])),
            key=lambda c: (counts[c], class_totals[c[0]], fiber_totals[c[1]],
                           tie_breaks[c]),
        )
        counts[cell] += 1
        class_totals[cell[0]] += 1
        fiber_totals[cell[1]] += 1
        # re-randomize the cell's tie-break so remainders don't follow a fixed order
        tie_breaks[cell] = rng.random()
    return counts


def sample_images(rows: list[dict], n_images: int, seed: int) -> list[dict]:
    """Randomly sample n_images manifest rows, balanced over fork classes and
    fiber ranges (see balanced_cell_counts)."""
    rng = random.Random(seed)
    cells: dict[tuple[str, int], list[dict]] = {}
    for row in sorted(rows, key=lambda row: row["tem"]):
        cells.setdefault((row["class"], row["fiber_min"]), []).append(row)
    cells = dict(sorted(cells.items(), key=lambda item: (
        FORK_TYPES.index(item[0][0]), item[0][1])))

    counts = balanced_cell_counts(cells, n_images, rng)
    sampled = []
    for cell, cell_rows in cells.items():
        sampled += sorted(rng.sample(cell_rows, counts[cell]),
                          key=lambda row: row["tem"])
    return sampled


def print_balance(samples: list[dict], n_available: int):
    df = pd.DataFrame(samples)
    df["fibers"] = df["fiber_min"].astype(str) + "-" + df["fiber_max"].astype(str)
    table = pd.crosstab(df["class"], df["fibers"], margins=True)
    table = table[sorted(table.columns[:-1], key=lambda c: int(c.split("-")[0]))
                  + ["All"]]
    print(f"Sampled {len(samples)} of {n_available} available {SPLIT} images:")
    print(table.to_string())


def load_excluded_images(case_mapping_path: Path) -> set[str]:
    """File names of the train split images used in the segmentation dataset."""
    with open(case_mapping_path) as f:
        mapping = json.load(f)
    return {case["original_filename"] for case in mapping
            if case["split"] == SPLIT}


def image_stem(row: dict) -> str:
    return Path(row["tem"]).stem


def segment_probabilities(img_path: Path, stem: str, predictor) -> torch.Tensor:
    """Stitched (4096x4096) nnU-Net foreground probability map of one full image,
    predicted in 4x4 patches of 1024x1024."""
    with tempfile.TemporaryDirectory(prefix="synth_landmark_seg_") as tmp:
        tmp_dir = Path(tmp)
        patches = create_patches_from_img(img_path, patch_size=PATCH_SIZE[0])
        input_file_lists = []
        for idx in range(patches.shape[0]):
            fname = nnunet_input_patch_filename(stem, idx)
            TF.to_pil_image(patches[idx]).save(tmp_dir / fname)
            input_file_lists.append([str(tmp_dir / fname)])

        pred_dir = tmp_dir / "pred"
        run_nnunet_predict_from_patches(
            predictor, input_file_lists, pred_dir, save_probabilities=True)

        prob_patches, _ = load_probability_pred_patches(pred_dir, stem)
        stitched = stitch_mask_tiles(
            prob_patches, grid_size=GRID_SIZE,
            original_input_patch_img_size=PATCH_SIZE, as_uint=False)
        return stitched.squeeze(0).detach().cpu()


def write_probability_images(samples: list[dict], output_dir: Path, predictor):
    images_out_dir = output_dir / NNUNET_IMAGES_DIR
    for idx, row in enumerate(samples, start=1):
        stem = image_stem(row)
        prob = segment_probabilities(
            fetch(f"{SPLIT}/{row['tem']}"), stem, predictor).numpy()
        if prob.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"Expected a {IMAGE_SIZE}x{IMAGE_SIZE} probability map, got "
                f"{prob.shape} for {row['tem']}")

        # same resizing as create_nnunet_heatmap_dataset.py
        prob = _resize_array(prob, (IMAGE_RESIZE, IMAGE_RESIZE), prob.dtype)
        tifffile.imwrite(
            images_out_dir /
            f"{_sanitize_case_id(stem)}_{RAW_CHANNEL_ID}{FILE_ENDING}",
            prob)
        print(f"Segmented {idx}/{len(samples)}: {stem}")


def annotations_df(samples: list[dict]) -> pd.DataFrame:
    """One annotation row per image, in the format create_labels expects."""
    rows = []
    for row in samples:
        label = CLASS_TO_LABEL[row["class"]]
        x = y = None
        if label != NEGATIVE_LABEL:
            x, y = float(row["junction_x"]), float(row["junction_y"])
            if not (0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE):
                raise ValueError(
                    f"Junction ({x}, {y}) of {row['tem']} lies outside the image")
        rows.append({"image": image_stem(row), "label": label, "x": x, "y": y})
    return pd.DataFrame(rows)


def write_metadata(samples: list[dict], output_dir: Path, model_dir: Path,
                   seed: int):
    dataset_json = {
        "channel_names": {"0": "segmentationProbability"},
        "labels": get_label_mapping(combine_reversed_crossing=True),
        "numTraining": len(samples),
        "file_ending": FILE_ENDING,
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    mapping = [{
        "case": _sanitize_case_id(image_stem(row)),
        "hf_repo": HF_REPO_ID,
        "hf_tem": f"{SPLIT}/{row['tem']}",
        "class": row["class"],
        "fiber_min": row["fiber_min"],
        "fiber_max": row["fiber_max"],
        "junction_x": None if row["class"] == "negative" else float(row["junction_x"]),
        "junction_y": None if row["class"] == "negative" else float(row["junction_y"]),
        "segmentation_model": str(model_dir),
        "sampling_seed": seed,
    } for row in samples]
    with open(output_dir / "case_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)


def upload_to_hf(output_dir: Path, repo_id: str):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id, repo_type="dataset",
        folder_path=str(output_dir), path_in_repo=output_dir.name,
        commit_message=f"Add {output_dir.name}")
    print(f"Uploaded {output_dir} to https://huggingface.co/datasets/{repo_id} "
          f"({output_dir.name}/)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="trained nnU-Net segmentation model dir (plans.json, "
                        "dataset.json, fold_*/checkpoint_final.pth)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="nnU-Net raw dataset dir to create, e.g. "
                        "<nnUNet_raw>/Dataset022_JunctionDetectionSynthetic_SegProbOnly")
    parser.add_argument("--exclude-case-mapping", type=Path, default=None,
                        help="case_mapping.json of the synthetic segmentation dataset; "
                        "its train images are excluded from sampling")
    parser.add_argument("--n-images", type=int, default=N_IMAGES,
                        help=f"number of full images to sample (default {N_IMAGES})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"sampling seed (default {SEED})")
    parser.add_argument("--hf-repo", default=None,
                        help="HuggingFace dataset repo to upload the result to")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index (default 0)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the output directory if it already exists")
    args = parser.parse_args()

    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Model dir not found: {args.model_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) \
            and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} already exists and is not empty; "
            f"pass --overwrite to replace it")

    rows = load_manifest(SPLIT)
    if args.exclude_case_mapping is not None:
        excluded = load_excluded_images(args.exclude_case_mapping)
        rows = [row for row in rows if Path(row["tem"]).name not in excluded]
        print(f"Excluded {len(excluded)} {SPLIT} image(s) used for segmentation training")
    samples = sample_images(rows, args.n_images, args.seed)
    print_balance(samples, len(rows))

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    predictor, _ = initialize_nnunet_predictor(
        args.model_dir, device,
        folds=NNUNET_DEFAULT_FOLDS, checkpoint=NNUNET_DEFAULT_CHECKPOINT)

    init_dir(args.output_dir)
    (args.output_dir / NNUNET_IMAGES_DIR).mkdir()
    (args.output_dir / NNUNET_LABELS_DIR).mkdir()

    write_probability_images(samples, args.output_dir, predictor)
    create_labels(
        annotations_df(samples), args.output_dir,
        original_shapes={image_stem(row): (IMAGE_SIZE, IMAGE_SIZE) for row in samples},
        image_resize=IMAGE_RESIZE, combine_reversed_crossing=True)
    write_metadata(samples, args.output_dir, args.model_dir, args.seed)
    print(f"\nWrote nnU-Net dataset with {len(samples)} cases to {args.output_dir}")

    if args.hf_repo is not None:
        upload_to_hf(args.output_dir, args.hf_repo)


if __name__ == "__main__":
    main()
