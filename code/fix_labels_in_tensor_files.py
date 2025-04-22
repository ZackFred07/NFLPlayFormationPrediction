import glob
import os
import json
import torch
from tqdm import tqdm


def create_label_remap(label_metadata_path):
    with open(label_metadata_path) as f:
        label_meta = json.load(f)
    label_names = list(label_meta.keys())

    merged_labels = []
    old_to_new = {}
    seen = {}
    merge_logic = set()

    for i, name in enumerate(label_names):
        if name == "rushLocationType_Unknown":
            old_to_new[i] = None
            continue

        canonical_name = (
            "passLocationType_Unknown"
            if name in ["passLocationType_UNKNOWN", "passLocationType_Unknown"]
            else name
        )

        if canonical_name not in seen:
            new_idx = len(merged_labels)
            merged_labels.append(canonical_name)
            seen[canonical_name] = new_idx
        else:
            new_idx = seen[canonical_name]
            merge_logic.add(new_idx)

        old_to_new[i] = new_idx

    return old_to_new, merged_labels, merge_logic


def remap_label_tensor(label_tensor, old_to_new_map, num_new_labels, merge_logic):
    new_tensor = torch.zeros(num_new_labels, dtype=label_tensor.dtype)
    for old_idx, new_idx in old_to_new_map.items():
        if new_idx is None:
            continue
        val = label_tensor[old_idx].item()
        if new_idx in merge_logic:
            new_tensor[new_idx] = new_tensor[new_idx] | val
        else:
            new_tensor[new_idx] = val
    return new_tensor


def fix_pt_folder_inplace(directory, label_metadata_path):
    files = glob.glob(os.path.join(directory, "**", "*.pt"), recursive=True)

    # Remap config
    old_to_new, new_label_names, merge_logic = create_label_remap(label_metadata_path)
    num_new_labels = len(new_label_names)

    skipped_files = 0
    total_samples_before = 0
    total_samples_after = 0

    for path in tqdm(files, desc="Fixing label tensors (in-place)"):
        try:
            data = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"❌ Failed to load file: {path} — {e}")
            continue

        if not (isinstance(data, tuple) and len(data) == 3):
            print(f"⚠️ Skipping invalid format: {path}")
            continue

        heatmaps, global_feats, labels = data

        valid_heatmaps = []
        valid_globals = []
        valid_labels = []

        for i in range(len(labels)):
            label = labels[i]
            if label.ndim != 1:
                print(f"⚠️ Sample {i} in {os.path.basename(path)} has invalid label shape {label.shape}, skipping.")
                continue

            new_label = remap_label_tensor(label, old_to_new, num_new_labels, merge_logic)
            valid_heatmaps.append(heatmaps[i])
            valid_globals.append(global_feats[i])
            valid_labels.append(new_label)

        total_samples_before += len(labels)
        total_samples_after += len(valid_labels)

        if not valid_labels:
            print(f"❌ All samples in {os.path.basename(path)} were invalid. Skipping file.")
            skipped_files += 1
            continue

        new_data = (
            torch.stack(valid_heatmaps),
            torch.stack(valid_globals),
            torch.stack(valid_labels),
        )

        torch.save(new_data, path)

    print(f"\n✅ Finished processing {len(files)} files.")
    print(f"📉 Skipped {skipped_files} file(s) with no valid samples.")
    print(f"📊 Samples before: {total_samples_before}, after cleaning: {total_samples_after}")

    # Save label names used in new data
    label_output_path = os.path.join(directory, "label_names_fixed.json")
    with open(label_output_path, "w") as f:
        json.dump(new_label_names, f, indent=2)
    print(f"📝 Saved updated label names to: {label_output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fix label inconsistencies in .pt tensor files (in-place)")
    parser.add_argument("--directory", default=os.environ["DATA_DIR"], help="Directory containing .pt files and label metadata")
    parser.add_argument("--label_metadata", default="label_metadata.json", help="Path to original label_metadata.json")

    args = parser.parse_args()

    fix_pt_folder_inplace(
        directory=args.directory,
        label_metadata_path=args.label_metadata
    )


if __name__ == "__main__":
    main()
