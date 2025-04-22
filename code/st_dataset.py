import torch
import os
import json
import numpy as np

class TwoDimensionalTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_path_or_dir=os.path.join(os.environ["DATA_DIR"], "ST_2D_Tensor"), label_metadata_path="label_metadata.json"):
        self.max_timesteps = None
        self.lazy_mode = os.path.isdir(tensor_path_or_dir)

        if self.lazy_mode:
            self.file_paths = [
                os.path.join(tensor_path_or_dir, f)
                for f in sorted(os.listdir(tensor_path_or_dir)) if f.endswith(".pt")
            ]
            self.index_map = []
            self.seq_lens = []
            for file_idx, path in enumerate(self.file_paths):
                h, _, _ = torch.load(path, map_location="cpu")
                for i in range(h.size(0)):
                    self.index_map.append((file_idx, i))
                    self.seq_lens.append(h[i].shape[0])
        else:
            self.heatmaps, self.global_feats, self.labels = torch.load(tensor_path_or_dir)
            self.seq_lens = [hm.shape[0] for hm in self.heatmaps]

        # Setup label fixing
        self.fix_indices = None
        if label_metadata_path:
            with open(label_metadata_path) as f:
                meta = json.load(f)
            label_names = list(meta.keys())
            try:
                idx_or = label_names.index("passLocationType_Unknown")
                idx_and = label_names.index("passLocationType_UNKNOWN")
                idx_drop = label_names.index("rushLocationType_Unknown")
                self.fix_indices = {"or_into": idx_or, "or_from": idx_and, "drop": idx_drop}
            except ValueError:
                print("⚠️ Could not find all target label indices — skipping label fix.")

    def __len__(self):
        return len(self.index_map) if self.lazy_mode else self.heatmaps.size(0)

    def __getitem__(self, idx):
        if self.lazy_mode:
            file_idx, local_idx = self.index_map[idx]
            h, g, l = torch.load(self.file_paths[file_idx], map_location="cpu")
            heatmap_seq, global_feat, label = h[local_idx], g[local_idx], l[local_idx]
        else:
            heatmap_seq, global_feat, label = self.heatmaps[idx], self.global_feats[idx], self.labels[idx]

        if self.max_timesteps is not None:
            heatmap_seq = heatmap_seq[:self.max_timesteps]

        if self.fix_indices:
            # Bitwise OR passLocationType_UNKNOWN into passLocationType_Unknown
            i_to = self.fix_indices["or_into"]
            i_from = self.fix_indices["or_from"]
            i_drop = self.fix_indices["drop"]
            label[i_to] = label[i_to] | label[i_from]

            # Drop passLocationType_UNKNOWN and rushLocationType_Unknown
            keep_indices = [
                i for i in range(len(label)) if i not in {i_from, i_drop}
            ]
            label = label[keep_indices]

        return heatmap_seq, global_feat, label

    def truncate_timesteps(self, max_timesteps):
        self.max_timesteps = max_timesteps
        return self

    def truncate_by_percentile(self, percentile):
        if not self.seq_lens:
            raise RuntimeError("Sequence lengths not available.")
        self.max_timesteps = int(np.percentile(self.seq_lens, percentile))
        return self
