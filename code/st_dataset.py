import pandas as pd
import torch
import os
import json
import numpy as np
from tqdm import tqdm

class TwoDimensionalTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_path_or_dir=os.path.join(os.environ["DATA_DIR"], "ST_2D_Tensor"), label_metadata_path="label_metadata.json"):
        self.max_timesteps = None
        self.lazy_mode = os.path.isdir(tensor_path_or_dir)
        self.seq_len_cache="sequence_lengths.csv"
        if self.lazy_mode:
            # Get all the files
            self.file_paths = [
                os.path.join(tensor_path_or_dir, f)
                for f in sorted(os.listdir(tensor_path_or_dir)) if f.endswith(".pt")
            ]
            if os.path.exists(self.seq_len_cache):
                # Load precomputed sequence lengths
                self.seq_lens = pd.read_csv(self.seq_len_cache)["length"].tolist()
            else:
                # Compute and save sequence lengths
                self.seq_lens = []
                file_lens = []
                for path in tqdm(self.file_paths, desc="Initialize sequence lengths"):
                    try:
                        h, _, _ = torch.load(path, map_location="cpu")
                        seq_len = h.shape[0]
                        self.seq_lens.append(seq_len)
                        file_lens.append({"file": os.path.basename(path), "length": seq_len})
                    except Exception as e:
                        print(f"Skipping file {path} due to error: {e}")
                pd.DataFrame(file_lens).to_csv(self.seq_len_cache, index=False)
        else:
            # Load all at once
            self.heatmaps, self.global_feats, self.labels = torch.load(tensor_path_or_dir)
            self.seq_lens = [hm.shape[0] for hm in self.heatmaps]

    def __len__(self):
        return len(self.file_paths) if self.lazy_mode else self.heatmaps.size(0)

    def __getitem__(self, idx):
        if self.lazy_mode:
            # Load as we need to
            h, g, l = torch.load(self.file_paths[idx], map_location="cpu")
            heatmap_seq, global_feat, label = h, g, l
        else:
            # Load all at once
            heatmap_seq, global_feat, label = self.heatmaps[idx], self.global_feats[idx], self.labels[idx]

        if self.max_timesteps is not None:
            heatmap_seq = heatmap_seq[:self.max_timesteps]

        return heatmap_seq, global_feat, label

    def truncate_timesteps(self, max_timesteps):
        self.max_timesteps = max_timesteps
        return self

    def truncate_by_percentile(self, percentile):
        if not self.seq_lens:
            raise RuntimeError("Sequence lengths not available.")
        self.max_timesteps = int(np.percentile(self.seq_lens, percentile))
        return self
