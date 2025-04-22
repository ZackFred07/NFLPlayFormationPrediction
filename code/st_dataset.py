import torch
import os
import numpy as np

class TwoDimensionalTensorDataset(torch.utils.data.Dataset):
        # TODO: Dataset mistakes: passLocationType_UNKNOWN (index 36) and passLocationType_Unknown (index 35) need to be one; rushLocationType_Unknown (index 37) needs to be removed

    def __init__(self, tensor_path_or_dir):
        self.max_timesteps = None  # use full length by default
        self.lazy_mode = os.path.isdir(tensor_path_or_dir)

        if self.lazy_mode:
            self.file_paths = [
                os.path.join(tensor_path_or_dir, fname)
                for fname in sorted(os.listdir(tensor_path_or_dir))
                if fname.endswith(".pt")
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

    def __len__(self):
        if self.lazy_mode:
            return len(self.index_map)
        return self.heatmaps.size(0)

    def __getitem__(self, idx):
        if self.lazy_mode:
            file_idx, local_idx = self.index_map[idx]
            h, g, l = torch.load(self.file_paths[file_idx], map_location="cpu")
            heatmap_seq = h[local_idx]
            global_feat = g[local_idx]
            label = l[local_idx]
        else:
            heatmap_seq = self.heatmaps[idx]
            global_feat = self.global_feats[idx]
            label = self.labels[idx]

        if self.max_timesteps is not None:
            heatmap_seq = heatmap_seq[:self.max_timesteps]

        return heatmap_seq, global_feat, label

    def truncate_timesteps(self, max_timesteps):
        """
        Sets max_timesteps to truncate the time dimension in __getitem__.
        Handles variable-length sequences gracefully.
        """
        self.max_timesteps = max_timesteps
        return self

    def truncate_by_percentile(self, percentile):
        """
        Sets max_timesteps using a percentile of all sequence lengths.
        """
        if not self.seq_lens:
            raise RuntimeError("Sequence lengths not available.")
        self.max_timesteps = int(np.percentile(self.seq_lens, percentile))
        return self
