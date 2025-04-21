import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm

import argparse
import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from PIL import Image

data_dir = os.environ["DATA_DIR"]

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d",
    "--data_dir",
    help="Select the raw data directory",
    default=data_dir,
)

args = parser.parse_args()

heatmapDim = (
    160,
    330,
)  # The ball can be snapped max of 80 ft from side field of play (sideline) and max of 330 from end field of play(endzone)


class TwoDimensionalTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_file):
        self.heatmaps, self.global_feats, self.labels = torch.load(tensor_file)

    def __len__(self):
        return self.heatmaps.size(0)

    def __getitem__(self, idx):
        return self.heatmaps[idx], self.global_feats[idx], self.labels[idx]


def print_nan_rows(df, name="DataFrame", max_rows=5):
    print(f"\n🔍 NaN Rows in '{name}':")

    nan_rows = df[df.isna().any(axis=1)]
    total_nans = len(nan_rows)

    if total_nans == 0:
        print("✅ No NaN rows found.")
    else:
        print(f"⚠️ Found {total_nans} rows with NaNs.")
        print(f"📋 Displaying first {min(max_rows, total_nans)} row(s):\n")
        print(nan_rows.head(max_rows))

        nan_summary = nan_rows.isna().sum()
        print("\n📊 Columns with NaNs:")
        print(nan_summary[nan_summary > 0])


def inspect_nans(df, df_name="DataFrame"):
    print(f"\n🔍 NaN Check — {df_name}")
    total_nans = df.isna().sum().sort_values(ascending=False)
    total = df.shape[0]
    with_nans = total_nans[total_nans > 0]

    if with_nans.empty:
        print("✅ No NaNs found.")
    else:
        print(f"⚠️ Found NaNs in {len(with_nans)} column(s):")
        print(with_nans)
        print(f"ℹ️ Total rows: {total}")
        print(f"🧼 Recommend dropping or filling NaNs before training.")


# Given dataset directory create an object that has functions to create a dataset
class NFL_Data_Preprocessing:
    def __init__(self, data_dir=None):
        print("Starting preprocessing")

        # Set the dataset directory
        if data_dir is None:
            data_dir = args.data_dir

        self.data_dir = data_dir
        self.raw_data_dir = os.path.join(self.data_dir, "raw")
        self.out_data_dir = os.path.join(self.data_dir, "ST_2D_Tensor")

        # Get all the Dataframe values
        games_df = pd.read_csv(os.path.join(self.raw_data_dir, "games.csv"))
        player_play_df = pd.read_csv(os.path.join(self.raw_data_dir, "player_play.csv"))
        players_df = pd.read_csv(os.path.join(self.raw_data_dir, "players.csv"))
        plays_df = pd.read_csv(os.path.join(self.raw_data_dir, "plays.csv"))

        # Were concatenating all the tracking files
        tracking_df = pd.concat(
            [
                pd.read_csv(file)
                for file in glob.glob(
                    os.path.join(self.raw_data_dir, "tracking_week_*.csv")
                )
            ],
            ignore_index=True,
        )

        # Do all the data processing with the global features
        games_features = games_df[["gameId", "week", "gameTimeEastern"]]
        plays_features = plays_df[
            [
                "gameId",
                "playId",
                "quarter",
                "down",
                "yardsToGo",
                "possessionTeam",
                "defensiveTeam",
                "gameClock",
                "preSnapHomeScore",
                "preSnapVisitorScore",
                "absoluteYardlineNumber",
            ]
        ]

        # Dataframe where each row is a time series (join games onto plays via gameId)
        self.global_feature_df = pd.merge(
            plays_features, games_features, on="gameId", how="left"
        )

        # Merge in the play direction
        filtered = tracking_df[
            (tracking_df["frameType"] == "SNAP")
            & (tracking_df["displayName"] == "football")
        ][["gameId", "playId", "playDirection"]]

        self.global_feature_df = pd.merge(
            self.global_feature_df,
            filtered,
            on=["gameId", "playId"],
            how="left",
        )

        self.global_feature_df["gameTimeEastern"] = pd.to_datetime(
            self.global_feature_df["gameTimeEastern"], format="%H:%M:%S"
        ).dt.time

        def time_to_float(t):
            return (t.hour * 3600 + t.minute * 60 + t.second) / 86400

        self.global_feature_df["gameTimeEastern"] = self.global_feature_df[
            "gameTimeEastern"
        ].apply(time_to_float)

        # Do all the data processing with the local features
        print("Feature engineering")

        # List of Dataframes where each row is a frame (Join player_play and players onto tracking via nflId)
        players_features = players_df[["nflId", "height", "weight", "position"]]
        tracking_features = tracking_df[
            [
                "gameId",
                "playId",
                "nflId",
                "frameId",
                "frameType",
                "displayName",
                "time",
                "club",
                "playDirection",
                "x",
                "y",
                "s",
                "a",
                "dis",
                "o",
                "dir",
                "event",
            ]
        ]

        # Will be by frame
        self.local_features_df = pd.merge(
            tracking_features, players_features, on="nflId", how="left"
        )

        ## coordinates and heading need to be remapped
        self.local_features_df["x"] = np.where(
            self.local_features_df["playDirection"] == "left",
            120
            - self.local_features_df[
                "x"
            ],  # Flip horizontally across the 120-yard field
            self.local_features_df["x"],
        )

        self.local_features_df["y"] = np.where(
            self.local_features_df["playDirection"] == "left",
            160 / 3
            - self.local_features_df[
                "y"
            ],  # Flip across width of the field (160/3 = ~53.33 yards)
            self.local_features_df["y"],
        )

        ### perform sin and cos on heading and o
        self.local_features_df["o"] = np.deg2rad(self.local_features_df["o"])
        self.local_features_df["dir"] = np.deg2rad(self.local_features_df["dir"])
        self.local_features_df["sin_o"] = np.sin(self.local_features_df["o"])
        self.local_features_df["cos_o"] = np.cos(self.local_features_df["o"])
        self.local_features_df["sin_dir"] = np.sin(self.local_features_df["dir"])
        self.local_features_df["cos_dir"] = np.cos(self.local_features_df["dir"])

        # Step 1: Get ball positions per frame
        ball_position_df = tracking_df[
            (tracking_df["displayName"] == "football")
            & (tracking_df["frameType"] == "SNAP")
        ][["gameId", "playId", "x", "y"]].rename(columns={"x": "ball_x", "y": "ball_y"})

        # Step 2: Merge ball position into full local_features_df
        self.local_features_df = pd.merge(
            self.local_features_df,
            ball_position_df,
            on=["gameId", "playId"],
            how="left",
        )

        # Step 3: Subtract ball position from player position (to get relative coords)
        self.local_features_df["rel_x"] = (
            self.local_features_df["x"] - self.local_features_df["ball_x"]
        )
        self.local_features_df["rel_y"] = (
            self.local_features_df["y"] - self.local_features_df["ball_y"]
        )

        # Set the position to football to avoid NA
        self.local_features_df["position"] = self.local_features_df["position"].fillna(
            self.local_features_df["displayName"].apply(
                lambda x: "football" if x == "football" else np.nan
            )
        )

        ## Need to make height in inches
        def convert_height(h):
            try:
                feet, inches = map(int, h.split("-"))
                return feet * 12 + inches
            except:
                return None  # or np.nan

        self.local_features_df["height"] = self.local_features_df["height"].apply(
            convert_height
        )

        def convert_seconds(time):
            try:
                min, sec = map(int, time.split(":"))
                return min * 60 + sec
            except:
                return None  # or np.nan

        self.global_feature_df["gameClock"] = self.global_feature_df["gameClock"].apply(
            convert_seconds
        )

        # Going right increases absoluteYardlineNumber (football is in direction of the offense) Going left decreases absolute yardline number
        self.global_feature_df["absoluteYardlineNumber"] = np.where(
            self.global_feature_df["playDirection"] == "left",
            100 - self.global_feature_df["absoluteYardlineNumber"],
            self.global_feature_df["absoluteYardlineNumber"],
        )

        ## All the nominal labels used (all should be global except for routes)

        ## label routeRan needs to be set by left to right
        # 1 QB label, 5 WR label, 3 RB label, 3 TE label

        # Join data sources
        temp_df = pd.merge(
            player_play_df[player_play_df["wasRunningRoute"] == 1],
            tracking_df[tracking_df["frameType"] == "SNAP"],
            how="left",
            on=["gameId", "playId", "nflId"],
        )

        # Add position info
        temp_df = pd.merge(
            temp_df, players_df[["nflId", "position"]], how="left", on="nflId"
        )

        # Ensure routeRan is always a string
        temp_df["routeRan"] = temp_df["routeRan"].fillna("NA").astype(str)
        temp_df["position"] = temp_df["position"].fillna("UNKNOWN")

        all_keys = [
            "routeRanQB0",
            "routeRanWR0",
            "routeRanWR1",
            "routeRanWR2",
            "routeRanWR3",
            "routeRanWR4",
            "routeRanRB0",
            "routeRanRB1",
            "routeRanRB2",
            "routeRanTE0",
            "routeRanTE1",
            "routeRanTE2",
        ]

        def extract_routes_per_play(group):
            group = group.sort_values(by="y")
            route_data = {key: "NA" for key in all_keys}  # Initialize with "NA"
            position_counter = {"QB": 0, "WR": 0, "RB": 0, "TE": 0}

            for _, row in group.iterrows():
                pos = row["position"]
                if pos not in position_counter:
                    continue  # skip unexpected positions

                idx = position_counter[pos]
                key = f"routeRan{pos}{idx}"
                if key in route_data:  # only fill valid keys
                    route_data[key] = row["routeRan"]
                    position_counter[pos] += 1

            return pd.Series(route_data)

        # Will come out to labels per play.
        self.labels = (
            temp_df.groupby(["playId", "gameId"])
            .apply(extract_routes_per_play)
            .reset_index()
        )

        route_cols = [col for col in self.labels.columns if col.startswith("routeRan")]
        self.labels[route_cols] = self.labels[route_cols].fillna("NA").astype(str)

        self.labels = pd.merge(
            self.labels,
            plays_df[
                [
                    "playId",
                    "gameId",
                    "offenseFormation",
                    "receiverAlignment",
                    "playAction",
                    "dropbackType",
                    "passLocationType",
                    "passTippedAtLine",
                    "unblockedPressure",
                    "qbSpike",
                    "qbKneel",
                    "qbSneak",
                    "rushLocationType",
                    "isDropback",
                    "pff_runConceptPrimary",
                    "pff_runConceptSecondary",
                    "pff_runPassOption",
                    "pff_passCoverage",
                    "pff_manZone",
                ]
            ],
            how="left",
            on=["playId", "gameId"],
        )

        # print("Global features", self.global_feature_df.columns.tolist())
        # print(self.global_feature_df.head(5))
        # print("Local features", self.local_features_df.columns.tolist())
        # print(self.local_features_df.head(5))
        # print("Labels", self.labels.columns.tolist())
        # print(self.labels.head(5))

        print("Applying encodings")

        # === Multi-label boolean columns ===
        multilabel_cols = [
            "playAction",
            "passTippedAtLine",
            "unblockedPressure",
            "qbSpike",
            "qbKneel",
            "qbSneak",
            "isDropback",
            "pff_runPassOption",
        ]
        self.labels[multilabel_cols] = (
            self.labels[multilabel_cols].fillna(False).astype(int)
        )

        # === One-hot categorical columns ===
        onehot_cols = [
            "offenseFormation",
            "receiverAlignment",
            "dropbackType",
            "passLocationType",
            "rushLocationType",
            "pff_runConceptPrimary",
            "pff_runConceptSecondary",
            "pff_passCoverage",
            "pff_manZone",
        ]
        self.labels[onehot_cols] = self.labels[onehot_cols].fillna("Unknown")
        onehot_encoded = pd.get_dummies(self.labels[onehot_cols], dummy_na=False)

        # === Route columns (e.g. routeRanWR0, routeRanRB1, etc.) ===
        route_cols = [col for col in self.labels.columns if col.startswith("routeRan")]
        self.labels[route_cols] = self.labels[route_cols].fillna("NA")
        route_encoded = pd.get_dummies(self.labels[route_cols], dummy_na=False)

        # === Combine everything cleanly ===
        self.encoded_labels = pd.concat(
            [
                self.labels[["gameId", "playId"]],  # Optional: keep tracking IDs
                self.labels[multilabel_cols],
                onehot_encoded,
                route_encoded,
            ],
            axis=1,
        )
        
        # Global encoding
        
        # One-hot encode both team columns
        team_ohe = pd.get_dummies(
            self.global_feature_df[["possessionTeam", "defensiveTeam"]],
            prefix=["possTeam", "defTeam"]
        )

        # Drop original team columns and join one-hot columns
        self.global_feature_df = pd.concat(
            [
                self.global_feature_df.drop(columns=["possessionTeam", "defensiveTeam"]),
                team_ohe
            ],
            axis=1
        )
        

        # === NaN cleanup for local features ===
        cols_to_check = [
            "height",
            "weight",
            "s",
            "a",
            "dis",
            "sin_o",
            "cos_o",
            "sin_dir",
            "cos_dir",
            "rel_x",
            "rel_y",
        ]
        self.local_features_df[cols_to_check] = self.local_features_df[
            cols_to_check
        ].fillna(0)

        # Drop columns that are not going to be used
        self.local_features_df.drop(["o", "dir"], axis=1, inplace=True)
        self.global_feature_df.drop(["playDirection"], axis=1, inplace=True)

        print(tracking_df[tracking_df["frameType"] == "SNAP"]["playId"].nunique())
        print(self.local_features_df["playId"].nunique())

        print_nan_rows(
            self.local_features_df[["gameId", "playId", "ball_y"]],
            "Ball NaNs",
            max_rows=10,
        )
        

    @staticmethod
    def create_heatmap(frame_df, feature_cols, grid_shape):
        C = len(feature_cols)
        heatmap = np.zeros((C, *grid_shape), dtype=np.float32)

        for _, row in frame_df.iterrows():
            x, y = int(row["grid_x"]), int(row["grid_y"])
            if 0 <= x < grid_shape[1] and 0 <= y < grid_shape[0]:
                for i, col in enumerate(feature_cols):
                    heatmap[i, y, x] += row[col]

        return heatmap  # shape: [C, H, W]

    def create_ST_Dataset_2D_Tensors(self, grid_h=160 * 2, grid_w=330 * 2):
        center_x = grid_w // 2
        center_y = grid_h // 2

        # apply values to a grid
        self.local_features_df["grid_x"] = np.clip(
            (self.local_features_df["rel_x"].round().astype(int) + center_x),
            0,
            grid_w - 1,
        )

        self.local_features_df["grid_y"] = np.clip(
            (self.local_features_df["rel_y"].round().astype(int) + center_y),
            0,
            grid_h - 1,
        )
        
        # Encoding positions here otherwise memory would expload
        positions = [
            "QB",
            "RB",
            "WR",
            "TE",
            "T",
            "SS",
            "OLB",
            "NT",
            "MLB",
            "LB",
            "ILB",
            "G",
            "FS",
            "FB",
            "DT",
            "DE",
            "CB",
            "C",
        ]
        for pos in positions:
            self.local_features_df[f"pos_{pos}"] = (
                self.local_features_df["position"] == pos
            ).astype(float)


        position_channels = [f"pos_{pos}" for pos in positions]

        # Group by play
        grouped = self.local_features_df.groupby(["gameId", "playId"])
        
        for (gameId, playId), play_df in tqdm(grouped, desc="Creating Heatmaps"):
            play_heatmaps = []
            save_path = os.path.join(
                self.data_dir, "ST_2D_Tensor", f"{gameId}_{playId}.pt"
            )
            print(len(play_df))
            if os.path.exists(save_path):
                continue
            # if(len(play_df) > 4600):
            #     continue
            frame_groups = list(play_df.groupby("frameId"))
            for _, frame_df in frame_groups[::2]:  # Step size 2
                heatmap = self.create_heatmap(
                    frame_df,
                    feature_cols=[
                        "s",
                        "a",
                        "dis",
                        "height",
                        "weight",
                        "sin_o",
                        "cos_o",
                        "sin_dir",
                        "cos_dir",
                        *position_channels,
                    ],
                    grid_shape=(grid_h, grid_w),
                )
                play_heatmaps.append(heatmap)

            # Stack to tensor: [T, C, H, W]
            play_tensor = torch.tensor(np.stack(play_heatmaps), dtype=torch.bfloat16)
            
             # === Lookup Global Features ===
            global_row = self.global_feature_df[
                (self.global_feature_df["gameId"] == gameId) &
                (self.global_feature_df["playId"] == playId)
            ]

            if global_row.empty:
                print(f"⚠️ Skipping play {gameId}-{playId}: No global features.")
                continue
            
            
            global_np = global_row.drop(columns=["gameId", "playId"]).select_dtypes(include=[np.number, np.bool_]).values.astype(np.float32)
            global_tensor = torch.tensor(global_np[0], dtype=torch.bfloat16)


            # === Lookup Labels ===
            label_row = self.encoded_labels[
                (self.encoded_labels["gameId"] == gameId) &
                (self.encoded_labels["playId"] == playId)
            ]

            if label_row.empty:
                print(f"⚠️ Skipping play {gameId}-{playId}: No label found.")
                continue

            label_np = label_row.drop(columns=["gameId", "playId"]).select_dtypes(include=[np.number, np.bool_]).values.astype(np.float32)
            label_tensor = torch.tensor(label_np[0], dtype=torch.bfloat16)

            # Save to disk
            torch.save((play_tensor, global_tensor, label_tensor), save_path)

    def tabular(self):

        raise NotImplementedError


if __name__ == "__main__":
    data_preprocessor = NFL_Data_Preprocessing()
    data_preprocessor.create_ST_Dataset_2D_Tensors()
