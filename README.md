# NFL Play prediction

To run this, edit the `docker-compose.yml` to how you want to configure the directories including the data directory (DATA_DIR). requirements.txt should have all the libraries.

Outputs directory contains all the output information in csv format by `outputs/name` and will also have (gitignored) checkpoints in `outputs/name/checkpoints`

## Code Directory

Using the containers default conda environment

run `run.sh` to train the model on the data found in the (DATA_DIR)

- will train off of to `[DATA_DIR]/ST_2D_Tensor`

run `preprocessing.py` to create the dataset (NOTE: Will use 60GB of RAM)

- expects `[DATA_DIR]/raw` to contain each of the CSVs from https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data

- will write to `[DATA_DIR]/ST_2D_Tensor`


`st_dataset.py` is the data implementation

`cnn_lstm.py` is the model

`sequence_lengths.csv` is metadata from preprocessing

`label_metadata.json` is metadata created for the labels and onehot relationship

`nohup1.out` and `nohup2.out` show the training cmd on separate GPUs; they are running the first and second script in `run.sh`, respectively

# Previous code outputs
