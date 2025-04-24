# # Run with default numbers
# /opt/conda/bin/python /app/train_eval.py --name "default"

# # Run with lower learning rate
# /opt/conda/bin/python /app/train_eval.py --lr 1e-4 --name "lower_lr"

# # Run with no percentiles
# /opt/conda/bin/python /app/train_eval.py --percentiles 100 --name "fullset"

# Run higher learning rate
# /opt/conda/bin/python /app/train_eval.py --lr 1e-2 --epochs 15 --name "higher_lr_more_epochs" --device 'cuda:0' --percentiles 100

# More epochs
# /opt/conda/bin/python /app/train_eval.py --epochs 15 --name "more_epochs" --device 'cuda:1' --percentiles 100
