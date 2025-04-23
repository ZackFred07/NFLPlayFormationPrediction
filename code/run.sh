# Run with default numbers
/opt/conda/bin/python /app/train_eval.py --name "default"

# Run with lower learning rate
/opt/conda/bin/python /app/train_eval.py --lr 1e-4 --name "lower_lr"

# Run with no percentiles
/opt/conda/bin/python /app/train_eval.py --percentiles 100 --name "fullset"