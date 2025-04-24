# Run higher learning rate
/opt/conda/bin/python /app/train_eval.py --lr 1e-2 --epochs 15 --name "higher_lr_more_epochs" --device 'cuda:0' --percentiles 100

# More epochs
/opt/conda/bin/python /app/train_eval.py --epochs 15 --name "more_epochs" --device 'cuda:1' --percentiles 100
