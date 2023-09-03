# Individual privacy accounting for DP-SGD

## Build the environment
```pip install -r requirements.txt```

## Run DP-SGD with individual privacy accounting 

This command trains a resnet20 model on CIFAR-10. The training takes ~1.5 hours with a single A100 GPU.
```python main.py --private --sess example_exp --sigma 2.2 --n_epoch 200 --clip 15 ```

After training, you can visualize the histogram of individual privacy and estimation error. The figures are saved at `figs`.
```python visualization.py --sess example_exp```
