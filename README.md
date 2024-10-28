# Multiplierless MLP Using Successive Vector Approximation in Post-Training Quantization

Run experiments using the `main.py` file. You'll need Python >= 3.10, `numpy`, `pytorch`, `numba`, and `matplolib`.

You must first train a model using:
```{bash}
$ python main.py -m train
```
Depending on the machine and whether you use GPU, this should take a few hours.

Fig. 2 can be reproduced with the following:
```{bash}
$ python main.py -m run
```

For Fig. 3, run the following command:
```{bash}
$ python main.py -m run2
```

For Fig. 4, you must have the quantized models from `-m run` and run the command:
```{bash}
$ python main.py -m run3
```
