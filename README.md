## How To Use

### Clone

Clone this GitHub repository:

```
git clone https://github.com/G-U-N/PyCIL.git
cd PyCIL
```

### Dependencies

1. [torch 1.81](https://github.com/pytorch/pytorch)
2. [torchvision 0.6.0](https://github.com/pytorch/vision)
3. [tqdm](https://github.com/tqdm/tqdm)
4. [numpy](https://github.com/numpy/numpy)
5. [scipy](https://github.com/scipy/scipy)
6. [quadprog](https://github.com/quadprog/quadprog)
7. [POT](https://github.com/PythonOT/POT)

### Run experiment

1. Edit the `pbpc.json` file for global settings.
2. Edit the hyperparameters in the corresponding `models/pbpc.py` file.
3. Run:

```bash
python main.py --config=./exps/pbpc/pbpc.json
```

### Datasets

We have implemented the pre-processing of `CIFAR100`, `Tiny-ImageNet,`, `CUB,`, `Imagenet-R,`, and `Imagenet-Sub`. 

## Acknowledgments

We thank the following repos providing helpful components/functions in our work.

- [PyCIL](https://github.com/G-U-N/PyCIL?tab=readme-ov-file)


