## CIRL

This repo provides a demo for the CVPR 2022 paper "Causality Inspired Representation Learning for Domain Generalization" on the PACS dataset.

### Requirements

* `Python 3.6`
* `Pytorch 1.1.0`

### Training from scratch 
Please first download the PACS dataset from http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017. Then update the files with suffix `_train.txt` and `_val.txt` in `data/datalists` for each domain, following styles below:

```
/home/user/data/images/PACS/kfold/art_painting/dog/pic_001.jpg 0
/home/user/data/images/PACS/kfold/art_painting/dog/pic_002.jpg 0
/home/user/data/images/PACS/kfold/art_painting/dog/pic_003.jpg 0
...
```

Please make sure you are using the official train-val-split. Once the data is prepared, then remember to update the path of train&val files and output logs in `shell_train.py`:

```
input_dir = 'path/to/train/files'
output_dir = 'path/to/output/logs'
```

Then running the code:

```
python shell_train.py -d=art_painting
```

Use the argument `-d` to specify the held-out target domain.



### Evaluation
After training the model, firstly create directory `ckpt/` and drag your model under it.  For running the evaluation code, please update the files with suffix `_test.txt` in `data/datalists` for each domain, following  the same styles as the train/val files above. 

Then update the path of test files and output logs in `shell_test.py`:

``` 
input_dir = 'path/to/test/files'
output_dir = 'path/to/output/logs'
```

then simply run:

```
 python shell_test.py -d=art_painting
```

You can use the argument `-d` to specify the held-out target domain.

### Acknowledgements
Some codes are adapted from [FACT](https://github.com/MediaBrain-SJTU/FACT). We thank them for their excellent projects.

### Contact
If you have any problem about our code, feel free to contact fangruilv@bit.edu.cn or describe your problem in Issues.