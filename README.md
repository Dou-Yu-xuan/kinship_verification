# Kinship verification

This is a GitHub repository of my Bachelor Thesis at UCU, named "Ablation study of the approaches for the kinship verification". Here, you will find the code of my study, overall information about the dataset and its splits, models for all experiments, and their checkpoints.




## Dataset
In my research, I used a part of the [5-fold variation](https://drive.google.com/file/d/1vZ5NujjO172rtqFEouGoYrngHLyLssWn/view?usp=sharing) of the [Families in the Wild](https://web.northeastern.edu/smilelab/fiw/) (FIW) dataset. 

You can use [dataset_split.ipynb](https://github.com/franchukpetro/kinship_verification/blob/master/siamese_keras/dataset_split.ipynb) Jupyter Notebook to split the dataset in the same way as me. Also, [here](https://drive.google.com/drive/folders/1kAZ-fAgv9rnrraNq2VvSvsj7onWn1AaP?usp=sharing) you can find .csv files with those dataset entries, which were used during training, validation and testing.


## Models
For experiments, I have used several pretrained models from the [deepface](https://github.com/serengil/deepface) and [keras-vggface](https://github.com/rcmalli/keras-vggface) repositories. You also can find them all together in [Google Drive](https://drive.google.com/drive/folders/1EcIyVsnXIVjeb25DkV61WafAwaFnfaCw?usp=sharing).

## Checkpoints
To reproduce the results of all the experiments, you can download checkpoints [here](https://drive.google.com/drive/folders/1NQlKRcP2tltS2IGqrKyyCCje-RVucTt9?usp=sharing).


## Installation and usage

In the [siamese_keras](https://github.com/franchukpetro/kinship_verification/tree/master/siamese_keras) is an implementation in Keras framework, while in [siamese_vggface](https://github.com/franchukpetro/kinship_verification/tree/master/siamese_vggface) - in PyTorch. I was not able to train the pretrained models in the PyTorch, so, please, consider using first folder.

Firstly, please install all necessary packages by running

```bash
pip install -r requirements.txt
```
Secondly, download all model weights from the [Google Drive](https://drive.google.com/drive/folders/1EcIyVsnXIVjeb25DkV61WafAwaFnfaCw?usp=sharing) and place directory with them in the project folder.


