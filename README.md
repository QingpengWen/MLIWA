# MLIWA-Chinese-SLU
***Note: This paper is a coursework paper for a course design on text information processing***

English|[中文](README_CN.md)

This repository contains the PyTorch implementation of the paper: 

****A Chinese Spoken Language Understanding Model Based on Multilevel Iteration and Word Adapters****. 

**Author:** [Qingpeng Wen](mailto:wqp@mail2.gdut.edu.cn). 

**Instructor:** [Yiyang Yang](mailto:yyygou_yang@163.com)

**Special thanks to:** Prof. [Bi Zeng](mailto:zb9215@gdut.edu.cn), Mentor [Pengfei Wei](mailto:wpf@gdut.edu.cn)
## Architecture

<img src="Figures\fig.png">

## Requirements
Our code is based on Python 3.7.6 and PyTorch 1.1. Requirements are listed as follows:
> - torch==1.1.0
> - transformers==2.4.1
> - numpy==1.18.1
> - tqdm==4.42.1
> - seqeval==0.0.12

We highly suggest you using [Anaconda](https://www.anaconda.com) to manage your python environment.

## Datasets
The two publicly available datasets used in this project, CAIS and SMP-ECDT datasets, are available [here](https://github.com/AaronTengDeChuan/MLWA-Chinese-SLU/tree/main/data).

For the home-made dataset used in the project, **the Genshin Impact Encyclopedic Dictionary**, it can be obtained by running the code in the **Work-Demo** folder.

## Quick start
The script **train.py** acts as a main function to the project, you can run the experiments by replacing the unspecified options in the following command with the corresponding values.

```shell
    CUDA_VISIBLE_DEVICES=$1 python train.py -dd ${dataDir} -sd ${saveDir} -u -bs 16 -dr 0.3 \ 
        -ced 128 -wed 128 -ehd 512 -aod 128 -sed 32 -sdhd 64
```

Due to some stochastic factors, It's necessary to slightly tune the hyper-parameters using grid search. If you have any question, please issue the project or email [me](mailto:wqp@mail2.gdut.edu.cn) and we will reply you soon.


## Issues/PRs/Questions 

Feel free to contact [me](mailto:wqp@mail2.gdut.edu.cn) for any question or create issues/prs.

## Acknowledgement
This work was supported by ***ESAC Lab***, in part by the National Science Foundation of China under Grant 62172111, in part by the Natural Science Foundation of Guangdong Province under Grant 2019A1515011056, in part by the Key technology project of Shunde District under Grant 2130218003002.
