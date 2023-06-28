# MLIWA-Chinese-SLU
***注意：本文是一篇关于文本信息处理的课程设计的课程作业论文***

[English](README.md) | 中文

这个资源库包含该论文的PyTorch实现：

****基于多级迭代与词适配器的中文口语理解模型****

**作者：** [温庆鹏](mailto:wqp@mail2.gdut.edu.cn)

**指导教师：** [杨易扬](mailto:yyygou_yang@163.com)

**特别感谢：** [曾碧](mailto:zb9215@gdut.edu.cn)教授，[魏鹏飞](mailto:wpf@gdut.edu.cn)导师

## 模型框架
<img src="Figures\fig.png">
## 环境要求
我们的代码基于Python 3.7.6和PyTorch 1.1. 相关环境资源包要求如下:
> - torch==1.1.0
> - transformers==2.4.1
> - numpy==1.18.1
> - tqdm==4.42.1
> - seqeval==0.0.12

我们强烈建议您使用 [Anaconda](https://www.anaconda.com)来管理您的环境.

## 快速复现
脚本 **train.py** 是项目的主函数, 你可以通过用相应的值替换以下命令中未指定的选项来运行实验。

```shell
    CUDA_VISIBLE_DEVICES=$1 python train.py -dd ${dataDir} -sd ${saveDir} -u -bs 16 -dr 0.3 \ 
        -ced 128 -wed 128 -ehd 512 -aod 128 -sed 32 -sdhd 64
```

由于一些随机因素，有必要使用网格搜索对超参数进行轻微调整。如果你有任何问题，请发布项目或给[我](mailto:wqp@mail2.gdut.edu.cn) 发送电子邮件，我们会尽快回复。

## 问题与建议 

如果对项目有任何问题或者好的建议，欢迎联系[我](mailto:wqp@mail2.gdut.edu.cn)。

## 致谢
本次工作由 ***ESAC实验室*** 协助完成, 部分由[国家科学基金](62172111)资助，部分由[广东省自然科学基金](2019A1515011056)资助，部分由[顺德区重点科技项目](2130218003002)资助。
