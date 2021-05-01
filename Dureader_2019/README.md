# Dureader-Bert
2019 Dureader机器阅读理解 单模型代码。

### 哈工大讯飞联合实验室发布的中文全词覆盖BERT
[论文地址]( https://arxiv.org/abs/1906.08101)  
[预训练模型下载地址]( https://github.com/ymcui/Chinese-BERT-wwm)  
* 只需将要加载的预训练模型换为压缩包内的chinese_wwm_pytorch.bin，即修改from_pretrained函数中weights_path和config_file即可。

### 谷歌发布的中文bert与哈工大发布的中文全词覆盖BERT在Dureader上的效果对比

| 模型 | ROUGE-L | BLEU-4 |
| ------ | ------ | ------ |
| 谷歌bert | 49.3 | 50.2 | 
| 哈工大bert| 50.32 | 51.4 |

由于官方没有给出测试集，实验数据是在验证集上跑出来的

## 许多人询问，说明一下：
* 1、代码是自己写的，不用squad的数据处理，可以换其他任何数据集，数据输入符合就行，也可以自己重写
* 2、比赛提升主要使用 Multi-task训练、以及答案抽取，由于代码繁重，故这份代码只有单任务训练
* 3、对于输出层，我只使用了一层全连接，可参考论文里的输出层，如下：
![image](https://github.com/basketballandlearn/MRC_Competition_Repositories/blob/master/Dureader_2019/2.png)

## 代码：
* 代码主要删减大量不必要代码，也将英文的数据处理改为中文的数据处理，方便阅读和掌握bert的代码。
* handle_data文件夹是处理Dureader的数据，与比赛有关，与bert没有多大关系。
* dataset文件夹是处理中文数据的代码，大致是将文字转化为bert的输入：(inputs_ids,token_type_ids,input_mask), 然后做成dataloader。
* predict文件夹是用来预测的，基本与训练时差不多，一些细节不一样（输出）。
* 总的来说，只要输入符合bert的输入：(inputs_ids,token_type_ids,input_mask)就可以了。

## 小小提示：
* 竞赛最终结果第七名, ROUGE-L:53.62, BLEU-4:54.97
* 代码上传前已经跑通，所以如果碰到报错之类的信息，可能是代码路径不对、缺少安装包等问题，一步步解决，可以提issue。
* 若有提升模型效果的想法，十分欢迎前来交流

### 环境(不支持cpu)
* python3  
* torch 1.0
* 依赖包 pytorch-pretrained-bert、tqdm、pickle、torchtext

### Reference
&emsp;[Bert论文](https://arxiv.org/pdf/1810.04805.pdf)  
&emsp;[Dureader](https://github.com/baidu/DuReader)  
&emsp;[Bert中文全词覆盖论文]( https://arxiv.org/abs/1906.08101)  
&emsp;[pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)

### 运行流程  
###### 一、数据处理：
* 将trainset、devset等数据放在data文件里 (data下的trainset、devset有部份数据，可以换成全部数据。)
* 到handle_data目录下运行 sh run.sh --para_extraction, 便会将处理后的数据放在extracted下的对应文件夹里
###### 二、制作dataset：
* 到dataset目录下运行两次 python3 run_squad.py，分别生成train.data与dev.data,第一次运行结束后要修改run_squad.py的参数，具体做法run_squad.py末尾有具体说明
###### 三、训练：
* 到root下运行 python3 train.py，边训练边验证
###### 四、测试:
* 到predict目录下运行 python3 util.py (测试集太多，也可以在该文件里将路径改为验证集，默认为验证集路径)
* 运行 python3 predicting.py
* 到metric目录下， 运行 python3 mrc_eval.py predicts.json ref.json v1 即可

#### 排行榜：
![image](https://github.com/basketballandlearn/MRC_Competition_Repositories/blob/master/Dureader_2019/1.png)
