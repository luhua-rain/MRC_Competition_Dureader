# Dureader-Bert
2019 Dureader机器阅读理解 单模型代码
### 许多人询问，说明一下：
* 1、数据处理是自己写的，不用squad的数据处理，可以换其他任何数据集，数据输入符合就行，也可以自己重写
* 2、比赛提升主要使用 Multi-task训练，由于代码繁重，故这份代码只有单任务训练
* 3、对于输出层我只使用了一层全连接，也可以自己修改为论文里的输出层，如下：
![image](https://github.com/basketballandlearn/Dureader-Bert/blob/master/2.png)

### 小小提示：
* 竞赛最终结果第七名, ROUGE-L:53.62, BLEU-4:54.97
* 代码上传前已经跑通，所以如果碰到报错之类的信息，可能是代码路径不对、缺少安装包等问题，一步步解决，可以提issue。
* 若有提升模型效果的想法，十分欢迎前来交流（邮箱：1643230637@qq.com）

### 环境(不支持cpu)
* python3
* torch 1.0
* 依赖包 pytorch-pretrained-bert、tqdm、pickle、torchtext

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

##### 排行榜：
![image](https://github.com/basketballandlearn/Dureader-Bert/blob/master/1.png)