## 机器阅读理解代码及预训练模型开源分享


***************************************************** **更新** *****************************************************
* 5/21：**开源基于大规模MRC数据再训练**的模型（包括`roberta-wwm-large`、`macbert-large`、`albert-xxlarge`、`albert-xlarge`）
* 5/18：开源代码


## Contents
  - [仓库介绍](#仓库介绍)
  - [运行流程](#运行流程)
  - [小小提示](#小小提示)
  - [基于大规模MRC数据（百万）再训练的模型发布](#基于大规模MRC数据再训练)
  - [相关比赛](#相关比赛)


## 仓库介绍
* **优化**
  * 代码基于Hugginface的squad代码。之前自己开发，迭代版本多，且很多细节没有考虑到（如：空格、特殊字符、滑窗后答案的移动等处理），<br>
    索性就直接在Hugginface的squad代码上迭代。但其实现的类缺乏对中文的支持，推理结果有一些影响，**修改之后 此库能较好的支持中文，抽取的答案精度也尽可能不受影响**
* **由来**
  * 此仓库来源于我2019年参加Dureader比赛的代码库，之后就在此库上开源我参加各种中文机器阅读理解（MRC）比赛的方案和代码 [相关比赛](#相关比赛)
* **目的**
  * "散户"的我上分困难😰，所以这个库旨在为大家提供一个效果不错的`强基线`。大家可以在这基础上改进，也可以借鉴融合进自己的方法
  * 其次，我开源了多个基于机器阅读理解数据（百万）训练后的模型，在MRC任务下微调，效果大幅优于使用预训练的语言模型
  * 有些由于"年代久远"也整理不过来了（`others`文件夹下的比赛）。不过方案和代码都有，对比着看很容易就看懂了


## 运行流程

脚本参数解释

* `--lm`: 要加载的模型的文件夹名称
* `--do_train`: 开启训练
* `--evaluate_during_training`: 开启训练的时候验证
* `--do_test`:  开启预测
* `--version_2_with_negative`: 开启适配于数据中有无答案数据（如：squad2.0、dureader2021）
* `--threads`: 数据处理所使用的线程数（可以通过os.cpu_count()查看机器支持的线程数）
  
##### 一、数据 & 模型：
* 将train、dev、test等数据放在datasets文件夹下 ([格式符合squad的就行](https://aistudio.baidu.com/aistudio/competition/detail/66))
* 通过 export lm=xxx 指定模型目录

##### 二、一键运行
```
cd main
```
```
sh train_bert.sh
```
```
sh test_bert.sh
```

##### 三、无答案问题
* 如果包含无答案类型数据（如：squad2.0、dureader2021），加入--version_2_with_negative就行
* 将数据替换为Dureader2021_checklist的数据可以直接跑


## 小小提示：
* 代码上传前已经跑通。文件不多，所以如果碰到报错之类的信息，可能是代码路径不对、缺少安装包等问题，一步步解决，可以提issue
* 拿此库发布的再训练模型，基于这套代码微调可以有一个很高的基线，已有小伙伴在Dureader2021比赛中取得**top10**的成绩了😁
* **环境**
  ```
  pip install transformers==2.10.0 
  ```

## 基于大规模MRC数据再训练

* 数据来源（百万规模）
  * 网上收集的大量中文MRC数据
  （其中包括 Dureader、WebQA、SogoQA、Squad-2.0、CMRC-2018、法研杯、军事阅读理解、EpidemicQA以及自己爬取的网页数据等，
  这里面包括了百度百科、搜狗搜索、军事、法律、医疗、教育等领域。）

* 训练数据构造
  * 过滤
    * 对于所有数据：context>512的舍弃、question>32的舍弃、answer不是百分百出现在文档中的舍弃、网页标签占比超过30%的舍弃。
  * 标注
    * 对于只有答案没有位置标签的数据进行位置标注：若答案多次出现在文档中，则选择上下文与问题最相似的答案片段作为标签答案（使用F1来计算相似度），若答案只出现一次，则直接默认该答案为标签答案。
  * 无答案数据（正样本 : 负样本 = 1 : 1）
    * 对于每一个问题，随机从数据中捞取context，并保留对应的title;（50%）
    * 对于每一个问题，使用BM25算法召回得分最高的前十个文档，然后根据得分采样出一个context作为负样本。对于非实体类答案，剔除得分最高的context（50%）
* 用途  
  * 此mrc模型可直接用于`open domain`，[点击体验](https://huggingface.co/luhua/chinese_pretrain_mrc_roberta_wwm_ext_large)
  * 将此mrc模型放到下游任务微调可比直接使用预训练语言模型提高`3个点`以上（Dureader上的的实验结果，其他数据集还没具体测试）
* **商务合作**
  * 相关训练数据以及使用更多数据训练的模型可邮箱联系~ 

```
基于此库代码发布的再训练模型，在dureader2021 A榜的详细效果对比
+-----------------------------------------------+
|                                       |  F1   |
+-----------------------------------------------+
| macbert-large (基于哈工大预训练语言模型)  | 65.x  |
+-----------------------------------------------+
| macbert-large                         | 68.13 |
+-----------------------------------------------+
| roberta-wwm-ext-large                 | 66.91 |
+-----------------------------------------------+
| albert-xlargee                        | 65.24 |
+-----------------------------------------------+
| xlabert-xxlarge                       | 65.79 |
+-----------------------------------------------+
```
```
----- 使用方法 -----
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_name = "chinese_pretrain_mrc_roberta_wwm_ext_large" # "chinese_pretrain_mrc_macbert_large"

# Use in Transformers
tokenizer = AutoTokenizer.from_pretrained(f"luhua/{model_name}")
model = AutoModelForQuestionAnswering.from_pretrained(f"luhua/{model_name}")

# Use locally（通过 https://huggingface.co/luhua 下载模型及配置文件）
tokenizer = BertTokenizer.from_pretrained(f'./{model_name}')
model = AutoModelForQuestionAnswering.from_pretrained(f'./{model_name}')
```

## 相关比赛

* **Dureader checklist 2021语言与智能技术竞赛**
* **疫情政务问答助手**（第一）
* **Dureader robust 2020语言与智能技术竞赛**（第二）
* **成语阅读理解**（第二）
* **莱斯杯**（第三）
* **Dureader 2019语言与智能技术竞赛**（第七）