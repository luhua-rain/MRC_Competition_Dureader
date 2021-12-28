## 机器阅读理解预训练模型及代码开源


*********************** **更新** ***********************
* 5/21：开源**基于大规模MRC数据再训练**的模型（包括`roberta-wwm-large`、`macbert-large`）
* 5/18：开源比赛代码


## Contents
  - [基于大规模MRC数据再训练的模型](#基于大规模MRC数据再训练)
  - [仓库介绍](#仓库介绍)
  - [运行流程](#运行流程)
  - [小小提示](#小小提示)


## 基于大规模MRC数据再训练

此库发布的再训练模型，在 阅读理解/分类 等任务上均有大幅提高<br/>
（已有多位小伙伴在 Dureader、法研杯、医疗问答 等多个比赛中取得**top5**的好成绩😁）

|                模型/数据集                 |  Dureader-2021  |  tencentmedical |
| ------------------------------------------|--------------- | --------------- |
|                                           |    F1-score    |    Accuracy     |
|                                           |  dev / A榜     |     test-1      |
| macbert-large (哈工大预训练语言模型)         | 65.49 / 64.27  |     82.5        |
| roberta-wwm-ext-large (哈工大预训练语言模型) | 65.49 / 64.27  |     82.5        |
| macbert-large (ours)                      | 70.45 / **68.13**|   **83.4**    |
| roberta-wwm-ext-large (ours)              | 68.91 / 66.91   |    83.1        |


* **数据来源**
  * 网上收集的大量中文MRC数据
  （其中包括公开的MRC数据集以及自己爬取的网页数据等，
  囊括了医疗、教育、娱乐、百科、军事、法律、等领域。）

* **数据构造**
  * 清洗
    * 舍弃：context>1024的舍弃、question>64的舍弃、网页标签占比超过30%的舍弃。
    * 重新标注：若answer>64且不完全出现在文档中，则采用模糊匹配: 计算所有片段与answer的相似度(F1值)，取相似度最高的且高于阈值（0.8）
  * 数据标注
    * 收集的数据有一部分是不包含的位置标签的，仅仅是(问题-文章-答案)的三元组形式。
      所以，对于只有答案而没有位置标签的数据通过正则匹配进行位置标注：<br/>
      ① 若答案片段多次出现在文章中，选择上下文与问题最相似的答案片段作为标准答案（使用F1值计算相似度，答案片段的上文48和下文48个字符作为上下文）；<br/>
      ② 若答案片段只出现一次，则默认该答案为标准答案。
    * 采用滑动窗口将长文档切分为多个重叠的子文档，故一个文档可能会生成多个有答案的子文档。
  * 无答案数据构造
    * 在跨领域数据上训练可以增加数据的领域多样性，进而提高模型的泛化能力，而负样本的引入恰好能使得模型编码尽可能多的数据，加强模型对难样本的识别能力：<br/>
      ① 对于每一个问题，随机从数据中捞取context，并保留对应的title作为负样本;（50%）<br/>
      ② 对于每一个问题，将其正样本中答案出现的句子删除，以此作为负样本；（20%）<br/>
      ③ 对于每一个问题，使用BM25算法召回得分最高的前十个文档，然后根据得分采样出一个context作为负样本，
      对于非实体类答案，剔除得分最高的context（30%）
* **用途**  
  * 此mrc模型可直接用于`open domain`，[点击体验](https://huggingface.co/luhua/chinese_pretrain_mrc_roberta_wwm_ext_large)
  * 将此模型放到下游 MRC/分类 任务微调可比直接使用预训练语言模型提高`2个点`/`1个点`以上
* **合作**
  * 相关训练数据以及使用更多数据训练的模型/一起打比赛 可邮箱联系(luhua98@foxmail.com)~ 
  
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

## 仓库介绍
* **目的**
  * **开源了基于MRC数据再训练的模型**，在MRC任务下微调，效果大幅优于使用预训练的语言模型，其次，旨在提供一个效果不错的`强基线`
  * 有些[mrc比赛](#比赛)由于"年代久远"整理不过来（`others`文件夹），但方案和代码都有，对比着看就看懂了
* **优化**
  * 代码基于Hugginface的squad代码。之前自己开发，版本多且许多细节没有考虑，便转移到squad代码上迭代。但其实现的类缺乏对中文的支持，推理结果有一些影响，**修改之后 此库能较好的支持中文，抽取的答案精度也尽可能不受影响**
  

## 运行流程

脚本参数解释

* `--lm`: 要加载的模型的文件夹名称
* `--do_train`: 开启训练
* `--evaluate_during_training`: 开启训练时的验证
* `--do_test`:  开启预测
* `--version_2_with_negative`: 开启适配于数据中有`无答案数据`（如：squad2.0、dureader2021）
* `--threads`: 数据处理所使用的线程数（可以通过os.cpu_count()查看机器支持的线程数）
  
##### 一、数据 & 模型：
* 将train、dev、test等数据放在datasets文件夹下(样例数据已给出，符合格式即可)
* 通过 export lm=xxx 指定模型目录

##### 二、一键运行
```python 
sh train_bert.sh  # sh test_bert.sh
```

##### 三、无答案问题
* 如果包含无答案类型数据（如：squad2.0、dureader2021），加入--version_2_with_negative就行
* 将数据替换为Dureader2021_checklist的数据, 加入--version_2_with_negative即可


## 小小提示：
* 代码上传前已经跑通。文件不多，所以如果碰到报错之类的信息，可能是代码路径不对、缺少安装包等问题，一步步解决，可以提issue
* 环境
  ```
  pip install transformers==2.10.0 
  ```
* 代码基于transformers 2.10.0版本，但是预训练模型可以使用其他版本加载。转换为tf可使用[转换](https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/convert_bert_pytorch_checkpoint_to_original_tf.py)
* 预训练相关参数 [参考](https://github.com/basketballandlearn/MRC_Competition_Dureader/issues/33)


## 感谢
[zhangxiaoyu](https://github.com/Decalogue)  [huanghui](https://github.com/huanghuidmml)  [nanfulai](https://github.com/nanfulai)

