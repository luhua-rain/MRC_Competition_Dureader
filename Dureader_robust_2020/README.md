# Dureader-robust
2020 Dureader机器阅读理解 单模型代码 (亚军)

## 数据分析
analysis.ipynb: 对比赛数据的统计分析


## 预训练模型
roberta_wwm_ext_large


### 环境(不支持cpu)
* python3  
* torch 1.4.0
* transformers 2.10.0

## 小小提示：
* 竞赛最终结果第二名, F1:79.448, EM:65.13
* 代码上传前已经跑通。文件不多，所以如果碰到报错之类的信息，可能是代码路径不对、缺少安装包等问题，一步步解决，可以提issue
* 拿现有的预训练模型基于这套代码微调可以有一个很高的基线

### 运行流程  
###### 一、数据 & 模型：
* 将train、dev、test等数据放在datasets文件夹下 (格式符合squad的就行)
* 指定模型目录
###### 二、一键运行
* sh train_bert.sh
* 包括训练/验证/预测，--do_train、--evaluate_during_training、--do_test
###### 三、无答案问题
* 如果包含无答案类型，加入--version_2_with_negative就行
* 将数据替换为Dureader2021_checklist的数据可以直接跑

#### 排行榜：
![image](https://github.com/basketballandlearn/MRC_Competition_Repositories/blob/master/Dureader_robust_2020/1.png)
