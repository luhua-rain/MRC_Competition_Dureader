# Dureader-Bert
2019 Dureader机器阅读理解 排名第七

### 环境：
    python3
    安装pytorch-pretrained-bert（pip install pytorch-pretrained-bert）

### 运行步骤如下:
##### 一、数据准备：
    ##### 将trainset、devset等数据放在data文件里
    2、到handle_data目录下运行 sh run.sh --para_extraction
##### 二、制作dataset：
    1、到dataset目录下运行两次 python3 run_squad.py，第一次运行结束后要修改run_squad.py的参数，具体做法run_squad.py末尾有具体说明
##### 三、训练：
    1、到root下运行 python3 train.py，边训练边验证
##### 四、测试:
    1、到predict目录下运行 python3 util.py (测试集太多，也可以在该文件里将路径改为验证集，默认为验证集路径)
    2、运行 python3 predicting.py
    3、到metric目录下， 运行 python3 mrc_eval.py predicts.json ref.json v1 即可
