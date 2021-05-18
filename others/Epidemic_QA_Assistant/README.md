
## 最终排名：冠军

### 比赛介绍

* 比赛名称：科技战役-疫情政务问答助手
* [比赛链接](https://www.datafountain.cn/special/BJSJ/talent)

* 疫情政务问答助手比赛提供以疫情相关的**政策数据集**、用户问题以及标注好的答案，其中答案片段一定出现在政策数据集中，因此可以认为答案一定是文档中的一个片段（span）
* 本队将该赛题看成多文档阅读理解任务（Multi-Document Machine Reading Comprehension, MDMRC），这一问题已经在学术界有了较多的研究。大部分的研究将MDMRC分解为文档检索任务和抽取型阅读理解任务，**其中文档检索任务在文档库中检索到和问题最相关的若干文档，抽取型阅读理解任务在从检索到的文档中抽取问题对应的答案片段**。这种框架形式较为灵活，可以在不同的子任务中尝试不同的方法，然后再将子任务的结果组合得到最终答案。

### 队伍介绍

* 队伍名称：中国加油-湖北加油
* 成员介绍
  * 陆华、钟嘉伦和王力来自**华中科技大学Dian团队AI实验室**，组内研究方向包括自然语言处理、计算机视觉等。
  * 余嘉豪是**北京邮电大学**本科三年级在读生，现在在**北京大学王选计算机研究所**严睿老师团队下实习，该团队致力于人机对话、认知计算等自然语言处理方向的研究。
  * 张原来自**北京来也网络科技有限公司**。北京来也网络科技有限公司创办于2015年，致力于做人机共生时代具备全球影响力的智能机器人公司）。

### 方法概要

* 本队采用了**文档检索-阅读理解框架**，使用了预训练模型、**召回优化**、**负样本选择策略**、**多任务训练**等方法辅助模型训练，最后通过模型融合得到了在测试集(A榜)0.744的Rouge-L分数，**排名第一**，验证了自然语言处理可以有效应用于实际生活中。

### ppt分享

* 本队的答辩ppt已上传，供大家学习交流（时间匆忙，所以ppt制作得比较粗糙）

### 比赛结果

* A榜和B榜得分均top1

* <center class="half">
      <img src="https://github.com/basketballandlearn/MRC_Competition_Repositories/blob/master/Epidemic_QA_Assistant/image/B榜排名.PNG" alt="B榜排名" width = "45%" align=right style="zoom: 20%;" />
      <img src="https://github.com/basketballandlearn/MRC_Competition_Repositories/blob/master/Epidemic_QA_Assistant/image/A榜排名.PNG" alt="A榜排名" width = "45%" align=left style="zoom: 20%;" />
  <center>

  
	
   	   
  
  
