**<center>【大数据框架使用技巧总结-修炼师】</center>**
&nbsp;

----------
[TOC]

## 🎯 1.基本介绍
- hadoop的架构图
- groupby例子
- hadoop中数据倾斜问题
- Hadoop中的常用命令

## 💡 2. 代码用法
### 2.1 查看文件系统状态
### 2.2 列出目录内容
### 2.3 创建目录
### 2.4 上传文件到HDFS
### 2.5 下载HDFS文件到本地
### 2.6 删除文件或目录

## 💡 3. hadoop中难记的命令

## 💡 4. 注意事项

## 💡 5. 总结

# Spark介绍

## 🎯 1.基本介绍

## 💡 2. spark架构图

## 💡 3. spark进行DAG/TASK任务调度流程

## 💡 4. DAGScheduler的具体流程

# PySpark读取数据

## 🎯 1.基本介绍

## 💡 2. 代码用法
### 2.1 初始化spark环境
### 2.2 读取hdfs上的数据
### 2.3 从hive中读取数据
### 2.4 读取Parquet文件

## 💡 3. 注意事项

## 💡 4. 总结

# PySpark中RDD介绍

## 🎯 1.基本介绍

## 💡 2. 代码用法
### 2.1 常用rdd的transfromation
### 2.2 常用的action操作

## 💡 3. 注意事项

## 💡 4. 总结

# PySpark中DataFrame基本操作

## 🎯 1.基本介绍

## 💡 2. 代码用法
### 2.1 初始化spark环境
### 2.2 spark中dataframe基础操作

## 💡 3. 注意事项

## 💡 4. 总结

# PySpark中DataFrame中withColumnRenamed

## 🎯 1.基本介绍

## 💡 2. 代码用法
### 2.1 初始化spark环境
### 2.2 spark中dataframe创建
### 2.3 使用withColumnRenamed重命名列

## 💡 3. 高级用法
### 3.1 批量重命名列
### 3.2 与其它操作结合使用

## 💡 4. 注意事项

## 💡 5. 总结

# PySpark中DataFrame中dropna

## 🎯 1.基本介绍

## 💡 2. 代码用法
### 2.1 初始化spark环境
### 2.2 创建DataFrame
### 2.3 删除包含缺失值的行
### 2.4 删除包含缺失值的列

## 💡 3. 高级用法
### 3.1 指定删除条件

## 💡 4. 注意事项

## 💡 5. 总结

# PySpark中DataFrame中数据过滤

## 🎯 1.基本介绍

## 💡 2. 代码用法
### 2.1 初始化spark环境
### 2.2 创建DataFrame
### 2.3 使用when进行条件筛选和数据转换
### 2.4 使用filter进行条件筛选

## 💡 3. 高级用法
### 3.1 链式使用when和filter

## 💡 4. 注意事项

## 💡 5. 总结

# PySpark中dropDuplicates和sort

## 🎯 1.基本介绍

## 💡 2. 代码用法
### 2.1 初始化spark环境
### 2.2 创建DataFrame
### 2.3 删除重复的行
### 2.4 对数据进行排序

## 💡 3. 高级用法
### 3.1 多列排序

## 💡 4. 注意事项

## 💡 5. 总结

# PySpark中when和otherwise

## 🎯 1.基本介绍

## 💡 2. 代码用法
### 2.1 初始化spark环境
### 2.2 创建DataFrame
### 2.3 使用when和otherwise进行条件数据转换

## 💡 3. 高级用法
### 3.1 嵌套使用when和otherwise
### 3.2 结合多个条件

## 💡 4. 注意事项

## 💡 5. 总结

# PySpark中增加Hive表列及添加注释

## 🎯 1.基本介绍

## 💡 2. 代码用法
### 2.1 初始化spark环境
### 2.2 创建或使用现有Hive表
### 2.3 使用PySpark为列添加注释

## 💡 3. 高级用法
### 3.1 批量增加多个列
### 3.2 在增加列时直接添加注释

## 💡 4. 注意事项

## 💡 5. 总结

# PySpark中写入Hive表

## 🎯 1.基本介绍

## 💡 2. 代码用法
### 2.1 初始化spark环境
### 2.2 创建临时表
### 2.3 创建Hive表
### 2.3 将数据插入Hive表

## 💡 3. 高级用法
### 3.1 动态表名和分区
### 3.2 数据类型和格式转换

## 💡 4. 注意事项

## 💡 5. 总结

# PySpark中表连接

## 🎯 1.基本介绍

## 💡 2. 代码用法
### 2.1 初始化spark环境
### 2.2 创建临时表
### 2.3 创建Hive表
### 2.3 内连接
### 2.4 左连接
### 2.5 全连接

## 💡 3. 高级用法
### 3.1 使用多个连接条件
### 3.2 使用别名

## 💡 4. 注意事项

## 💡 5. 总结

# PySpark中对json字段解析

## 🎯 1.基本介绍

## 💡 2. 代码用法
### 2.1 初始化spark环境
### 2.2 创建包含JSON的DataFrame
### 2.3 使用get_json_object提取数据

## 💡 3. 高级用法
### 3.1 提取多个字段

## 💡 4. 注意事项

## 💡 5. 总结


### 💡 相关面试题总结
1.ck和doris的区别
2.hadoop有哪些角色
3.hadoop哪个负责存储哪个负责管理元数据
4.自己部署过hadoop吗
5.hadoop提交yarn的流程
6.hivesql的数据倾斜
7. 窗口函数有哪些
8. rank和dense rank的区别
9. 宽窄依赖
10. rdd的特性
11. watermark机制
12.flink的执行图
13.spark和flink的区别
14. final关键字的用途
15. string为什么会被final修饰
16. stringbuffer和stringbuilder的区别

## 💡 相关高频面试题

### Clickhouse销售主题数仓项目
- 疫情数据平台问题
- 校企合作关系管理系统
- 项目细节
- 项目介绍

### MPP架构是什么
- 数据测试都做了什么
- 需求评审
- 哪些使用增量表，哪些使用全量表
- 核算数据
- 版本号
- 经历过几个版本

### 工作经历
- 做过的需求的整个流程
- 如何和业务和产品进行沟通
- 数据全部存在ck中吗
- 自己负责的表有多少张，举例？
- 做大数据平台，同时两个人要提交SQL处理数据应该怎么做
- 平时做了哪些工作
- 数仓数据量
- 项目中有哪些维度
- 项目开发流程
- 团队作战代码如何管理
- 代码提测流程
- 项目中实际工作流程
- 业务上面的准确性，一致性如何保证呢（业务如何保证准确性）
- 任务时长要求
- 你们部门提交多少job，你负责多少
- 开发了多少个指标
- 你们的表整体跑完需要多少时间？每张表跑多久

### 数据治理
- 血缘追踪/血缘关系
- 有和业务单独讨论业务需求吗
- 任务跑多久
- 抽数推数
- 查询脚本审批流程
- 需求文档
- 有没有遇到过脚本有问题，数据需要回滚的场景
- 代码上线流程
- 验收
- 数据质量监控
- 最严重的就是延迟出数据
- 告警多不多
- 出了告警如何解决
- 最有价值的数据质量检测
- 如何确保指标的数据是正确的（确实是真实的统计结果，但是有异常）
- 如何保证数据质量检测全覆盖
- 检测哪些数据是如何决定的，有对应的文档吗

### 数据测试
- 如果发现SQL写错的怎么办
- 数据分层
- 数据分层的通用方法
- 分层规范
- 拿到指标之后怎么做开发的
- 全链路开发
- 维度建模
- 每一层要做的事情
- 可能遇到的问题
- 任务发生异常有可能是什么原因？
- 任务流异常如何解决（除了重跑）
- 怎么知道指标在哪个主题域（其实是问主题域如何划分）
- 你开发的报表有哪些，有几张(开发的模型有哪些)
- 最满意的一张报表，里面包含了什么信息
- 一张表的数据量
- 告警发给谁
- 全链路开发花多久
- 还有哪些主题
- DWB层和DWS层有什么区别
- 指标完成之后如何交付的
- 怎么确定指标的一致性
- 指标如何去保证稳定性，检测异动
- 这家公司是做什么事情的
- 你的工作对业务的影响
- 工作中比较复杂的点
- 工作中遇到的问题？如何解决
- 全链路开发最后上线了嘛
- 如何将调研需求转化为开发需求
- 需求一：统计指定时间段内，访问客户的总数量。能够下钻到小时数据
- 产品销量波动大吗
- 工作中收获最大的点
- 假如数据出现了抖动，如何判断是技术的问题还是其他问题
- 如何判断指标抖动是哪个维度出了问题
- 假设数仓中遇到了商品类目的变化，应该如何解决

### 面试问题
- 数仓概念
- 常见架构模型
- Kimball 和inmon的区别（维度建模和范式建模的区别）
- 为什么数仓开发慢慢要变成维度建模（为什么实际建模不遵循3NF）
- 能介绍一下维度建模开发的案例吗（维度建模流程）
- 数仓建设流程
- 怎么理解数据仓库
- 为什么要做数仓
- 数仓建模的意义
- 如何判断哪些数据该放在哪些主题域
- 分区，分桶和排序字段的选择
- 介绍一下缓慢变化维（SCD）
- 模型开发的时候，有什么思路是可以固定下来的
- 如何保证数据仓库稳定性
- 如何保证ADS数据准确性（指标的准确性）
- 数仓分层和分主题域有什么用
- 主题域如何划分
- 什么是总线矩阵？有什么用？
- 如何面向主题和维度建模/你们如何划分主题域的
- 如何构建高可靠，高可用的数仓中间表体系
- 为什么建数仓不直接用关系型数据库
- OLTP和OLAP的区别（OLAP和OLTP的区别）
- 数据漂移问题
- 如何对最终开发出来的数仓模型进行度量
- 一条hive sql语句， 3个groupby， 2个 join，共计5个shuffle，已知发生了数据倾斜，请问如何定位是哪里发生的
- ETL和ELT的区别
- 如何面向主题域和维度建模
- ODS层的存在意义
- 是否可以不要ODS层
- 有小时任务吗
- 数据延迟生成
- 疫情项目可视化的设计，核心指标如何拆解，应该展示哪些字段
  - 核心指标拆解：
  - 展示字段：

### HDFS相关
- 介绍一下HDFS
- HDFS的工作机制
- HDFS的读写流程
- HDFS 写文件什么时候知道写完了
- 写入的时候一个datanode挂掉怎么办
- Namenode和datanode的区别，namenode挂掉HDFS就挂掉了吗？有没有高可用方案？（datanode和namenode）
- Namenode的工作机制（namenode的工作流程）
- Yarn中的nodemanager和resourcemanager的区别
- HDFS架构图
- 大数据中如何进行框架选择

### MR相关
- MR中的shuffle过程
- Shuffle的优化
- 设定Map个数和Reduce个数的策略
- 为什么MapReduce阈值设为80%？
- 缓冲区阈值的调整策略？
- MR和Spark的区别
- Map的数量和reduce的数量由什么决定（map数量）
- MR为啥要排序
- MR中join的操作树有哪些

### Hadoop相关
- Hadoop框架架构组成及功能
- Hadoop的容错机制
- Hadoop提交yarn的流程

### Hive相关
- 介绍一下hive
- HDFS框架架构组成及功能
- 星型模型和雪花模型的区别
- 数据块为什么要设计成128M？数据块太小或者太大会带来什么问题
- 你提到了数据块设计的太小造成的两个问题，你认为哪个是最主要的
- 阐述一下DN的工作
- Hive和Hadoop的关联
- Hive和Spark的区别
- Hive优化
- Hive内部表和外部表的区别
- 为啥ORC file是最优
- Hive有几个进程？
- HQL转化成MR的过程
- Reduce端为什么会发生数据倾斜
- 为什么map join可以解决数据倾斜
- 发生数据倾斜的场景
- 如何定位到数据倾斜
- Join中MR的详细流程
  - Common join
  - Map join
- 分区
- 分桶
- Hive的4个by
- Hive 索引
- 小文件相关（小文件的危害，小文件危害）
- 事务事实表，周期快照事实表，累计快照事实表的区别
- 去重
- Hive中使用UDF的步骤
- 列式存储相比于行式存储的优势
- Hive中的文件存储格式，以及他们对应的使用场景
- 一致性哈希
- Hive中发生数据倾斜如何确定哪个key发生的

### 数据库相关
- mysql的使用场景
- mysql的索引是什么，为什么要用b+树
- b+树的特征
- 聚集索引和非聚集索引
- 经常加索引的字段
- B+树比起B-树的优势
- B+树比起B树的优势
- 事务的四大隔离级别，有什么区别，分别解决什么问题
- mysql默认的隔离级别是什么
- 可重复读是如何加锁的
- MVCC
- 事务的四大特性
- 三范式
- sql执行顺序
- mysql中什么时候会不走索引
- 如何解决数据两次读取结果不一致的问题
- count(字段)和count(*)的区别
- char和varchar的区别

### Sqoop相关
- sqoop 导入原理
- sqoop和datax的区别
- sqoop导入的时候对造成数据库查询缓慢

### Spark相关
- 说说Spark的编程抽象
- 什么是RDD
- RDD存数据吗？
- RDD的特点
- RDD的不可变性
- RDD的底层实现
- Spark的宽窄依赖
- Spark的缓存机制
  - cache
  - persist
  - checkpoint
- 这些缓存有什么区别
- Spark的内存模型
- Spark的统一内存模型
- Spark的部署模式有哪些
- Spark的执行模式有哪些
- Spark中的角色（进程）
- Spark中task的数量由什么决定
- Spark提交任务的常用参数
- 格式
- Spark的端口号
- spark调优
  - sql层面
  - Map端优化
  - Reduce端优化
Spark中的参数设置策略

Spark的任务提交流程
Spark的任务执行流程
Spark的几种join
介绍Spark Shuffle
为什么Spark中会发生shuffle
Spark shuffle的优化
ReduceByKey和GroupByKey的区别
什么时候用GroupByKey比用ReduceByKey更好
action算子和transform算子的区别
什么情况会导致Shuffle
简述spark中共享变量（广播变量和累加器）的基本原理
spark广播的原理
Spark共享变量有哪些，分别有哪些优点
如何理解spark中的job和task
Spark的通信机制
orderby和sortby在spark执行引擎上的实现思路
orderby:
sortby:
sparksql转化成执行计划的流程
spark在优化逻辑执行计划和物理执行计划的时候做了什么
Spark数据倾斜的处理方式
executor的数量如何确定
Sparksql如何分析执行计划里的指标
Spark权限控制
Spark小组项目业务逻辑
Flink相关
Flink的模块
Flink安装部署
Flink的组件
Flink常见算子
watermark
Flink如何保证迟到数据不丢失
Flink 的三种不同的时间概念
什么情况下event time和processing time会不一致
如何处理event time和processing time不一致的情况
Flink的部署模式有哪些？分别说明一下
Flink的dataStream和dataset有什么区别
Flink的窗口主要是干什么的
Flink窗口有哪些类型，分别适用于什么场景下
窗口的顺序是靠什么保障的
Flink的精准一次性如何保证的
介绍一下Flink的checkpoint
Flink中checkpoint的实现流程
Flink中的精确一次性是如何保证的
Flink的checkpoint如何保证精确一次性
Flink背压(反压)
Flink中的表
Flink table有哪些类型
Flink从kafka中拿到的元数据
Flink的状态后端
Flink项目
Flink中有哪些常见的state
Flink和kafka如何做到数据不丢失
Flink和mysql如何做到数据不丢失
Hudi
介绍下Hudi
Hudi概念
数仓的优点和缺点
数据湖的优点和缺点
Flink的执行图
Flink的状态图
做Flink的时候遇到的问题
Kafka相关
Kafka的特点有什么
kafka的分区有什么用？
Kafka如何保证消息的顺序消费
如何保证Kafka不重复消费数据
Kafka如何保证精确的ETL
同步副本认定（isr和osr的区别）
kafka中的AR
如何增加Kafka吞吐量（除了加分区）
Kafka底层是如何保证高吞吐量的
Kafka ISR的底层实现
Kafka中的零拷贝
Kafka中的再均衡原理
为什么要往Kafka里写处理结果？
Kafka中如何实现数据的一致性
为什么有Kafka这个消息队列，直接用内存存不行吗
Hbase相关
hbase中rowkey的设计原则
Yarn相关
Yarn的资源调度器
Yarn的任务调度流程
Clickhouse相关（ck）
clickhouse的join
Replacing Merge Tree(RMT)
去重是如何实现的
rollup，withcube的区别
minmax跳数索引
MergeTree的种类
ck如何连接kafka并同步数据
clickhouse的sql优化
clickhouse和doris的使用场景
Doris
数据模型
Java相关
如何理解java中的多态
如何实现java中的多线程
HashMap极端情况下的时间复杂度
Java中的HashMap如何实现的？扩容机制是什么？涉及到数据的迁移吗？插入的过程
ConcurrentHashMap实现原理
Java的重载和重写有什么区别
java中的string为啥要用final修饰
Java的反射机制
线程和进程有哪些区别（进程和线程）
怎么看一个正在执行的java程序状态
futureTask和传统线程有什么差异
常见的负载均衡算法
线程池有什么用，有哪些类型
Springboot中bean的生命周期
JVM垃圾回收
jvm内存结构
jvm什么时候要垃圾回收
垃圾回收，如何判断对象是否是垃圾？不同方式有什么区别？
垃圾回收器分别有啥
垃圾回收策略
JVM堆内存
锁
ReentrantLock的底层实现
java如何实现线程安全
String, StringBuilder, StringBuffer的区别
进程调度算法有哪些
Java方法中的变量和对象存储在哪
==和equals的区别
排序
堆排序
归并排序
快排
正向索引和倒排索引
责任链
优点
缺点
Redis
介绍一下redis cluster（redisc）
优点
redis-cluster的协议
主观下线与客观下线
redis-cluster主从同步
Zookeeper（zk）
zookeeper的高可用是如何保证的
计算机网络相关（计网）
TCP三次握手过程
TCP四次挥手过程
TCP 为啥要三次握手，两次不行吗
TCP保证可靠的机制有哪些，UDP能不能也实现
并行和并发的区别
并发编程中线程如何进行的通信
TCP分层
进程之间的通讯
文件上传到网络要进行几次拷贝
流量控制和拥塞控制
一些常见的拥塞控制算法有哪些
BitMap
linux
高频SQL
行转列
列转行
连续登录问题(通解)
直播间最大同时在线人数
开放题
个人最大的优势
缺点
三个关键词描述你的性格
为什么选择大数据
以后遇到的挑战
认为任职这个职位需要什么技能
遇到困难的时候是如何解决的
遇到矛盾如何解决的
找公司的时候考虑的点
如何看待加班
为什么会选择出国留学读书？为什么在海外留学多年后，选择回国工作？
你的中长期职业规划是什么？
最有成就感的一件事情
最近在看的书
最近有没有关注大数据相关的新闻
怎么看待国内互联网
数据科学是做什么的
数据管理是做什么的
反问环节