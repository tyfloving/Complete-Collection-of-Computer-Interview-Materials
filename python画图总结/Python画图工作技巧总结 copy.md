**<center>【Pandas工作技巧总结-修炼师】</center>**

# Pandas介绍
## 🎯 一、Pandas 是什么？

&emsp;&emsp;Pandas是小量数据分析的大杀器，在目前国内数据挖掘比赛、工作、金融量化中常用工具， 常用来处理类似execl中的结构化数据，和Python语言以及其他可视化的工具包结合使得其在数据分析、数据挖掘、算法分析领域大放异彩。
&emsp;&emsp;如果想要从事数据分析以及算法等相关的工作，Pandas是一个必不可少的工作，本来带大家来认识Pandas中数据格式`daframe`和`series`的使用和区别。

## 💡 二、Pandas中Series

&emsp;&emsp;在pandas中，Series是一种一维的数据结构，类似于数组或列表。它由两部分组成：数据的序列和相应的索引。可以使用pandas中的pd.Series()函数来创建一个Series对象。其中，数据可以是Python列表、NumPy数组或标量。索引可以是默认的整数索引，也可以是自定义的标签索引。

&emsp;&emsp;`Series`对象的特点之一是它的元素是有序排列的，并且每个元素都有相应的索引。这使得对数据的访问和处理更加方便。。以下是一个基本的使用示例：
```python
import pandas as pd

# 创建一个Series对象，可以通过列表，字典，数组都行
data = [10, 20, 30, 40, 50]
index = ['A', 'B', 'C', 'D', 'E']
series = pd.Series(data, index)

data = [10, 20, 30, 40, 50]
index = ['A', 'B', 'C', 'D', 'E']
series = pd.Series(data, index)

print(series)

print(series)

print(output.shape)  # 应该输出 (10, 32, 512)，与query的shape一致

A    10
B    20
C    30
D    40
E    50
dtype: int64

# 查询相关原始的方法
series[0]  # 通过整数索引访问第一个元素
series['a']  # 通过标签索引访问键为'a'的元素
series[1:3]  # 获取索引为1到2的元素
```
&emsp;&emsp;
## 🔍 三、Pandas中的DataFrame

&emsp;&emsp;在pandas中，`DataFrame`是一种二维数据结构，类似于关系型数据库中的表格。它由`多个Series`对象按列组成，并且每列可以具有不同的数据类型。
&emsp;&emsp;DataFrame可以看作是一个带有行和列索引的二维表格，其中每一行表示数据集中的一条记录，每一列表示一种特征或属性。
&emsp;&emsp;可以使用pandas中的pd.DataFrame()函数来创建一个DataFrame对象。可以传入多种类型的数据作为数据源，例如Python列表、NumPy数组、字典或其他DataFrame对象。下面是一个创建DataFrame对象的示例代码：
```python
import pandas as pd

# 创建一个DataFrame对象
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'City': ['Beijing', 'Shanghai', 'Guangzhou', 'Chengdu']}

df = pd.DataFrame(data)

print(df)
Name  Age       City
0    Alice   25    Beijing
1      Bob   30   Shanghai
2  Charlie   35  Guangzhou
3    David   40    Chengdu
```
&emsp;&emsp;DataFrame对象将会显示每一列的名称和对应的数据。默认情况下，每一列将使用整数索引，从0开始递增。同时，DataFrame对象也会有一个通用的行索引，从0开始递增。可以通过以下方式来访问和操作DataFrame对象：
- 通过列名访问列数据：df['Name']将返回'Name'列的数据；
- 通过位置索引访问行数据：df.iloc[0]将返回第一行的数据；
- 使用切片操作获取子集：df.iloc[1:3]将返回索引为1到2的行数据；
- 使用布尔条件筛选行数据：df[df['Age'] > 30]将返回年龄大于30的行数据；
- 使用函数操作列数据：df['Age'].apply(func)将对'Age'列的每个元素应用指定的函数。
- 此外，DataFrame对象还提供了很多其他方法和属性，可以用于对数据进行统计、计算、排序、重塑和处理等操作。

>- 总结一下，pandas中的DataFrame是一种强大的数据结构，用于表示和操作二维数据集。它提供了丰富的功能和方法，使得数据的处理和分析更加灵活和高效。
>- 通常工作中使用的都是Dataframe类型的数据格式，后续文章将对dataframe的数据格式进行详细的使用以及日常问题分享。

# Pandas读取文件总结
## 🎯 一、Pandas中的read_csv文件

&emsp;&emsp;Pandas支持各种类型的文件格式的读写操作例如：csv、txt、json、execl等，实际工作中一般以CSV文件格式为主，大部分时间使用的函数为`read_csv`函数，少部分json数据格式使用`read_json`，对于大多数情况下的结构化数据通过read_csv读取数据并对其进行处理，execl有其他的read_execl函数。
&emsp;&emsp;read_csv()是Pandas库中用于读取CSV文件的函数。CSV文件是一种常用的数据文件格式，通常由逗号分隔的文本组成。read_csv()函数可以将CSV文件中的数据读取到Pandas的DataFrame对象中，便于进行数据分析和处理。

## 💡 二、pd.read_csv重要参数

&emsp;&emsp;**read_csv()** 函数的常用参数包括：
- filepath_or_buffer: CSV文件的路径或文件对象。
- sep: 分隔符，默认为逗号。
- delimiter: 分隔符，默认为None。
- header: 指定数据文件的行数作为列名，默认为0，表示第一行是列名。设为None时表示无列名。
- names: 自定义列名。
- index_col: 指定某列作为行索引。
- usecols: 从数据文件中选择特定的列进行读取。
- dtype: 指定列的数据类型。
- skiprows: 跳过指定行数不读取。
- nrows: 读取指定行数的数据。
- na_values: 将特定值识别为缺失值。
- parse_dates: 指定日期列进行日期解析。
- 以下为一个read_csv的用法
```python
import pandas as pd

df = pd.read_csv('data.csv')
print(df.head())
```
&emsp;&emsp;
## 🔍 三、pd.read_csv读取错误解决
&emsp;&emsp; read_csv读取数据常用的错误总的来说为读取的时候`数量变少问题、utf编码问题、c token问题`，对于这等问题大多数为环境因素、编码因素、里面中文空格符等因素导致，本文针对不同的方案进行针对性处理。
### 读取数量变少
&emsp;&emsp;**quoting:** 当读取csv文件时，如果数据中有"等特殊符号，则可能会出现读取数据少了很多，这个时候就需要加上这个参数保证数据没有出错,quoting=3,具体如下：
```python

df = pd.read_csv('test.csv', sep='\t', header='infer',
              names=None, usecols=None, prefix=None, 
              dtype=None, engine='python', skiprows=None,  nrows=None, quoting=3,
              enconding='utf-8')
```
### 读取报编码错误
&emsp;&emsp;遇到pandas读取出现utf-8的编码问题，可以使用shell中的iconv将数据转为utf-8,`iconv -f utf-8 -t utf-8 > aa`,然后read_csv的时候加上参数quoting=3, engine='python', error_bad_lines=False
### 读取报C Token问题
&emsp;&emsp; 对于上述的问题，如果觉得少数的错误数据是可以去掉丢失的，那么这个时候可以采用读取的时候丢掉的方法即可，但是需要注意一下Pandas的版本问题，如果想要使用上述的`error_bad_lines=False`参数来跳过错误，具体将pandas的版本设置为**pip install pandas=1.42**，不然的话会报不存在该参数的错误
&emsp;&emsp; 对于版本超过1.42的2.0的Pandas版本可以使用参数：on_bad_lines可以指定通过该参数设置为skip来跳过错误
# Pandas基础统计函数
## 🎯 一、基本介绍

&emsp;&emsp;Pandas中的统计函数是数据分析中不可或缺的工具，它们可以帮助我们快速计算数据集中的描述性统计数据，如均值、中位数、标准差等，可以快速的对数据进行分布分析、异常值分析、数据类型等基本数据统计分析。


## 💡 二、使用方法
### 常用函数
&emsp;&emsp;Pandas 提供了很多统计函数，以下是一些常用的：
- mean(): 计算均值
- median(): 计算中位数
- std(): 计算标准差
- var(): 计算方差
- sum(): 计算总和
- min(): 找到最小值
- max(): 找到最大值
- count(): 数值的个数
- info(): 总体数据分布
### 创建DataFrame
```python
import pandas as pd
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [24, 27, 22, 32, 29],
    'Income': [50000, 54000, 35000, 62000, 58000]
}
df = pd.DataFrame(data)
# 计算年龄的均值
mean_age = df['Age'].mean()
print("Mean Age:", mean_age)

# 计算收入的中位数
median_income = df['Income'].median()
print("Median Income:", median_income)

# 计算年龄的标准差
std_age = df['Age'].std()
print("Standard Deviation of Age:", std_age)

# 计算年龄的方差
var_age = df['Age'].var()
print("Variance of Age:", var_age)

# 计算所有人的总收入
total_income = df['Income'].sum()
print("Total Income:", total_income)

# 找到年龄的最大值和最小值
max_age = df['Age'].max()
min_age = df['Age'].min()
print("Max Age:", max_age, "Min Age:", min_age)

```
&emsp;&emsp;

## 🔍 三、进阶用法
&emsp;&emsp; 当我们想要对整体的数据进行分布的查看时，需要查看各个列是否有缺失值，以及每个列的数据格式是什么样子时，这个时候需要可以通过info函数来获取相关的结果，具体的代码如下所示：
```python
    print(df.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Data columns (total 3 columns):
    #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
    0   Name    5 non-null      object
    1   Age     5 non-null      int64 
    2   Income  5 non-null      int64 
    dtypes: int64(2), object(1)
    memory usage: 248.0+ bytes
    None

```
&emsp;&emsp;从上面的输出结果可以看出来，每个列是否有缺失值，以及每个列中的数据格式是什么样子的。
&emsp;&emsp;
## 🔍 四、注意事项
&emsp;&emsp;对上述的各个统计函数在使用的过程中需要注意的一些事项，不然可能会出现error，具体主要为：
- 确保在使用统计函数之前，数据是干净且适合进行统计分析的。
- 某些统计函数，如 mean() 和 median()，可能会受到异常值的影响。在这种情况下，可能需要先进行数据清洗或转换。
- 当使用 std() 和 var() 时，要注意它们计算的是样本标准差和方差还是总体标准差和方差。默认情况下，Pandas 计算的是总体标准差和方差（不使用 Bessel's correction）。

# pandas中去重、翻转、分布分析
## 🎯 1. 基本介绍

&emsp;&emsp;在处理数据集时，我们经常需要执行一些基本操作，如去除重复项、获取数据的描述性统计信息，以及对数据进行翻转操作。本文将介绍 Pandas 中的 drop_duplicates、describe 函数以及翻转操作的使用方法。


## 💡 2. 使用方法
### 2.1 去重drop_duplicates
&emsp;&emsp;drop_duplicates 函数用于删除 DataFrame 中的重复行。默认情况下，它会检查所有列，找出重复的行，并只保留第一次出现的行。
```python
import pandas as pd

# 创建一个包含重复行的 DataFrame
data = {'Name': ['Alice', 'Bob', 'Alice', 'David'],
        'Age': [24, 27, 24, 32]}
df = pd.DataFrame(data)

# 去除重复项，默认保留第一个出现的重复项
df_unique = df.drop_duplicates()
print(df_unique)
 # 删除数据中的重复项数据 
 df.drop_duplicated() # 有subset， keep等参数可以选择，
 # 对哪些列重复数据 进行操作，保留最重复项中的哪一个 
 # 输出所以数据中重复的数据 
 df[df.duplicated()], 
 #原理和上述输出空值差不多，都是将重复的数据转为True和False来提取为True的数据
```
### 2.2 描述信息describe
&emsp;&emsp;describe 函数提供了一个快速的方法来获取 DataFrame 中数值列的描述性统计信息，包括计数、平均值、标准差、最小值、四分位数和最大值。
```python
import pandas as pd

# 使用 describe 获取描述性统计信息
desc_stats = df.describe()
print(desc_stats)

```
### 2.3 行列的翻转
&emsp;&emsp;Pandas 中的翻转操作包括轴向翻转（transpose）和行或列的反转。transpose 方法用于交换 DataFrame 的行和列，而行或列的反转可以使用 iloc 或布尔索引实现，具体的用法如下所示：
```python
# 使用 transpose 翻转 DataFrame 的行和列
df_transposed = df.transpose()
print(df_transposed)

# 使用 iloc 反转 DataFrame 的行
df_reversed_rows = df.iloc[::-1]
print(df_reversed_rows)

# 使用 iloc 反转 DataFrame 的列
df_reversed_columns = df.iloc[:, ::-1]
print(df_reversed_columns)

```

## 🔍 3. 高阶用法
### 3.1 describe高阶用法
&emsp;&emsp; 默认情况下，describe()函数只会包括数值类型的列，而会忽略对象类型的列。如果想要包括对象类型的列，可以通过设置参数include='all'来实现。下面是一个示例代码，演示如何使用describe()函数包括对象类型的列：
```python
import pandas as pd

# 创建示例数据
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['Beijing', 'Shanghai', 'Guangzhou']}
df = pd.DataFrame(data)

# 默认describe()
print(df.describe())

# 包括对象类型的列
print(df.describe(include='all'))

Age
count   3.000000
mean   30.000000
std     5.000000
min    25.000000
25%    27.500000
50%    30.000000
75%    32.500000
max    35.000000

       Name        Age      City
count     3   3.000000         3
unique    3        NaN         3
top     Bob        NaN  Shanghai
freq      1        NaN         1
mean    NaN  30.000000       NaN
std     NaN   5.000000       NaN
min   Alice  25.000000       NaN
25%     NaN  27.500000       NaN
50%     NaN  30.000000       NaN
75%     NaN  32.500000       NaN
max  Charlie  35.000000       NaN

```
&emsp;&emsp;
## 🔍 4. 注意事项
&emsp;&emsp;对上述的各个函数在使用的过程中需要注意的一些事项，不然可能会出现error，具体主要为：
- 使用 drop_duplicates 时，可以指定 subset 参数来只对某些列进行去重。
- describe 默认不包括对象类型的列，如果需要包括，可以设置 include='all'。
- 在执行翻转操作时，要确保索引的使用是正确的，以避免出现错误或不符合预期的结果。

# pandas中的增删修改排序空值
## 🎯 1. 基本介绍
&emsp;&emsp;对于结构化的数据Dataframe，我们通常归纳为多少行，多少列，在通过Pandas对Dataframe进行数据分析、处理过程中，通过的操作需要对数据进行增、删、修、改、判断缺失值、以及排序、本文对pandas中的上述操作进行实践，总结实际工作中常用到的函数用法和技巧。
## 💡 2. 使用方法
### 2.1 DataFrame数据查找
- 切片方式： 类似python中list的操作方法： df[3:]
- iloc函数操作方法： df.iloc[:, [1,2,3]], 按照行列切片的方式进行选择数据
- loc函数操作方法： df.loc[:, ‘列名’], 行按照切片的方式进行选择，列要按照列名进行选择
- 按条件查找方法： df[条件], 例如查找为空的数：df[df[‘a’].isnull()],这里要注意一点的是，如果数据类型是Series格式的，它支持numpy那种数据过滤方法，例
如：df[df>3]
- 这里有一点就是有时数据需要输出偶数列的数据，有用到这种写法df.iloc[::2, :]，其中第一个里面为::2代表的意思是从开始到最后，每隔2输出数据。
```python
import pandas as pd
import numpy as np

# 创建一个示例 DataFrame
df = pd.DataFrame({
    'A': range(1, 6),
    'B': range(6, 11),
    'C': range(11, 16)
})

# 使用切片选择第三行之后的所有行
df_slice = df[3:]
print(df_slice)

# 使用 iloc 选择第二列和第三列
df.iloc[:, [1, 2]]

# 使用 loc 选择第一行和列 'B'
df.loc[0, 'B']

# 查找列 'A' 中大于3的所有行
df[df['A'] > 3]

# 选择偶数列
df.iloc[::2, :]
```
### 2.2 DataFrame数据插入
- 插入一行或一列数据：df.insert()
- 将表中数据的某个值替换为其它的值：df.replace(old, new)
```python
import pandas as pd

# 创建一个示例 DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# 在位置1插入一列新数据，列名为 'C'，值为 [7, 8, 9]
df.insert(1, 'C', [7, 8, 9])

# 在位置2插入一行新数据，索引为2，值为 {'A': 4, 'B': 5, 'C': 10}
df.loc[2] = {'A': 4, 'B': 5, 'C': 10}

# 将列 'A' 中的所有 1 替换为 100
df['A'] = df['A'].replace(1, 100)

# 替换多个值，例如将列 'B' 中的 4 和 5 替换为 100 和 200
df['B'] = df['B'].replace([4, 5], [100, 200])


```
###  2.3 DataFrame数据空值NAN
&emsp;&emsp;真实的数据分析工作中，我们经常会碰到数据缺失的情况，这个时候需要对缺失的数据进行清洗，dataframe中使用dropna函数来对缺失数据进行处理
- 1.删除空值： df.dropna()
- 2.删除以行列数据： df.drop()，其中axis=0，1用于调节按行还是按列，如果想要批量的删除行数据，可参考操作：drop_index= df[条件].index.tolist(),df =df.drop(drop_index, axis=0)
- 3.按条件删除数据：df = df[条件]
```python
import pandas as pd
import numpy as np

# 创建一个包含空值的 DataFrame
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [np.nan, 2, 3, 4],
    'C': [1, 2, 3, np.nan]
})

# 删除含有空值的行
df_dropped_rows = df.dropna(axis=0)
print(df_dropped_rows)

# 删除含有空值的列
df_dropped_columns = df.dropna(axis=1)
print(df_dropped_columns)

# 假设我们要删除索引为 [2, 3] 的行
drop_index = df.index.tolist()[2:4]
df_dropped_rows = df.drop(drop_index, axis=0)

# 假设我们要删除列 'B'
df_dropped_column = df.drop('B', axis=1)
```
###  2.3 DataFrame修改列名
- 1.df.rename({‘old_name’:’new_name’}, axis=1, inplace=True) 对文件的某些列进行重新命名
- 2.df.columns = [‘a’, ‘b’] 直接对整个文件的列进行重新命名
## 🔍 3. 高阶用法
### 3.1 sort_values对dataframe进行排序
- 对DataFrame类型的数据的行列进行排序： df.sort_values([‘a’, ‘b’, ‘c’], ascendig= [False, False, True] ), 对列a,b,c按照不同的排序方式进行排序。
```python
import pandas as pd

# 创建示例数据
data = {'Movie': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
        'Rating': [8.5, 9.0, 7.2, 8.7],
        'Director': ['Director X', 'Director Y', 'Director Z', 'Director W']}
df = pd.DataFrame(data)

# 按照电影评分进行降序排序
sorted_df = df.sort_values(by='Rating', ascending=False)
print(sorted_df)

Movie  Rating    Director
1  Movie B     9.0  Director Y
3  Movie D     8.7  Director W
0  Movie A     8.5  Director X
2  Movie C     7.2  Director Z

```
### 3.1 fillna函数进行数据填充
&emsp;&emsp;数据分析真实场景中，缺失值的存在是不可明显存在的，对很多的算法不支持缺失数据的出现，因此，经常需要对缺失的数据进行填充，具体的填充方法为：
```python
import pandas as pd
import numpy as np

# 创建示例数据
data = {'A': [1, np.nan, 3, 4],
        'B': [5, 6, np.nan, 8]}
df = pd.DataFrame(data)

# 填充缺失值为指定值
filled_df = df.fillna(value=0)
print(filled_df)

# 使用列的统计值填充缺失值
mean_filled_df = df.fillna(value=df.mean())
print(mean_filled_df)

# 使用前一个有效值填充缺失值
ffill_filled_df = df.fillna(method='ffill')
print(ffill_filled_df)

A    B
0  1.0  5.0
1  0.0  6.0
2  3.0  0.0
3  4.0  8.0

     A    B
0  1.0  5.0
1  2.7  6.0
2  3.0  6.3
3  4.0  8.0

     A    B
0  1.0  5.0
1  1.0  6.0
2  3.0  6.0
3  4.0  8.0
```
## 🔍 4. 注意事项
&emsp;&emsp;对上述的各个函数在使用的过程中需要注意的一些事项，不然可能会出现error，具体主要为：
- 切片操作时，行的索引默认从 0 开始，列的索引默认从 1 开始。
- 使用 iloc 和 loc 时，要确保索引的范围不会超出 DataFrame 的实际大小。条件查找时，确保条件表达式正确无误，并且适用于 DataFrame 中的数据类型。
- 使用 insert 方法时，如果插入的是列，需要确保指定的位置索引是正确的，并且列名不与现有列名冲突。
- 使用 replace 方法时，可以传递单个值对，也可以传递列表或字典来替换多个值。
- replace 方法默认只替换 exact 精确匹配的值。如果需要替换正则表达式匹配的值，可以设置 regex=True。
- 使用 dropna 方法时，可以通过 how 参数来指定删除规则，例如 how='any' 删除任何包含空值的行或列，how='all' 仅删除所有值都是空值的行或列。
- 使用 drop 方法时，如果指定了 inplace=True，则原 DataFrame 将被修改，而不是返回一个新的 DataFrame。
- 按条件删除数据时，确保条件表达式正确，并且适用于 DataFrame 中的数据类型。
# pandas中索引问题
## 🎯 1. 基本介绍

&emsp;&emsp;在Pandas中，DataFrame 是一种非常灵活的数据结构，它允许我们以表格的形式存储和操作数据。stack 和 unstack 是两个用于操作多级索引（multi-index，也称为层次化索引）的函数，它们可以帮助我们重塑数据的形状，以适应不同的分析需求。
&emsp;&emsp;在介绍上述的两个函数之前，先得对pandas数据格式得索引有一定得了解会比较容易发挥这两个函数得强大功能，个人感觉可以将其理解为数据的一种Hashmap，如下图片中左边的红色框中为一层索引 key-value的不同之处，右边的为两层索引，需如要果注使意用的是行索列引转可换以函重数复不，设这置个索和引字典的中话的，会使用默认的索引(0，1，2.....)这样也发挥不出开列转行函数的作用,大家如果用过pandas里面的神奇函数pivot, 可以去看看里面的核心代码就是这两个函数的转换
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/5025937138664faaa19cf7b5e66132d0.png)

## 💡 2. 使用方法
### 2.1 stack函数使用
&emsp;&emsp;为了要大家更加方便的看到改函数的作用，首先，我们创建一个具有多级索引的 DataFrame。具体如下所示：
```python 
import pandas as pd

# 创建多级索引
index = pd.MultiIndex.from_product([['A', 'B'], ['X', 'Y', 'Z']], names=['Letter', 'Category'])

# 创建 DataFrame
df = pd.DataFrame({'Data': range(6)}, index=index)

# 显示 DataFrame
print(df)

Letter  Category
A       X            0
        Y            1
        Z            2
B       X            3
        Y            4
        Z            5
Name: Data, dtype: int32

```
&emsp;&emsp;对上述的多级索引数据，stack 方法用于将索引的一个级别转换为列。具体操作如下:
```python
# 使用 stack 将 'Category' 级别转换为列
stacked_df = df.stack('Category')

# 显示 stack 后的 DataFrame
print(stacked_df)

Category  X  Y  Z
Letter         
A        0  1  2
B        3  4  5
```
### 2.2 unstack函数使用
&emsp;&emsp;unstack 方法与 stack 相反，它将一个级别的列转换回索引。
```python
# 使用 unstack 将列转换回索引
unstacked_df = stacked_df.unstack('Category')

# 显示 unstack 后的 DataFrame
print(unstacked_df)

Category  X  Y  Z
Letter         
A        0  1  2
B        3  4  5
```
## 🔍 3. 高阶用法
### 3.1 特征工程中的trick
&emsp;&emsp;当有两层行索引的时候，如果想要去掉设置的索引改为默认的直接重设即可：
```python
 # 设置多层索引  result.unstack(level=0) 
 #将第2层索引翻转为列，-1为第一层索引  
 # 这个翻转函数在进行特征工程的时候经常会用到： 
 1. 当有两层行索引的时候，如果想要去掉设置的索引改为默认的直接重设即可：
 df.reset_index()
 2. 当有两层列索引的时候，往往进行特征提取的时候，需要将多张表进行meger或者concat，这个时候表的columns都是单层的，
 这个时候可以使用ravel骚函数或者将其转为元组类型的list，将多层的表转换为单层的表：
 pair_cols= df.columns.ravel() 
 df.columns = [str(i) + '_' str(j) for i, j in pair_cols]

```
### 3.1 fillna函数进行数据填充
&emsp;&emsp;数据分析真实场景中，缺失值的存在是不可明显存在的，对很多的算法不支持缺失数据的出现，因此，经常需要对缺失的数据进行填充，具体的填充方法为：
```python
import pandas as pd
import numpy as np

# 创建示例数据
data = {'A': [1, np.nan, 3, 4],
        'B': [5, 6, np.nan, 8]}
df = pd.DataFrame(data)

# 填充缺失值为指定值
filled_df = df.fillna(value=0)
print(filled_df)

# 使用列的统计值填充缺失值
mean_filled_df = df.fillna(value=df.mean())
print(mean_filled_df)

# 使用前一个有效值填充缺失值
ffill_filled_df = df.fillna(method='ffill')
print(ffill_filled_df)

A    B
0  1.0  5.0
1  0.0  6.0
2  3.0  0.0
3  4.0  8.0

     A    B
0  1.0  5.0
1  2.7  6.0
2  3.0  6.3
3  4.0  8.0

     A    B
0  1.0  5.0
1  1.0  6.0
2  3.0  6.0
3  4.0  8.0
```
## 🔍 4. 注意事项
&emsp;&emsp;对上述的各个函数在使用的过程中需要注意的一些事项，不然可能会出现error，具体主要为：
- 使用 stack 时，如果原始 DataFrame 有多个列，stack 后将只保留一个列，其他列的数据将丢失。
- unstack 可以用于恢复 stack 操作之前的状态，但要注意，如果 stack 后的数据经过了修改或筛选，unstack 可能无法完全恢复原始结构。
- 当使用 stack 或 unstack 时，如果指定的级别不存在，会引发 KeyError。
# pandas中的透视表
## 🎯 1. 基本介绍
&emsp;&emsp;在数据处理中，经常需要对数据进行重塑以适应不同的分析需求。Pandas 提供了 pivot 函数，允许用户重构长格式（long format）的数据为宽格式（wide format），通过指定索引（index）、列（columns）和值（values），可以快速地创建一个新的派生表，使得数据的展示更加直观。
## 💡 2. 使用方法
&emsp;&emsp; 为了使得大家更加清晰的看情况pivot函数的用法，我们创建示例 DataFrame并再次基础上进行实验，具体的代码如下：
```python 
import pandas as pd

# 创建一个示例 DataFrame
df = pd.DataFrame({
    'Person': ['John', 'John', 'Lisa', 'Lisa'],
    'Year': [2017, 2018, 2017, 2018],
    'Age': [24, 25, 35, 36]
})

# 显示原始 DataFrame
print("原始 DataFrame:")
print(df)

原始 DataFrame:
    Person  Year  Age
0    John  2017   24
1    John  2018   25
2    Lisa  2017   35
3    Lisa  2018   36

```
&emsp;&emsp;使用 pivot 函数，我们可以将 'Person' 作为行索引，'Year' 作为列，'Age' 作为值。具体操作如下:
```python
# 使用 pivot 重塑 DataFrame
pivot_df = df.pivot(index='Person', columns='Year', values='Age')

# 显示 pivot 后的 DataFrame
print("\npivot 后的 DataFrame:")
print(pivot_df)

pivot 后的 DataFrame:
      Year      
2017    2018
John   24     25
Lisa   35     36
```
## 🔍 3. 注意事项
&emsp;&emsp;对上述的各个函数在使用的过程中需要注意的一些事项，不然可能会出现error，具体主要为：
- pivot 函数要求 values 参数指定的列只能有一个，如果存在多个，则需要先进行数据聚合。
- 使用 pivot 时，如果某些索引和列的组合在原始数据中不存在，Pandas 会填充缺失值（NaN）。
- pivot 可以与 pivot_table 函数结合使用，pivot_table 提供了更多的灵活性，如数据聚合和处理多重索引。
# pandas中一行变多行
## 🎯 1. 基本介绍
&emsp;&emsp;在Pandas中，explode是一个用于将序列值分解成多行的函数。当DataFrame中的某一列包含序列（如列表或数组），而你希望将这些序列中的每个元素转换为DataFrame的一行时，explode就非常有用。
&emsp;&emsp;真实的数据分析工作中，通过用到explode是和str中的split结合起来用，因此，真实数据中，组成的list大多数为字符串格式，因此，通常需要将字符串转换成列表，然后在将使用explode函数将一列数据转换成多列数据。
## 💡 2. 使用方法
### 2.1 explode函数使用
&emsp;&emsp;为了要大家看起来更好的理解explode函数的使用，我们创建一个dataframe，然后将其一列为多个数组的列，转换成多个列。具体如下所示：
```python 
import pandas as pd

# 创建包含列表的 DataFrame
df = pd.DataFrame({
    'ID': [1, 2],
    'Values': [['A', 'B', 'C'], ['D', 'E', 'F']]
})

# 显示原始 DataFrame
print("原始 DataFrame:")
print(df)

原始 DataFrame:
   ID Values
0   1    [A, B, C]
1   2    [D, E, F]

```
&emsp;&emsp;将使用explode将Values列中的每个元素转换为一行:
```python
# 使用 explode 将 Values 列的元素转换为多行
df_exploded = df.explode('Values')

# 显示 explode 后的 DataFrame
print("\nexplode 后的 DataFrame:")
print(df_exploded)

explode 后的 DataFrame:
   ID Values
0   1      A
0   1      B
0   1      C
1   2      D
1   2      E
1   2      F
```
### 2.2 split函数使用
&emsp;&emsp;str中的split函数是对一列的字符串安装某个分隔符进行切分，然后将其转换成列表的操作。
```python
import pandas as pd

df = pd.DataFrame({'a':[1,2,3], 'b':[2,3,4], 'c':['a, b,c', 'b,c', 'd,e']})
df['c'] = df['c'].str.split(',')
df

	a	b	c
0	1	2	[a, b, c]
1	2	3	[b, c]
2	3	4	[d, e]
```
&emsp;&emsp; 接着我们可以将上述的安装c列通过explode函数对其进行展开，具体如下：
```python
df.explode('c')
	a	b	c
0	1	2	a
0	1	2	b
0	1	2	c
1	2	3	b
1	2	3	c
2	3	4	d
2	3	4	e

```
## 🔍 3. 高阶用法
### 3.1 explode函数底层解析
&emsp;&emsp;上次的操作直接使用explode函数进行，下面将explode的执行过程给大家进行解析，方便理解，具体代码如下所示：
```python
 import pandas as pd

df = pd.DataFrame({'a':[1,2,3], 'b':[2,3,4], 'c':['a, b,c', 'b,c', 'd,e']})
   a  b     c
0  1  2  a, b,c
1  2  3    b,c
2  3  4    d,e

# 接下来，我们将'a'和'b'列设置为索引，并选择'c'列：
df = df.set_index(['a', 'b'])['c']
a  b
1  2        a, b, c
2  3            b, c
3  4            d, e
Name: c, dtype: object
```
&emsp;&emsp;然后，我们使用str.split方法将'c'列中的字符串按逗号分割，并设置expand=True来将分割后的列表转换为单独的列：
```python
df = df.str.split(',', expand=True)
    0    1    2
0   a     b    c
1   b     c  NaN
2   d     e  NaN
# 接着，我们使用stack方法将列转换为行，创建一个层次化索引：
df = df.stack()
a  b
1  2  a    0
   2  b    1
   3  c    2
2  3  b    0
3  4  d    0
   4  e    1
dtype: object
```
&emsp;&emsp;然后，我们使用reset_index方法重置索引，并在drop=True参数下删除原来的列索引：
```python
df = df.reset_index(drop=True, level=1)
   a    0
0  1    a
1  1    b
2  1    c
3  2    b
4  3    d
5  3    e
# 最后，我们再次使用reset_index方法重置索引，并将列名0改为'c'：
df = df.reset_index().rename(columns={0:'c'})
   a  b  c
0  1  2  a
1  1  2  b
2  1  2  c
3  2  3  b
4  3  4  d
5  3  4  e

```
## 🔍 4. 注意事项
&emsp;&emsp;对上述的各个函数在使用的过程中需要注意的一些事项，不然可能会出现error，具体主要为：
- explode只适用于一维序列，如果你的数据是多维的（如二维数组），则需要先将其展平。
- 如果序列中包含NaN或其他缺失值，explode会将它们转换为对应行中的缺失值。
- explode默认不会改变其他列的数据，如果需要，可以通过ignore_index参数重置索引。
# pandas中字符串使用技巧
## 🎯 1. 基本介绍
&emsp;&emsp;Pandas 提供了一个非常强大的字符串处理功能，通过 str 访问器，可以对 Series 或 DataFrame 中的字符串类型列进行各种操作，如大小写转换、字符串分割、正则表达式匹配、检查字符串内容、计数、搜索、长度获取、正则提取以及补零操作。这些操作使得文本数据的处理变得简单而高效。
## 💡 2. 使用方法
&emsp;&emsp;对于本文中的函数的使用前提就是这列的数据格式为str类型，如果不是该类型，需要将其转化为str类型即可，具体的做法为：
``` python 
    df['cols'] = df['cols'].astype('str')
```
### 2.1 大小写转换lower

&emsp;&emsp;使用 str.lower() 和 str.upper() 进行大小写转换。。以下是一个基本的使用示例：
```python
import pandas as pd

# 创建包含字符串的 DataFrame
df = pd.DataFrame({
    'Name': ['alice', 'bob', 'charlie'],
    'Email': ['alice@example.com', 'bob@example.com', 'charlie@example.com'],
    'Text': ['hello world', 'HELLO WORLD', 'Hello PyData']
})

      Name               Email               Text
0    alice  alice@example.com      hello world
1      bob    bob@example.com     HELLO WORLD
2  charlie  charlie@example.com     Hello PyData

# 将 Name 列转换为小写
df['Name'] = df['Name'].str.lower()

# 将 Text 列转换为大写
df['Text'] = df['Text'].str.upper()

print(df)

      Name               Email               Text
0    alice  alice@example.com      HELLO WORLD
1      bob    bob@example.com     HELLO WORLD
2  charlie  charlie@example.com  HELLO PYDATA
```
### 2.2 字符串匹配contains函数
&emsp;&emsp;使用 str.contains() 函数可以对数据进行正则表达式匹配。具体的用法如下所示：
```python
# 检查 Text 列中是否包含 'PYDATA'
df['ContainsPYDATA'] = df['Text'].str.contains('PYDATA', case=False)

print(df)

      Name               Email               Text Username ContainsPYDATA
0    alice  alice@example.com      HELLO WORLD   alice         False
1      bob    bob@example.com     HELLO WORLD     bob           False
2  charlie  charlie@example.com  HELLO PYDATA  charlie          True
```
### 2.3 检查字符串内容isdigit、isalpha、isalnum
- isdigit(): 检查字符串是否只包含数字。
- isalpha(): 检查字符串是否只包含字母。
- isalnum(): 检查字符串是否只包含字母和数字。
- 具体的用法如下所示：
``` python 
 df = pd.DataFrame({
    'Values': ['123', 'abc', 'abc123', '!@#']
})

print(df['Values'].str.isdigit())  # 检查是否全为数字
print(df['Values'].str.isalpha())  # 检查是否全为字母
print(df['Values'].str.isalnum())  # 检查是否为字母和数字
0     True
1     True
2     True
3    False
Name: Values, dtype: bool
```
### 2.3 正则提取和补零操作
- extract(): 使用正则表达式提取字符串中的匹配部分。
- zfill(): 在字符串左侧填充0以达到指定长度。
``` python 
df['Digits'] = df['Values'].str.extract(r'\d+')  # 提取数字
df['Padded'] = df['Values'].str.zfill(10)  # 左侧补0至长度10

print(df)

  Values  CountA  ContainsABC  Length Digits Padded
0    123       0         False      3    123    000000123
1   abc       1          True      3     NaN      000000abc
2 abc123       1         False      6  123    00abc123
3   !@#       0         False      3     NaN        !@#
```
## 🔍 4. 注意事项
&emsp;&emsp;对上述的各个函数在使用的过程中需要注意的一些事项，不然可能会出现error，具体主要为：
- 确保使用 str 方法时，列的数据类型是字符串。
- extract() 方法需要正则表达式的模式匹配，如果列中的某些值不匹配，结果将会是 NaN。
- zfill() 方法会根据指定的宽度在字符串左侧填充0，如果字符串本身的长度大于或等于指定宽度，则不会填充。
- 一些 str 方法返回的是 Pandas 的 Series 或 DataFrame，可能需要进一步处理，如使用 .str.get() 获取分割后的特定部分。
- 当使用 str.contains() 进行正则表达式匹配时，可以通过 na 参数指定如何处理缺失值。
# pandas混合数据处理
## 🎯 1. 基本介绍
&emsp;&emsp;在实际工作中，由于数据采集的失误或者人工处理的时候不当，会造成原始数据类型经常会遇到一列数值型数据中，混杂一些字符串类型的数据，当我们要对这列数据进行统计运算时，就会报相应的错误，当遇到这样问题的时候，如果我们是在进行数据分析，需要找出具体是哪些行存在这样的问题，从而去修改原始数据的采集，而在进行数据建模或者特征提取时，需要对其进行删除或者采用均值数据进行修改，具体的骚操作方法如下：
&emsp;&emsp;`pd.to_numeric` 函数尝试将输入的数据转换为数值类型。当 `errors='coerce'` 参数被设置时，任何不能被转换为数值的数据将被赋值为 NaN（Not a Number），这是一种特殊的浮点数值，用于表示数据缺失。
&emsp;&emsp;`pd.isnull `函数用于识别数据中的缺失值，并返回一个布尔类型的 Series 或 DataFrame，其中的 True 表示对应的数据是 NaN。
## 💡 2. 使用方法
### 2.1 pd.to_numeric函数使用

&emsp;&emsp;使用pd.to_numeric函数将 clos1 列中的数据尝试转换为数值型，非数值型数据将被转换为 NaN。具体的代码如下所示：

```python
import pandas as pd

df = pd.DataFrame({
    'clos1': [1, 2, '3', 'four', 5, None, '7.5', 'eight']
})

   clos1
0      1
1      2
2      3
3   four
4      5
5    None
6    7.5
7   eight

df['clos1'] = pd.to_numeric(df['clos1'], errors='coerce')

    clos1
0    1.0
1    2.0
2    3.0
3    NaN
4    5.0
5    NaN
6    7.5
7    NaN
```
### 2.2  pd.isnull函数过滤
&emsp;&emsp;通过上述的函数可以将不同类型的数据赋值为空，接着我们可以使用isnull函数对其进行过滤，具体为：
```python
nan_mask = pd.isnull(df['clos1'])

print(nan_mask)

0    False
1    False
2    False
3     True
4    False
5     True
6    False
7     True
Name: clos1, dtype: bool
```
### 2.3 提取非数值型数据
&emsp;&emsp;通过布尔索引，我们可以提取出原始数据中那些被转换为 NaN 的非数值型数据。具体的用法如下所示：
``` python 
non_numeric_data = df[nan_mask]['clos1'].unique()
print(non_numeric_data)

array(['four', None, 'eight'], dtype=object)
```
## 🚀 3. 高阶用法
&emsp;&emsp;征工程中经常需要对数据类型进行转换pandas中astype可以为你解忧，在nlp比赛中各列的数据差异比较大时，需要选择所需的数据类型则可以使用select_dtypes，具体用法为：
``` python 
# 如果col1列为数值的字符串类型，可以用astype(float32)转为浮点型 
df["col1"] = df["col1"].astype(float32) 
# 如果col不是字符串类型，但是想使用字符串的运算，可以用astype(str)转为字符串类型 df["col1"] = df["col1"].astype(str) 
# 选择各列数据类型为数值型的数据,以及删除某个类型的数据 
need_type = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] df = df.select_dtypes(indclude=need_type) 
delete_type = ['int'] 
df= df.select_dtype(exclude=delete_type) 
```
## 🔍 4. 注意事项
&emsp;&emsp;对上述的各个函数在使用的过程中需要注意的一些事项，不然可能会出现error，具体主要为：
- 使用 pd.to_numeric 转换时，如果数据中包含 NaN 或 None，根据 errors 参数的设置，它们可以被保留或转换为 NaN。
- errors='coerce' 强制所有无法转换的值变为 NaN，这有助于数据清洗和后续处理。
- pd.isnull 仅能用于识别 NaN，如果需要识别其他类型的缺失值（如 None），需要先进行适当的转换。
# pandas中groupby函数
## 🎯 1. 基本介绍
&emsp;&emsp;对于分箱操作，在处理连续数据的特征工程时经常会用到，特别是在用户评分模型里面用的贼多，但是使用最优分箱进行数值离散化比较多。
&emsp;&emsp;在数据分析中，经常需要根据某些特征将数据分组，并在每个组内执行计算或分析。Pandas 提供了 groupby 功能来实现这一点。此外，qcut 可用于将连续数据分箱为离散区间，而 fillna 用于填充数据中的缺失值。
## 💡 2. 使用方法
### 2.1 cut函数使用
&emsp;&emsp;在进行特征工程时，经常需要按照一定的规则进行统计特征提取，这个gropuby操作和hadoop的mapreduce有一定的相似，groupby可以理解为对数据进行拆分再进行应用再进行合并，当理解了之前介绍的几个骚函数以及一些常用的统计函数然后如果能想象的到groupby之后的数据结构，基本就可以开始你无限的骚操作了，不管是解决产品经理的数据报告需求还是特征提取基本问题不大了，下面介绍一些个人比较喜欢用的操作：

```python
import pandas as pd

df = pd.DataFrame({
    'a': ['A', 'B', 'A', 'C', 'B', 'C', 'A'],
    'b': [1, 2, 3, 4, 5, 6, 7],
    'c': [10, 20, 30, 40, 50, 60, 70]
})
         a         b
0  12.05155  49.744408
1  67.84977  33.425537
2  53.72848  91.631309
3  45.52130  22.993242
4  28.46236  53.725090

```
&emsp;&emsp;使用 pd.cut列进行分箱。
``` python 
# 为等距分箱
bins_1 = pd.cut(df['a'], 4)
print("等距分箱结果：")
print(bins_1.value_counts())
等距分箱结果：
                  a  count
0  (29.071, 52.552]     31
1  (52.552, 76.032]     25
2   (5.497, 29.071]     22
3  (76.032, 99.513]     22
```
### 2.2  qcut函数使用
&emsp;&emsp;使用 pd.qcut列进行分箱，注意里面的参数labels为是否显示具体为：
```python
# 为等频分箱
bins_2 = pd.qcut(df['a'], 4)
print("\n等频分箱结果：")
print(bins_2.value_counts())

等频分箱结果：
                  a  count
0   (0.197, 28.495]     25
1  (28.495, 49.768]     25
2   (49.768, 72.88]     25
3   (72.88, 98.583]     25
```
### 2.3 高级用法
&emsp;&emsp; 按箱子分组并应用统计函数。使用 groupby 和 apply 对 'b' 列按箱子分组，并应用 help_static 函数。具体的用法如下所示：
``` python 
def help_static(group):
    return {
        'max': group.max(),
        'mean': group.mean(),
        'count': group.count()
    }
# 等距分箱统计
temp_1 = df.groupby(bins_1).apply(help_static).unstack()
print("\n等距分箱统计结果：")
print(temp_1)

# 等频分箱统计
temp_2 = df.groupby(bins_2).apply(help_static).unstack()
print("\n等频分箱统计结果：")
print(temp_2)

等距分箱统计结果：
           max       mean  count
0  89.668916  42.667183    25
1  96.302655  55.310322    25
2  95.345022  59.836174    25
3  97.875800  76.837120    25

等频分箱统计结果：
           max       mean  count
0   98.989428  46.483636    25
1   99.994949  67.079796    25
2  100.000000  87.500000    25
3   99.999998  98.000000     1  # 注意：最顶端可能只有一个数据点
```
### 2.4 和fillna连用
- 对于空值，在进行特征工程时，如果空值缺比较多的时候，常将这一列删除，如果缺的20%左右，要不就不对其进行处理，
- 将它当做一种情况看待，或者对空值进行填充，为了更加的使填充值得误差尽可能得小，如果一个id有多条样本，则可以对其进行分组后在填充，不然就用整体分布值进行填充。
- 在数据分析中，处理缺失值是一个常见且重要的任务。Pandas 提供了多种方法来填充缺失值，包括使用统计方法（如中位数）或数学模型（如线性插值）。
``` python 
import pandas as pd
import numpy as np

# 创建包含缺失值的 DataFrame
df = pd.DataFrame({
    'a': ['A', 'B', 'A', 'B', 'A', 'B', 'A'],
    'b': [1, 2, np.nan, 4, 5, np.nan, 7]
})


# 对列a分组后对列b中的空值用用中位数填充 
fuc_nan_median = lambda x: x.fillna(x.median())

# 对列 'b' 分组后填充缺失值
df_median_filled = df.groupby('a')['b'].apply(fuc_nan_median).reset_index()
print(df_median_filled)

    a    b
0  A  4.0
1  B  3.0
2  A  4.0
3  B  3.0
4  A  4.0
5  B  3.0
6  A  4.0
```
&emsp;&emsp;定义一个 lambda 函数，使用插值方法填充缺失值。
``` python 
func_nan_interpolate = lambda x: x.interpolate()

# 对列 'b' 分组后使用线性插值填充缺失值
df_interpolated = df.groupby('a')['b'].apply(func_nan_interpolate).reset_index()
print(df_interpolated)
    a    b
0  A  1.0
1  B  2.0
2  A  3.5
3  B  4.0
4  A  5.5
5  B  NaN # 注意：由于B组最后一个值后没有数据，插值无法进行
6  A  7.0

```
## 🔍 3. 注意事项
&emsp;&emsp;对上述的各个函数在使用的过程中需要注意的一些事项，不然可能会出现error，具体主要为：
- 在使用 fillna 时，确保使用中位数或其他统计量填充是有意义的，并且适用于数据的分布特性。
- interpolate 方法提供了多种插值方法，如 'linear', 'polynomial' 等，可以通过 method 参数指定。
- 使用 groupby 后，如果直接对结果使用 reset_index，可能会得到一个额外的列（如 'level_1'），这列可能需要被删除。
- 在使用 pd.cut 或 pd.qcut 时，labels=False 表示返回的分箱标签是数字而不是字符串。
- groupby.apply 可以应用任何函数，包括自定义函数，返回的结果将根据函数返回的数据结构进行调整。
- 使用 unstack 可以调整多级列索引的布局，使其更易于理解。
# pandas中groupby连用apply
## 🎯 1. 基本介绍
&emsp;&emsp;如果要说上面介绍的一些pandas的基本操作大部分execl厉害的人也能实现，个人感觉pandas处理数据贼有魅力的地方在于它的聚合分组统计操作，这也是在数据建模中特征提取用的最多的地方，在特征提取时，经常需要提取样本分组的统计信息特征。
&emsp;&emsp;因此，把这方面的骚操作掌握好了，不仅可以提升数据分析的质量，同时两个不同的操作在效率上也是数倍甚至几十倍的差距，在介绍groupby之前先介绍几个骚函数：
- **`map ：`** 只能对一列数据进行操作，且不能和groupby进行结合操作
- **`agg：`** 能够与groupby结合操作，但是时一列一列的输出数据的，因此，改方法不能修改在groupby后修改数据，但是它的优点在于可以对多个列数据进行多个不同的基本统计信息（sum, count, min, max等）
- **` transform：`** 能够与groupby结合操作，这个函数的优点就是不改变数据的形状来进行分组统计，数据即是多行多列也是一行一列进行输出的，但是虽然是多行多列的输出，不能够在transform内部调用某列进行操作，只能先选择某列在进行操作。
- **` apply：`** 能够与groupby结合操作，输出了多行多列的数据，因此可以对数据提取某列进行操作，上述骚函数中，apply函数的功能最为强大，只有你想不到的，没有它做不到的。
## 💡 2. 使用方法
### 2.1 groupby函数使用

&emsp;&emsp;在进行特征工程时，经常需要按照一定的规则进行统计特征提取，这个gropuby操作和hadoop的mapreduce有一定的相似，groupby可以理解为对数据进行拆分再进行应用再进行合并，当理解了之前介绍的几个骚函数以及一些常用的统计函数然后如果能想象的到groupby之后的数据结构，基本就可以开始你无限的骚操作了，不管是解决产品经理的数据报告需求还是特征提取基本问题不大了，下面介绍一些个人比较喜欢用的操作：
&emsp;&emsp; 对Dataframe数据进行Groupby之后，可以直接一些简单的统计操作，基本上该有的统计函数都有封装，如果只要统计某列，只需将groupby后的数据取出那一列进行相应的操作就可以，具体如下所示：

```python
import pandas as pd

df = pd.DataFrame({
    'a': ['A', 'B', 'A', 'C', 'B', 'C', 'A'],
    'b': [1, 2, 3, 4, 5, 6, 7],
    'c': [10, 20, 30, 40, 50, 60, 70]
})
print(df)

    a  b   c
0  A  1  10
1  B  2  20
2  A  3  30
3  C  4  40
4  B  5  50
5  C  6  60
6  A  7  70

```
&emsp;&emsp;使用 groupby 对列 a 进行分组，并计算其它列的均值。
``` python 
mean_grouped = df.groupby('a').mean()
print(mean_grouped)
          b     c
a               
A  4.000000  40.0
B  3.500000  35.0
C  5.000000  50.0
```

### 2.2  按列 a 分组统计列 b 的均值
&emsp;&emsp;如果只对特定列进行操作，可以在 groupby 后指定列名。，具体为：
```python
mean_grouped_b = df.groupby('a')['b'].mean()
mean_grouped_b
a
A    4.0
B    3.5
C    5.0
Name: b, dtype: float64
```
### 2.3 注意事项
- groupby 操作返回的是一个分组对象，可以通过 .mean()、.sum() 等聚合函数来计算统计量。
- 如果分组列中含有 NaN 值，它们将被自动排除在分组之外。
- 聚合函数默认不会修改原始 DataFrame，而是返回一个新的对象。
&emsp;&emsp;

## 🚀 3. 高阶用法
### 3.1 性能对比
&emsp;&emsp;在使用groupby时有多种不同的方式，下面为具体的实例：
``` python 
df = pd.DataFrame({'key1':['a', 'a', 'b', 'b', 'a'],
                    'key2':['one', 'two', 'one', 'two', 'onw'],  'data1':np.random.randn(5), 
                    'data2':np.random.randn(5)}) 
# 对上述的表按照key1，key2分组统计data1的均值的两种写法：
1.df['data1'].groupby([df['key1'], df['key2']]).count()
2.df.groupby(['key1', 'key2'])['data1'].count()
```
- 经过对上述的两段代码进行性能测试可以发现，第二段代码相对于第一段代码性能
更优，在执行的时间复杂度上有一定的优势。
- 第一段代码可以解释为，取df表中data1字段按照key1，key2字段进行统计，而第二段代码是先对df表中的key1，key2字段进行分组，然后取data1字段进行统计，个人更习惯于第二段代码不仅更好理解，同时写起来更加的优雅。
### 3.2 和lambda、函数的结合使用
&emsp;&emsp;groupby结合上述几个骚函数进行分组统计信息的使用总结，如果能使用lambda表达式完成的，尽量使用lambda表达式而不去写一个函数，具体lambda的使用为：
``` python 
 按照列a数据分组统计其列b的均值，求和并修改名字且名字前缀为hello_
 df.groupby('a'['b'].agg({'b_mean':'mean','b_sum':'sum'}).add_prefix('hello_') 
   其中add_prefix属性用于给列添加前缀，如果是要对行添加前缀则需要使用apply函数  
          hello_b_mean  hello_b_sum
a                                  
A          4.000000      16.000000
B          9.000000      18.000000
C          5.000000      11.000000
 # 将groupby与apply结合起来进行自定义函数设计(需要传入参数的写法)，不要传入 
 # 参数使用lambda表达式即可完成 
 1. 按照列a进行groupby取列b中值最大的n个数
 def the_top_values(data, n=3, cols_name='b'):
   return data.sort_values('b')[:n] 
 df.groupby('a').apply(the_top_values, n=3, clos_name='b')
    a   b
2  A   7
0  A   5
6  A   3
1  B  10
4  B   8
5  C   6
3  C   4
```
### 3.3 和agg函数结合使用的高级玩法
&emsp;&emsp; agg对数值型列进行多方式统计, 这里需要注意一点的就是，通过agg进行一列的多统计特征的时候，最后的输出结果是多个multiindex的columns，这个时候需要对其进行一下列名转换
``` python
import pandas as pd

# 示例 DataFrame
data = {
    'ip': ['ip1', 'ip1', 'ip2', 'ip2', 'ip3'],
    'a': [10, 20, 30, 40, 50],
    'b': [100, 200, 300, 400, 500],
    'c': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)
      ip   a   b  c
0    ip1  10 100  1
1    ip1  20 200  2
2    ip2  30 300  3
3    ip3  40 400  4
4    ip2  50 500  5

df = df.groupby('ip').agg({
                             'a' : ['sum', 'max', 'min'],
                             'b': ['sum', 'max', 'min'],
                             'c': lambda x: len(x)
                          })
df.columns = [i[0] + '_' + i[1] for i in df.columns]
df = df.reset_index()
    ip  a_sum  a_max  a_min  b_sum  b_max  b_min  c_len
0  ip1    30.0    20.0    10.0   300.0  200.0  100.0      2
1  ip2   120.0  500.0    30.0  1200.0  500.0  300.0      2
2  ip3    40.0  400.0    40.0   400.0  400.0  400.0      1
``` 
### 3.3 agg、apply、transforms、map对比
&emsp;&emsp;agg 是聚合（aggregation）的缩写，用于对数据集中的分组（groupby）应用一个或多个聚合函数。
&emsp;&emsp;apply 是一个通用的方法，用于对数据集中的轴（axis）应用一个函数，并返回函数的结果。
&emsp;&emsp;transform 与 apply 类似，但它返回的是对原始数据集的转换结果，保持原始数据的形状不变。
&emsp;&emsp;map 是一个用于将映射（mapping）应用到数据集中的元素的方法，通常用于一对一的映射。
![alt text](image-1.png)
``` python
df = pd.DataFrame({
    'A': [1, 2, 1, 2],
    'B': [10, 20, 30, 40]
})

result = df.groupby('A').agg({'B': 'sum'})
result = df.groupby('A')['B'].apply(list)
result = df.groupby('A')['B'].transform(lambda x: x * 2)
result = df['A'].map({1: 'one', 2: 'two'})
```
### 3.4 注意事项
- 使用 agg 时，可以一次对多个列应用不同的函数，但返回的结果列名需要特别注意。
- apply 可以返回任何形状的数组，而 transform 必须返回与原始数据相同形状的数组。
- map 只适用于一维数据，且映射操作必须是一对一的。。
- errors='coerce' 强制所有无法转换的值变为 NaN，这有助于数据清洗和后续处理。
# pandas数据拼接
## 🎯 1. 基本介绍
&emsp;&emsp;在数据分析中，经常需要将多个数据集合并为一个统一的数据结构以进行进一步的分析。Pandas 提供了 merge 和 concat 两个函数来实现数据的合并操作。merge 用于根据一个或多个键将不同的数据集按照一定的规则进行合并，类似于 SQL 中的 JOIN 操作。而 concat 用于将多个数据集沿某个轴进行合并，不涉及键的匹配。。
## 💡 2. 使用方法
### 2.1 concat函数使用
&emsp;&emsp;concat()函数是Pandas中用于沿轴（行或列）合并两个或多个DataFrame的函数。它将两个或多个DataFrame按照指定的轴进行连接，并返回一个新的DataFrame。
&emsp;&emsp;参数说明：
>- objs：一个DataFrame对象的序列，用逗号分隔，表示要合并的DataFrame。
>- axis：指定合并的轴，可以是0（默认值，沿行合并）或1（沿列合并）。
>- join：指定连接方式，默认为'outer'，表示对所有列进行外连接；也可以设置为'inner'，表示进行内连接。
>- ignore_index：是否忽略原始索引，如果设置为False（默认值），合并后的>- DataFrame将保留原始索引；如果设置为True，合并后的DataFrame将重新生成默认的整数索引。

具体的使用例子如下：

```python
import pandas as pd

# 创建示例数据
data1 = {'A': [1, 2, 3],
         'B': [4, 5, 6]}
df1 = pd.DataFrame(data1)

data2 = {'A': [7, 8, 9],
         'B': [10, 11, 12]}
df2 = pd.DataFrame(data2)

# 沿行合并两个DataFrame
concatenated_df = pd.concat([df1, df2])
print(concatenated_df)

# 沿列合并两个DataFrame
concatenated_df_axis1 = pd.concat([df1, df2], axis=1)
print(concatenated_df_axis1)

A   B
0  1   4
1  2   5
2  3   6
0  7  10
1  8  11
2  9  12

   A  B  A   B
0  1  4  7  10
1  2  5  8  11
2  3  6  9  12

```
### 2.2  merge函数使用
&emsp;&emsp;merge()函数是Pandas中用于合并（连接）两个DataFrame的函数。它根据指定的一个或多个键（key）将两个DataFrame的行连接起来，类似于SQL中的JOIN操作。
&emsp;&emsp;参数说明：
>- left和right：要进行合并的两个DataFrame。
>- on：指定用于连接的列名（键），可以是单个列名或多个列名组成的列表。
>- how：指定连接方式，默认为'inner'，表示进行内连接；也可以设置为'outer'（外连接）、'left'（左连接）或'right'（右连接）。
&emsp;&emsp;具体的例子使用说明如下所示：
```python
import pandas as pd

# 创建示例数据
data1 = {'A': [1, 2, 3],
         'B': [4, 5, 6]}
df1 = pd.DataFrame(data1)

data2 = {'A': [2, 3, 4],
         'C': [7, 8, 9]}
df2 = pd.DataFrame(data2)

# 按照'A'列进行内连接
merged_inner = pd.merge(df1, df2, on='A')
print(merged_inner)

# 按照'A'列进行左连接
merged_left = pd.merge(df1, df2, on='A', how='left')
print(merged_left)

# 按照'A'列进行外连接
merged_outer = pd.merge(df1, df2, on='A', how='outer')
print(merged_outer)

A  B  C
0  2  5  7
1  3  6  8
   A  B    C
0  1  4  NaN
1  2  5  7.0
2  3  6  8.0
```
## 🔍 3. 注意事项
&emsp;&emsp;对上述的各个函数在使用的过程中需要注意的一些事项，不然可能会出现error，具体主要为：
- merge 函数可以通过 how 参数指定合并的类型，如 inner（默认）、outer、left、right。
- concat 函数主要用于简单合并，不涉及基于键的合并，如果需要基于键的合并，应使用 merge。
- 当合并具有不同列名的 DataFrame 时，merge 会保留所有列，但未匹配的键对应的列将被填充为 NaN。
- 使用 concat 时，如果轴方向不一致，可以通过 axis 参数指定合并的轴向。
## pandas时间处理
 🎯 1. 基本介绍
&emsp;&emsp;如果要对时间序列相关的数据进行数据分析与挖掘，而时间做为一种特殊的数据格式，不同于字符串，整型的数据格式，但是它们之间又是有一定的联系，在介绍pandas时间处理的方法之前，首先介绍一下关于python的时间处理的相关知识以及常用的包：
&emsp;&emsp;首先对于时间维度信息在进行数据分析以及特征工程的时候经常挖掘分析的时间维度信息如下：年、月、日、是否周末、是否节假日、一年中的第几周、一周中的第几天、距离节假日的时间距离、年月日结合相关业务操作的时间范围、针对业务特征日期的处理
- 因此，对于上述时间维度的数据分析和挖掘，朝天椒将自己在时间维度上的一些处理骚手段总结如下：
- python处理时间的常用包datetime
- datetime数据格式为：datetime(year，month，day，hour，minute，second， microsecond)， 然后可以通过相关的api得到一个时间戳的各个小字段。还有对于datetime类型的时间戳二者之间可以进行相互加，减等操作：

## 💡 2. 使用方法
### 2.1 date_range函数使用
&emsp;&emsp;该函数主要参数有开始(start)、结束时间(end)，多长时间范围periods，以及按什么频率(freq)方式进行时间移动，通过该函数得到的是一个pandas个数的datetime数据格式，因此可以直接使用datetime的相关属性函数对数据进行相关的操作。具体的使用例子如下：

```python
import pandas as pd

# 生成日期范围
date_index = pd.date_range('2024-06-01', periods=5, freq='D')

print(date_index)

DatetimeIndex(['2024-06-01', '2024-06-02', '2024-06-03',
               '2024-06-04', '2024-06-05'],
              dtype='datetime64[ns]', freq='D')

# 默认按天进行移动，可以按月，年等，还可以具体多少天进行移动 
tmp = pd.date_range(start='2018-01-01', periods=10, frep='3M') 
tmp = tmp.strftime('%F').tolist()
```

### 2.2  pd.resample函数使用
&emsp;&emsp;pd.resample()：神奇的对时间进行采样的函数， 该函数除了采样功能外，还可以进行一些关于时间的统计， 主要有加、减、乘等操作， 相关于一种快速的对时间按年月日等进行分组操作， 介绍几个重要的参数：
>- 1.colsed：如果为left就是将时间区间分为多少个区间，每个区间都是左闭右开，如果为right则是左开右闭
>- 2.label：如果就是选用区间的哪边作为结果的索引值
>- 3.rule：按什么方式进行采样，’3D’：三天
>- 4.how：通过什么样的方式， 这个可以通过参数控制， 也可以通过属性函数的写法，特别注意的一个是， 金融中有个函数可以对其进行快速的计算最大、最小值等’ohlc’
>- 5.fill_method：对缺失值进行填充的方法， ‘bfill’, 等
```python
import pandas as pd
import numpy as np

# 创建一个示例 DataFrame
df = pd.DataFrame({'values': np.arange(12)}, 
                   index=pd.date_range('2018-01-01', periods=12, freq='D'))

df_time = df.resample('3D', closed='left').sum()
print("每3天汇总一次的结果（closed='left'）:")
print(df_time)

            values
2018-01-01      3
2018-01-04      9
2018-01-07     15
...
#  升采样：按6小时填充
df_time2 = df.resample('3D', closed='right').sum()
print("\n每3天汇总一次的结果（closed='right'）:")
print(df_time2)
            values
2018-01-02      6
2018-01-05     12
2018-01-08     9
...
# 升采样：按6小时填充
df = pd.DataFrame({'values': np.arange(2)}, 
                   index=pd.date_range('2018-01-01', periods=2, freq='D'))
df_time = df.resample('6H').asfreq()
print("\n按6小时升采样（使用 asfreq）的结果：")
print(df_time)
             values
2018-01-01  0.0
2018-01-01  1.0  # 第二行是原始数据
...           ...
```
## 🔍 3. 注意事项
&emsp;&emsp;对上述的各个函数在使用的过程中需要注意的一些事项，不然可能会出现error，具体主要为：
- resample 方法的 closed 参数控制了如何定义分组的边界，'left' 表示闭左开右，'right' 表示闭右开左。
- asfreq 方法用于将时间序列升采样到指定的频率，但不会填充缺失值。
- ffill 方法用于向前填充缺失值，bfill 方法用于向后填充。
# pandas中特征工程常用函数
## 🎯 1. 基本介绍
&emsp;&emsp;`pd.get_dummies:`有时在进行特征工程时，当某列的值的种类不是大于20且不同的值的label差异性比较大时，像LR算法则一定需要将其进行one-hot编码，即使使用像xgb/gbm这样的算法，进行one-hot编码也会在拟合效果上有想不到的提升，当然如果对算法的速度有特别的要求，则需要去折中选择。
&emsp;&emsp;`pd.factorize: `这个函数主要对数据进行编码操作的，将类别数据转换为相关数值型数据
&emsp;&emsp; `diff():`，在时序问题中，有时需要提取不同时间的差值特征，比如说前一天和后一天的差值，这是可以采用diff方法，其用法如下：

## 💡 2. 使用方法
### 2.1 pd.get_dummies函数使用
&emsp;&emsp;使用 get_dummies 创建指示变量。下面为get_dummies的具体用法
```python
import pandas as pd
import numpy as np

# 创建示例 DataFrame
df = pd.DataFrame({
    'Color': ['Green', 'Red', 'Green', 'Blue', 'Red', 'Blue'],
    'Value': [1, 2, 3, 4, 5, 6]
})

# 创建指示变量
dummies_df = pd.get_dummies(df['Color'])

print(dummies_df)

   Blue  Green  Red
0   0.0    1.0   0.0
1   0.0    0.0   1.0
2   0.0    1.0   0.0
3   1.0    0.0   0.0
4   0.0    0.0   1.0
5   1.0    0.0   0.0
```

### 2.2  pd.factorize函数使用
&emsp;&emsp;使用 factorize 对分类数据进行编码。函数语法如下：

```python
# 对 'Color' 列进行编码
encoded_df = pd.factorize(df['Color'])

print(encoded_df)

(array([0, 1, 0, 2, 1, 2]), array(['Blue', 'Green', 'Red'], dtype=object))

```
### 2.3  pd.diff函数使用
&emsp;&emsp;diff()函数是Pandas中用于计算差分的函数。它可以计算两个相邻元素之间的差异，并返回一个新的Series或DataFrame。
``` python 
# 计算 'Value' 列的一阶差分
diff_df = df['Value'].diff()

print(diff_df)
0    NaN
1    1.0
2    1.0
3    1.0
4    1.0
5    1.0
Name: Value, dtype: float64

```
### 2.4  pd.rank函数使用
&emsp;&emsp;使用 rank 计算排名。可以对dataframe中的某列数据进行从0-1递增的增加数据，这个可以在特征工程中经常会被用到对连续的数据进行排序。
``` python  
# 计算 'Value' 列的排名
rank_df = df['Value'].rank(method='min')

print(rank_df)
0    1.0
1    2.0
2    3.0
3    4.0
4    5.0
5    6.0
Name: Value, dtype: float64
```

## 🔍 3. 注意事项
&emsp;&emsp;对上述的各个函数在使用的过程中需要注意的一些事项，不然可能会出现error，具体主要为：
- get_dummies 默认情况下会忽略 NaN 值，如果需要包含 NaN，可以设置 drop_first=False。
- factorize 返回的是编码后的数据和原始类别的映射，适用于处理未知类别。
使用 diff 时，差分的第一步将是 NaN，因为它没有前一个值进行比较。
- rank 方法有多种参数可以控制排名的计算方式，如 method 参数可以设置为 'min'、'max' 或 'average'。
# pandas中高性能query、eval函数
## 🎯 1. 基本介绍
&emsp;&emsp;pandas进行列的查询，经常会常使用df[条件]的方式，但是这种写法的性能不是很高， pandas基于Numexpr实现了两个高性能的函数，用于数据查询过滤**query()**和数据列值修改与增加新列**eval()**，这两个函数通过传入列名str的方式进行操作：：
&emsp;&emsp;Pandas 提供了 query 和 eval 函数，这两个函数在处理数据时非常有用。query 函数允许你用字符串表达式来筛选数据，而 eval 函数可以计算字符串表达式的值。这两个函数可以大大简化数据处理的代码。

## 💡 2. 使用方法
### 2.1 pd.query函数使用
&emsp;&emsp;query()函数是Pandas中用于根据条件筛选数据的函数。它可以基于一定的表达式筛选出符合条件的数据行。

>- 函数语法如下，参数说明：
>- expr：表示条件的字符串表达式。
>- inplace：是否对原始DataFrame进行就地修改，默认为False，即返回一个新的DataFrame。

```python
import pandas as pd

# 创建示例数据
data = {'A': [1, 2, 3, 4, 5],
        'B': [6, 7, 8, 9, 10]}
df = pd.DataFrame(data)

# 根据条件筛选数据
filtered_df = df.query('A > 2 and B < 9')
print(filtered_df)

A  B
2  3  8
```
### 2.2  pd.eval函数使用
&emsp;&emsp;eval()函数是Pandas中用于执行一些计算或表达式的函数。它可以基于字符串表达式进行快速的计算和转换。函数语法如下：
- 参数说明：
- expr：表示计算或表达式的字符串。
- inplace：是否对原始DataFrame进行就地修改，默认为False，即返回一个新的DataFrame。

```python
import pandas as pd

# 创建示例数据
data = {'A': [1, 2, 3, 4, 5],
        'B': [6, 7, 8, 9, 10]}
df = pd.DataFrame(data)

# 使用eval()函数计算新列
df.eval('C = A + B', inplace=True)
print(df)

A   B   C
0  1   6   7
1  2   7   9
2  3   8  11
3  4   9  13
4  5  10  15
```

## 🔍 3. 注意事项
&emsp;&emsp;对上述的各个函数在使用的过程中需要注意的一些事项，不然可能会出现error，具体主要为：
- query 函数使用的条件字符串应当是有效的 Python 表达式，并且可以包含Pandas的向量化函数。
- eval 函数可以执行更复杂的表达式，但需要小心使用，因为它会执行字符串中的代码，可能会引起安全问题。
- 在使用 eval 时，如果表达式中包含的列有 NaN 值，结果可能也会包含 NaN。
- 从性能角度来看，query 方法通常比使用 eval 更快，特别是在处理大型数据集时。

# pandas优雅的连接mysql
## 🎯 1. 基本介绍
&emsp;&emsp;在数据分析和数据科学项目中，经常需要将数据在不同的存储介质之间进行迁移。Pandas 提供了非常方便的功能，可以轻松地将 DataFrame 数据写入到 MySQL 数据库中。这通常通过 SQLAlchemy 这个 Python SQL 工具包来实现，它为数据库提供了一个优雅的接口。

## 💡 2. 使用方法
### 2.1 安装必要的库
&emsp;&emsp;首先，确保安装了 pandas, pymysql, 和 SQLAlchemy。
```python
pip install pandas pymysql sqlalchemy
```

### 2.2   创建数据库连接引擎
&emsp;&emsp;使用 create_engine 函数创建数据库连接。

```python
import pymysql
from sqlalchemy import create_engine

def connect_mysql(host='0.0.0.0',
                  port=3306, 
                  user='tanyunfei', 
                  password='tyf1994127', 
                  db='b2brec'):
    try:
        utf_flag = "charset=utf8"
        engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}?{utf_flag}")
        print("数据库连接成功！")
    except Exception as e:
        print("数据库连接失败：", e)
    return engine
```
### 2.3   将DataFrame写入MySQL
&emsp;&emsp;使用 to_sql 方法将 DataFrame 数据写入到 MySQL 数据库中。
``` python 
import pandas as pd

# 假设 df_res 是我们要写入数据库的 DataFrame
df_res = pd.DataFrame({
    'column1': [1, 2, 3],
    'column2': ['A', 'B', 'C']
})

# 获取数据库连接引擎
engine = connect_mysql()

# 写入数据，这里需要指定表名和更新标志
table_name = 'your_table_name'
update_flag = 'append'  # 或 'replace'

with engine.begin() as conn:
    df_res.to_sql(name=table_name, con=conn, if_exists=update_flag, index=False)

```

## 🔍 3. 注意事项
&emsp;&emsp;对上述的各个函数在使用的过程中需要注意的一些事项，不然可能会出现error，具体主要为：
- 确保在 connect_mysql 函数中正确设置了数据库连接参数，包括主机、端口、用户名、密码、数据库名。
- to_sql 方法中的 if_exists 参数可以设置为 'fail'、'replace' 或 'append'，以控制当表已存在时的行为。
- 设置 index=False 可以防止 Pandas 将 DataFrame 的索引作为一列写入数据库。
确保在写入数据之前，DataFrame 的列数据类型与数据库中的列数据类型兼容。
