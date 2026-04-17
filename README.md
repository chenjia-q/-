抖音电商用户分层与画像分析
本代码实现了基于用户行为数据的 RFM 分层、生命周期阶段识别、多维度画像对比及可视化分析。通过对用户年龄、性别、地区、消费能力、活跃度等特征进行加工，构建了 5 类宏观分层（核心用户、重要用户、潜力用户、一般用户、流失用户）以及 6 类生命周期标签（新用户、成长期、成熟期、预流失、流失、未知），并输出完整的用户标签表。

1. 数据预处理
读取 user_personalized_features.csv

删除无用列（Unnamed:0、Unnamed:0.1）

检查并确认无重复值与缺失值

将数值列转换为 numeric 类型，将订阅字段转为布尔型，将文本类别字段转为 category 类型

2. 基础属性标签构建
标签	构造方法
年龄段 (Age_Band)	将 Age 划分为 18-24、25-34、35-44、45-150
性别标准化 (Gender_Std)	Male→男，Female→女性别标准化 (Gender_Std)	Male→男，Female→女
消费能力标签 (Spending_Ability_Tag)	基于 Average_Order_Value 的三分位（低/中/高）
购买力标签 (Purchasing_Power_Tag)	基于 Total_Spending 的三分位（低/中/高）
3. 生命周期阶段识别
根据 Last_Login_Days_Ago（最近登录天数 R）和 Purchase_Frequency（购买频次 F）定义：

阶段	条件
新用户-14天内	F == 1 且 R ≤ 14
成熟期	R ≤ 7 且 F ≥ 5
🛒 抖音电商用户分层与画像分析
基于用户行为数据的 RFM 分层、生命周期识别、多维度画像对比及可视化分析。
最终输出完整的用户标签表，支持精细化运营、流失预警与个性化推荐。

📁 代码功能概览
模块	说明
数据预处理	清洗、类型转换、缺失值检查
基础属性标签	年龄段、性别标准化、消费能力/购买力分档
生命周期阶段	新用户 → 成长期 → 成熟期 → 预流失 → 流失
RFM 分层	R/F/M 五等分评分 → 8 类细分 → 5 类宏观分层
数据输出	导出 user_tags.csv 标签表
衍生指标 & 聚合	活跃率、AOV、F 均值、组内占比等
可视化分析	柱状图、堆叠图、热力图、饼图等 10+ 张图表
偏好与兴趣分析	品类/兴趣渗透率矩阵（RFM × 兴趣、生命周期 × 品类）
行为分档	活跃度分档、频次对比、生命周期 × 活跃热力图
🔧 1. 数据预处理
python
import pandas as pd   以pd方式导入熊猫
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'E:\抖音电商用户分层与画像分析\user_personalized_features.csv')
df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
✅ 无重复值（df.duplicated().sum() = 0）

✅ 无缺失值（df.isna().sum() 全为 0）

🔄 类型转换：

数值列 → numeric

Newsletter_Subscription → bool

文本类别 → category

🏷️ 2. 基础属性标签
标签	构造方法	示例值
Age_Band	年龄分箱：[18,25,35,45,150]	18-24, 25-34, 35-44, 45-150
Gender_Std	映射 Male→男, Female→女	男, 女
Spending_Ability_Tag	Average_Order_Value 三分位	低, 中, 高
Purchasing_Power_Tag	Total_Spending 三分位	低, 中, 高
⏳ 3. 生命周期阶段识别
基于 Last_Login_Days_Ago（R）和 Purchase_Frequency（F）：

阶段	条件
🆕 新用户-14天内	F == 1 且 R ≤ 14
🌱 成长期	R ≤ 14 且不满足新用户/成熟期
🌳 成熟期	R ≤ 7 且 F ≥ 5
⚠️ 预流失用户	15 ≤ R ≤ 21
💔 流失用户	R ≥ 22
❓ 未知	其他
生成 Lifecycle_Stage_Label

顺序编码 Stage_Order：新用户→1，成长期→2，成熟期→3，预流失→4，流失→5，未知→9

📊 4. RFM 分层
4.1 评分规则
R_Score：Last_Login_Days_Ago 倒序五等分（5=最近，1=最久）

F_Score：Purchase_Frequency 五等分（1=最低，5=最高）

M_Score：Total_Spending 五等分（1=最低，5=最高）

RFM_Score = R_Score + F_Score + M_ScoreRFM_Score = R_Score

4.2 分层映射
根据 R、F、M 组合 → 8 类细分标签 → 5 类宏观分层：

宏观分层	包含的细分标签
👑 核心用户	重要价值用户
⭐ 重要用户	重要发展用户、重要保持用户、重要挽留用户
🌱 潜力用户	一般价值用户
👥 一般用户	一般维持用户、一般发展用户
💀 流失用户	流失用户
📤 5. 数据输出
导出 user_tags.csv，包含字段：

python
cols = [
    "User_ID",
    "Age_Band","Gender_Std","Location","Product_Category_Preference","Interests",
    "Last_Login_Days_Ago","Purchase_Frequency","Average_Order_Value", "Spending_Ability_Tag",
    "Total_Spending","Purchasing_Power_Tag","Lifecycle_Stage_Label","Stage_Order",
    "Macro_Segment","Customer_Segment"
]
📈 6. 衍生指标与聚合分析
6.1 活跃度指标
python
df["active_7d"]  = (df["Last_Login_Days_Ago"] <= 7).astype(int)
df["active_14d"] = (df["Last_Login_Days_Ago"] <= 14).astype(int)
df["active_30d"] = (df["Last_Login_Days_Ago"] <= 30).astype(int)
6.2 核心聚合函数 agg_core
计算每个分组：

用户数（去重）

AOV 均值

总消费总额

F 均值

近7/14/30天活跃率

6.3 单维分布
📍 按 Age_Band / Gender_Std / Location 分别输出上述指标

6.4 二维交叉分析
RFM 宏观分层 × 年龄段/性别/地区

生命周期 × 年龄段/性别/地区

计算：组内占比、整体占比

📊 7. 可视化图表一览
图表类型	内容	用途
📊 柱状图	RFM 8 类人数、5 类宏观分层饼图、年龄段/性别/地区用户数	规模分布
📊 堆叠柱状图	RFM × 年龄段（组内占比）	分层结构
🌡️ 热力图	生命周期 × 地区人数、RFM × 兴趣渗透率、生命周期 × 活跃度	交叉密度
📊 分组柱状图	RFM × 消费能力、生命周期 × 消费/购买能力	价值对比
📈 专门图表	生命周期 F 均值对比、生命周期 × 活跃度分档热力图	行为诊断
💡 所有图表均通过 Matplotlib 生成，支持中文显示（SimHei 字体）。

🎯 8. 偏好与兴趣分析
整体 Top5：商品品类 & 兴趣标签（柱状图）

分组 Top3：按 RFM 宏观分层、生命周期、年龄段分别展示品类/兴趣的组内占比

渗透率矩阵：

🔥 RFM × 兴趣渗透率 → 用于跨品类推荐

🔥 生命周期 × 品类渗透率 → 用于阶段化营销

🏃 9. 行为分档与活跃度分析
9.1 活跃度分档
python
def map_active_band(x):
    if x <= 7: return "高活跃(≤7天)"if x <= 7: return "高活跃(≤7天)"
    elif x <= 14: return "中高活跃(8–14天)"elif x <= 14: return "中高活跃(8–14天)"
    elif x <= 30: return "中活跃(15–30天)"elif x <= 30: return "中活跃(15–30天)"
    else: return "低活跃(>30天)"   else: return "低活跃(>30天)"
9.2 输出图表
📊 生命周期阶段 × 购买频次（F 均值） 柱状图

🌡️ 生命周期 × 活跃度分档 人数热力图
→ 识别不同阶段的活跃度构成，预警高价值但活跃下滑的用户。

🧪 运行环境
text   文本
Python 3.7+   Python 3.7
pandas   熊猫
numpy
matplotlib
🔤 中文字体配置：

python
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = Falseplt.rcParams['轴。unicode_minus'] = Falseplt.rcParams['轴。unicode_minus‘] = false . rcparams[’轴。unicode_minus'] = False
📂 输入 / 输出
类型	路径	说明
输入	user_personalized_features.csv	用户基础行为与属性
输出	user_tags.csv	带有全部分层标签的用户表
图表	plt.show() 实时显示	所有分析图表
💎 应用场景
✅ 用户精细化分层（RFM + 生命周期）

✅ 流失预警与召回策略（预流失/流失用户）

✅ 个性化推荐（兴趣渗透率矩阵）

✅ 营销活动定向（活跃度+消费能力）
