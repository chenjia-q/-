import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv(r'E:\抖音电商用户分层与画像分析\user_personalized_features.csv')

# 删除无用列
df=df.drop(columns=['Unnamed: 0','Unnamed: 0.1'],errors='ignore')
df.head()

# 检查重复值
df.duplicated().sum() #无重复值
# 检查缺失值
df.isna().sum() # 无缺失值

num_cols = ["Age","Income","Last_Login_Days_Ago","Time_Spent_on_Site_Minutes","Pages_Viewed","Purchase_Frequency","Average_Order_Value","Total_Spending"]
for col in num_cols:
        df[col]=pd.to_numeric(df[col])
df["Newsletter_Subscription"] = df["Newsletter_Subscription"].astype(bool)
# 将纯文本类型转换为category
text_category = ["Gender","Location","Interests","Product_Category_Preference"]
for c in text_category:
    df[c]=df[c].astype('category')

#基础属性标签
# 年龄段
bins = [18, 25, 35, 45, 150]
labels = ["18-24", "25-34", "35-44", "45+"]
df["Age_Band"] = pd.cut(df["Age"], bins=bins, labels=labels)

# 性别标准化
gender_map = {"Male": "男", "Female": "女"}
df["Gender_Std"] = df["Gender"].str.strip().map(gender_map)

# 消费力 = AOV 三分位（低/中/高）
df["Spending_Ability_Tag"] = pd.qcut(
    df["Average_Order_Value"].rank(method="first"),#防止并列较多导致边界冲突
    q=[0, 1/3, 2/3, 1],
    labels=["低","中","高"]
)

# 购买力 = GMV 三分位（低/中/高）
gmv = df["Total_Spending"]
df["Purchasing_Power_Tag"] = pd.qcut(
    gmv.rank(method="first"),
    q=[0, 1/3, 2/3, 1],
    labels=["低","中","高"]
)

# 生命周期标签
def lifecycle_label(r, f, r1=14, r_active=7, f_mature=5):
    # 新用户-14天内
    if pd.notna(f) and pd.notna(r) and (f == 1) and (r <= r1):
        return "新用户-14天内"
    # 成熟期（高频且近7天）
    if pd.notna(f) and pd.notna(r) and (r <= r_active) and (f >= f_mature):
        return "成熟期"
    # 成长期（≤14天且未达成熟/新）
    if pd.notna(r) and r <= r1:
        return "成长期"
    # 预流失（15~21天）
    if pd.notna(r) and 15 <= r <= 21:
        return "预流失用户"
    # 流失（≥22天）
    if pd.notna(r) and r >= 22:
        return "流失用户"
    return "未知"

df["Lifecycle_Stage_Label"] = df.apply(
    lambda x: lifecycle_label(x.get("Last_Login_Days_Ago"), x.get("Purchase_Frequency")), axis=1
)
order_map = {"新用户-14天内":1, "成长期":2, "成熟期":3, "预流失用户":4, "流失用户":5, "未知":9}
df["Stage_Order"] = df["Lifecycle_Stage_Label"].map(order_map).fillna(9).astype(int)

#RFM 分层
#得到R、F、M值
df['R_Score']=pd.qcut(df['Last_Login_Days_Ago'],q=5,labels=[5,4,3,2,1])
df['F_Score']=pd.qcut(df['Purchase_Frequency'],q=5,labels=[1,2,3,4,5])
df['M_Score']=pd.qcut(df['Total_Spending'],q=5,labels=[1,2,3,4,5])
#转化数值类型--pd.qcut() 返回的是分类（Categorical）数据类型
df['R_Score'] = df['R_Score'].astype(int)
df['F_Score'] = df['F_Score'].astype(int)
df['M_Score'] = df['M_Score'].astype(int)
# 合并RFM得分
df['RFM_Score']=df['R_Score'] + df['F_Score'] + df['M_Score']
# 构建用户分层函数
def Customer_Segment (r,f,m):
    # 1. 三高：重要价值用户
    if r >= 4 and f >= 4 and m >= 4:
        return '重要价值用户'
    # 2. R&M高，F低：重要发展用户
    if r >= 4 and m >= 4 and f <= 3:
        return '重要发展用户'
    # 3. F&M高，R低：重要保持用户
    if f >= 4 and m >= 4 and r <= 3:
        return '重要保持用户'
    # 4. R&F低，M高：重要挽留用户
    if r <= 3 and f <= 3 and m>= 4:
        return '重要挽留用户'
    # 5. R&F高，M低：一般价值用户
    if r >= 4 and f >= 4 and m <= 2:
        return '一般价值用户'
    # 6. R高，但F&M低：一般发展用户
    if r >= 4 and (f <= 2 or m <= 2):
        return '一般发展用户'
    # 7. 三低：流失用户
    if r <= 2 and f <= 2 and m <= 2:
        return '流失用户'
    # 8. 其他：一般维持用户
    return '一般维持用户'

# 5类分层映射
df['Customer_Segment'] = df.apply(
    lambda x: Customer_Segment(x['R_Score'], x['F_Score'], x['M_Score']),
    axis=1
)
macro_map={
    '重要价值用户':'核心用户',
    '重要发展用户':'重要用户',
    '重要保持用户':'重要用户',
    '重要挽留用户':'重要用户',
    '一般价值用户':'潜力用户',
    '一般维持用户':'一般用户',
    '一般发展用户':'一般用户',
    '流失用户':'流失用户',
}
df['Macro_Segment']=df['Customer_Segment'].map(macro_map)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#人数柱状图
order = ['重要价值用户','重要发展用户','重要保持用户','重要挽留用户',
          '一般价值用户','一般维持用户','一般发展用户','流失用户']
cnt8 = df['Customer_Segment'].value_counts().reindex(order).astype(int)
ax = cnt8.plot(kind='bar')
ax.set_title('RFM  - 人数分布')
ax.set_xlabel('类别')
ax.set_ylabel('人数')
for i, v in enumerate(cnt8.values):
    ax.text(i, v, str(v), ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

#占比饼图
cnt5 = df['Macro_Segment'].value_counts()
plt.figure()
plt.pie(cnt5.values, labels=cnt5.index, autopct='%1.1f%%', startangle=90)
plt.title('5类分层 - 占比')
plt.tight_layout()
plt.show()

#导出新表
cols = [
    "User_ID",
    "Age_Band","Gender_Std","Location","Product_Category_Preference","Interests",
    "Last_Login_Days_Ago","Purchase_Frequency","Average_Order_Value", "Spending_Ability_Tag",
    "Total_Spending","Purchasing_Power_Tag","Lifecycle_Stage_Label","Stage_Order",'Macro_Segment','Customer_Segment',
]
out = df[cols].copy()
out.to_csv(r"E:\抖音电商用户分层与画像分析\user_tags.csv")

df = pd.read_csv(r"E:\抖音电商用户分层与画像分析\user_tags.csv")

# 活跃指标
df["active_7d"]  = (df["Last_Login_Days_Ago"] <= 7).astype(int)
df["active_14d"] = (df["Last_Login_Days_Ago"] <= 14).astype(int)
df["active_30d"] = (df["Last_Login_Days_Ago"] <= 30).astype(int)

u = df["User_ID"].nunique()

def agg_core(gb):
    return gb.agg(
        用户数=("User_ID","nunique"),
        AOV均值=("Average_Order_Value","mean"),
        总消费总额=("Total_Spending","sum"),
        F均值=("Purchase_Frequency","mean"),
        近7天活跃率=("active_7d","mean"),
        近14天活跃率=("active_14d","mean"),
        近30天活跃率=("active_30d","mean"),
    )
def add_derived(t):
    t = t.copy()
    t["人均总消费"] = t["总消费总额"] / t["用户数"].replace(0, pd.NA)
    t["占比"] = t["用户数"] / u if u else pd.NA
    return t

# 单维分布
def one_dim(col):
    t = agg_core(df.groupby(col)).reset_index().rename(columns={col:"维度值"})
    return add_derived(t).sort_values("用户数", ascending=False)

summary_age    = one_dim("Age_Band")
summary_gender = one_dim("Gender_Std")
summary_loc    = one_dim("Location")

# 分层对比（RFM 5类分层 / 生命周期）
def two_level(grp, cat):
    t = agg_core(df.groupby([grp,cat])).reset_index().rename(columns={grp:"分层", cat:"维度值"})
    t = add_derived(t)
    denom = t.groupby("分层")["用户数"].transform("sum").replace(0, pd.NA)  # 该分层总人数
    t["组内占比"]   = t["用户数"] / denom
    t["组占比（整体）"] = denom / u if u else pd.NA
    return t.sort_values(["分层","用户数"], ascending=[True, False])

# 5类分层 × 人口属性
by_rfm_age    = two_level("Macro_Segment","Age_Band")
by_rfm_gender = two_level("Macro_Segment","Gender_Std")
by_rfm_loc    = two_level("Macro_Segment","Location")

by_lc_age     = two_level("Lifecycle_Stage_Label","Age_Band")
by_lc_gender  = two_level("Lifecycle_Stage_Label","Gender_Std")
by_lc_loc     = two_level("Lifecycle_Stage_Label","Location")


# 关键交叉
cross_rfm_age = (df.groupby(["Macro_Segment","Age_Band"], observed=True)["User_ID"]
                   .nunique().unstack("Age_Band", fill_value=0).astype(int))

cross_lc_loc  = (df.groupby(["Lifecycle_Stage_Label","Location"], observed=True)["User_ID"]
                   .nunique().unstack("Location", fill_value=0).astype(int))
if order_map:
    cross_lc_loc = cross_lc_loc.reindex(pd.Series(order_map).sort_values().index)

#  图表
# 单维柱状图（人数）
def plot_bar(d, title):
    x = d["维度值"].astype(str)
    y = d["用户数"]
    plt.figure()
    plt.bar(x, y)
    plt.title(title); plt.xlabel("分类"); plt.ylabel("用户数")
    for i, v in enumerate(y):
        plt.text(i, v, int(v), ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.show()

plot_bar(summary_age,    "年龄段分布（用户数）")
plot_bar(summary_gender, "性别分布（用户数）")
plot_bar(summary_loc,    "地区分布（用户数）")


# RFM × 年龄段（组内占比堆叠柱状）
def plot_stacked_two_level(df_two, grp_col_name, cat_col_name, title):
    pv = df_two.pivot(index="分层", columns="维度值", values="组内占比").fillna(0)
    ax = pv.plot(kind="bar", stacked=True, rot=0, title=title)
    ax.set_xlabel(grp_col_name); ax.set_ylabel("组内占比")
    plt.tight_layout(); plt.show()

plot_stacked_two_level(by_rfm_age, "RFM 宏分层", "年龄段", "RFM × 年龄段（组内占比）")

# 生命周期 × 地区（人数热力图）
def plot_heatmap_counts(pvt, title, xlab, ylab):
    M = pvt.values
    plt.figure()
    im = plt.imshow(M, aspect="auto")
    plt.title(title); plt.xlabel(xlab); plt.ylabel(ylab)
    plt.xticks(range(M.shape[1]), list(pvt.columns))
    plt.yticks(range(M.shape[0]), list(pvt.index))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.show()

plot_heatmap_counts(cross_lc_loc,  "生命周期 × 地区（人数热力图）", "地区",   "生命周期阶段")

#  分层对比：RFM / 生命周期
def two_level_value(grp, cat):
    t = agg_core(df.groupby([grp, cat])).reset_index().rename(columns={grp:"分层", cat:"维度值"})
    t = add_derived(t)
    denom = t.groupby("分层")["用户数"].transform("sum").replace(0, pd.NA)
    t["组内占比"] = t["用户数"] / denom
    t["组占比（整体）"] = denom / u if u else pd.NA
    return t.sort_values(["分层","用户数"], ascending=[True, False])

by_rfm_spend_tag = two_level_value("Macro_Segment",        "Spending_Ability_Tag")
by_rfm_power_tag = two_level_value("Macro_Segment",        "Purchasing_Power_Tag")
by_lc_spend_tag  = two_level_value("Lifecycle_Stage_Label","Spending_Ability_Tag")
by_lc_power_tag  = two_level_value("Lifecycle_Stage_Label","Purchasing_Power_Tag")

# 交叉分析：RFM × AOV 价格带；生命周期 × 总消费等级
# RFM × AOV价格带
try:
    df["AOV_Bin"] = pd.qcut(df["Average_Order_Value"], q=5, duplicates="drop")
except Exception:
    df["AOV_Bin"] = pd.qcut(df["Average_Order_Value"], q=4, duplicates="drop")

cross_rfm_aovbin = (
    df.groupby(["Macro_Segment", "AOV_Bin"], observed=True)["User_ID"]
      .nunique()
      .unstack("AOV_Bin", fill_value=0)
      .astype(int)
)

# 生命周期 × 总消费等级
try:
    df["Spend_Tier"] = pd.qcut(df["Total_Spending"], q=5, duplicates="drop")
except Exception:
    df["Spend_Tier"] = pd.qcut(df["Total_Spending"], q=4, duplicates="drop")

cross_lc_spendtier = (
    df.groupby(["Lifecycle_Stage_Label", "Spend_Tier"], observed=True)["User_ID"]
      .nunique()
      .unstack("Spend_Tier", fill_value=0)
      .astype(int)
)

def plot_stacked_value(two_level_df, title):
    """
    two_level_df 来自 two_level_value，
    需要包含列：分层、维度值、组内占比
    """
    pv = (two_level_df
          .pivot(index="分层", columns="维度值", values="组内占比")
          .fillna(0))
    ax = pv.plot(kind="bar", stacked=True, rot=0, title=title)
    ax.set_xlabel("")
    ax.set_ylabel("组内占比")   # 每个分层内部的比例
    plt.tight_layout()
    plt.show()

#  分层堆叠柱状图：RFM / 生命周期下的价值标签（组内占比）
plot_stacked_value(by_rfm_spend_tag, "RFM × 消费能力标签（组内占比）")
plot_stacked_value(by_lc_spend_tag,  "生命周期 × 消费能力标签（组内占比）")
plot_stacked_value(by_lc_power_tag,  "生命周期 × 购买能力标签（组内占比）")

# 1) 单维分布：各生命周期阶段规模 & 价值/活跃差异
life_summary = (
    agg_core(df.groupby("Lifecycle_Stage_Label", observed=True))
    .reset_index()
    .rename(columns={"Lifecycle_Stage_Label": "维度值"})
)
life_summary = add_derived(life_summary)


#   维度值 + 用户数、占比、人均总消费、AOV均值、
#   F均值、近7/14/30天活跃率
# 用户数柱状图：生命周期阶段分布
def plot_bar_life(d, title):
    if d is None or d.empty:
        return
    x = d["维度值"].astype(str)
    y = d["用户数"]
    plt.figure()
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel("生命周期阶段")
    plt.ylabel("用户数")
    for i, v in enumerate(y):
        plt.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()


plot_bar_life(life_summary, "生命周期阶段分布（用户数）")

# 生命周期 × RFM 宏分层
by_life_rfm = two_level("Lifecycle_Stage_Label", "Macro_Segment")


def plot_stacked_life_rfm(two_level_df, title):
    if two_level_df is None or two_level_df.empty:
        return
    pv = (two_level_df
          .pivot(index="分层", columns="维度值", values="组内占比")
          .fillna(0))
    ax = pv.plot(kind="bar", stacked=True, rot=0, title=title)
    ax.set_xlabel("生命周期阶段")
    ax.set_ylabel("组内占比")
    plt.tight_layout()
    plt.show()


plot_stacked_life_rfm(by_life_rfm, "生命周期 × RFM 宏分层（组内占比）")

summary_cat  = one_dim("Product_Category_Preference")
summary_int  = one_dim("Interests")

# —— 按 RFM 宏分层看 Top 品类 / 兴趣
by_rfm_cat = two_level("Macro_Segment",        "Product_Category_Preference")
by_rfm_int = two_level("Macro_Segment",        "Interests")

# —— 按生命周期看 Top 品类 / 兴趣
by_life_cat = two_level("Lifecycle_Stage_Label","Product_Category_Preference")
by_life_int = two_level("Lifecycle_Stage_Label","Interests")

# —— 按年龄看 Top 品类 / 兴趣
by_age_cat = two_level("Age_Band","Product_Category_Preference")
by_age_int = two_level("Age_Band","Interests")

# 交叉：生命周期 × 品类；RFM × 兴趣（渗透率矩阵）

# 生命周期 × 品类：每个生命周期阶段内，各品类渗透率
life_total = df.groupby("Lifecycle_Stage_Label", observed=True)["User_ID"].nunique()

life_cat = (
    df.groupby(["Lifecycle_Stage_Label", "Product_Category_Preference"], observed=True)["User_ID"]
      .nunique()
      .reset_index(name="用户数")
)
life_cat["生命周期总用户"] = life_cat["Lifecycle_Stage_Label"].map(life_total)
life_cat["用户数"] = life_cat["用户数"].astype("float")
life_cat["生命周期总用户"] = life_cat["生命周期总用户"].astype("float")
life_cat["渗透率"] = life_cat["用户数"] / life_cat["生命周期总用户"].replace(0, pd.NA)

# 渗透率透视表：生命周期 × 品类
life_cat_pen = (
    life_cat
    .pivot(index="Lifecycle_Stage_Label",
           columns="Product_Category_Preference",
           values="渗透率")
    .fillna(0)
)

# RFM × 兴趣：每个 RFM 分层内，各兴趣渗透率（用于推荐 / 召回）
rfm_total = df.groupby("Macro_Segment", observed=True)["User_ID"].nunique()

rfm_int = (
    df.groupby(["Macro_Segment", "Interests"], observed=True)["User_ID"]
      .nunique()
      .reset_index(name="用户数")
)

rfm_int["RFM总用户"] = rfm_int["Macro_Segment"].map(rfm_total)
rfm_int["用户数"] = rfm_int["用户数"].astype("float")
rfm_int["RFM总用户"] = rfm_int["RFM总用户"].astype("float")
rfm_int["渗透率"] = rfm_int["用户数"] / rfm_int["RFM总用户"].replace(0, pd.NA)

rfm_int_pen = (
    rfm_int
    .pivot(index="Macro_Segment",
           columns="Interests",
           values="渗透率")
    .fillna(0)
)

# 单维柱状：整体 Top 品类 / 兴趣（可以只画品类）
def plot_bar_pref(d, title):
    if d is None or d.empty:
        return
    x = d["维度值"].astype(str)
    y = d["用户数"]
    plt.figure()
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel("类别")
    plt.ylabel("用户数")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_bar_pref(summary_cat.head(5), "整体 Top5 商品品类（用户数）")

# 堆叠柱状：按生命周期 / RFM 看 Top 品类/兴趣的组内占比
def plot_stacked_pref(two_level_df, title):
    if two_level_df is None or two_level_df.empty:
        return
    pv = (two_level_df
          .pivot(index="分层", columns="维度值", values="组内占比")
          .fillna(0))
    ax = pv.plot(kind="bar", stacked=True, rot=0, title=title)
    ax.set_xlabel("分层")
    ax.set_ylabel("组内占比")
    plt.tight_layout()
    plt.show()


# 举例：取每个阶段 Top3 品类再画
life_cat_top3 = by_life_cat.groupby("分层", observed=True).head(3)
plot_stacked_pref(life_cat_top3, "生命周期 × Top3 商品品类（组内占比）")

age_cat_top3 = by_age_cat.groupby("分层", observed=True).head(3)
plot_stacked_pref(age_cat_top3, "各年龄段 × Top3 商品品类（组内占比）")


# 热力图：RFM × 兴趣渗透率
def plot_heatmap_pref(pvt, title, xlab, ylab, fmt='.2f'):
    if pvt is None or pvt.empty:
        return
    M = pvt.values
    plt.figure()
    # 使用颜色深浅对比更明显的 colormap
    im = plt.imshow(M, aspect="auto", cmap='autumn')

    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xticks(range(M.shape[1]), list(pvt.columns), rotation=45)
    plt.yticks(range(M.shape[0]), list(pvt.index))

    # 获取颜色映射范围，用于智能判断文字颜色
    vmin, vmax = M.min(), M.max()
    mid = (vmin + vmax) / 2

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            # 根据数值与中位数的对比决定文字颜色
            text_color = 'black' if val > mid else 'white'
            plt.text(j, i, format(val, fmt),
                     ha="center", va="center",
                     color=text_color, fontsize=8,
                     weight='bold')  # 加粗让文字更清晰

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


plot_heatmap_pref(rfm_int_pen,
                  "RFM × 兴趣标签渗透率",
                  "兴趣标签", "RFM 宏分层")

#  行为分档：频次 + 活跃度
df["Purchase_Frequency"] = pd.to_numeric(df["Purchase_Frequency"], errors="coerce")
df["Last_Login_Days_Ago"] = pd.to_numeric(df["Last_Login_Days_Ago"], errors="coerce")

#  活跃度分档（基于最近登录天数）
def map_active_band(x):
    if pd.isna(x):
        return "未知"
    if x <= 7:
        return "高活跃(≤7天)"
    elif x <= 14:
        return "中高活跃(8–14天)"
    elif x <= 30:
        return "中活跃(15–30天)"
    else:
        return "低活跃(>30天)"

df["Active_Band"] = df["Last_Login_Days_Ago"].apply(map_active_band)

#  分层： RFM/生命周期下的频次与活跃
# RFM 维度：看每个分层的 F 均值 + 活跃率
by_rfm_behavior = (
    agg_core(df.groupby("Macro_Segment", observed=True))
    .reset_index()
    .rename(columns={"Macro_Segment": "分层"})
)
by_rfm_behavior = add_derived(by_rfm_behavior)
# 包含：用户数/占比、人均总消费、F均值、近7/14/30天活跃率

# 生命周期维度：各阶段 F 均值 + 活跃率
by_life_behavior = (
    agg_core(df.groupby("Lifecycle_Stage_Label", observed=True))
    .reset_index()
    .rename(columns={"Lifecycle_Stage_Label": "分层"})
)
by_life_behavior = add_derived(by_life_behavior)
# 包含：各生命周期阶段的规模、F均值、活跃率差异

# 生命周期阶段 F 均值（频次差异）
def plot_bar_F_by_life(d):
    if d is None or d.empty:
        return
    x = d["分层"].astype(str)
    y = d["F均值"]
    plt.figure()
    plt.bar(x, y)
    plt.title("生命周期阶段 × 购买频次（F均值）")
    plt.xlabel("生命周期阶段")
    plt.ylabel("F均值")
    for i, v in enumerate(y):
        plt.text(i, v, round(v, 2) if pd.notna(v) else "", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()

plot_bar_F_by_life(by_life_behavior)
