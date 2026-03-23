import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Python数据可视化教学平台",
    page_icon="📊",
    layout="wide"
)

# =========================
# GitHub 仓库配置
# =========================
GITHUB_USER = "fanghua-2018"
GITHUB_REPO = "Python_Visualization"
GITHUB_BRANCH = "main"

# 你的 data 文件夹在仓库根目录下
BASE_RAW_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/data"

# 数据集文件映射
DATASET_URLS = {
    "iris": f"{BASE_RAW_URL}/iris.csv",
    "tips": f"{BASE_RAW_URL}/tips.csv",
    "penguins": f"{BASE_RAW_URL}/penguins.csv",
    "titanic": f"{BASE_RAW_URL}/titanic.csv",
    "flights": f"{BASE_RAW_URL}/flights.csv",
    "diamonds": f"{BASE_RAW_URL}/diamonds.csv",
    "exercise": f"{BASE_RAW_URL}/exercise.csv",
}

# =========================
# 基础函数
# =========================
@st.cache_data
def load_dataset(name: str) -> pd.DataFrame:
    """从 GitHub 仓库加载 CSV 数据集"""
    url = DATASET_URLS.get(name)
    if not url:
        return pd.DataFrame()

    try:
        df = pd.read_csv(url)
        df = try_parse_datetime(df)
        return df
    except Exception as e:
        st.error(f"无法加载数据集 {name}，请检查 GitHub 路径或文件是否存在。错误信息：{e}")
        return pd.DataFrame()


def get_dataset_description(name: str):
    """数据集说明"""
    descriptions = {
        "iris": {
            "简介": "鸢尾花数据集，包含3类鸢尾花的花萼与花瓣测量值，是分类分析的经典示例。",
            "适用场景": "分类比较、变量关系分析、分布分析",
            "推荐图形": "散点图、直方图、箱线图、柱状图"
        },
        "tips": {
            "简介": "餐厅顾客消费与小费数据，包含总消费、小费、性别、星期、就餐时段等变量。",
            "适用场景": "分类统计、分组比较、关系分析",
            "推荐图形": "柱状图、饼图、散点图、箱线图"
        },
        "penguins": {
            "简介": "企鹅数据集，记录不同岛屿企鹅的嘴峰长度、体重、鳍长度及物种信息。",
            "适用场景": "多变量分析、分类分析、变量关系分析",
            "推荐图形": "散点图、箱线图、直方图"
        },
        "titanic": {
            "简介": "泰坦尼克号乘客数据，包含性别、年龄、舱位、票价及是否生还等信息。",
            "适用场景": "分类比较、生存分析、分组统计",
            "推荐图形": "柱状图、饼图、箱线图、直方图"
        },
        "flights": {
            "简介": "航班乘客数据，记录不同年份、月份的乘客数量，适合时间序列分析。",
            "适用场景": "趋势分析、时间变化分析",
            "推荐图形": "折线图、热力图、柱状图"
        },
        "diamonds": {
            "简介": "钻石数据集，包含价格、克拉、切工、颜色、净度等变量，适合综合分析。",
            "适用场景": "关系分析、分类比较、分布分析",
            "推荐图形": "散点图、箱线图、直方图、柱状图"
        },
        "exercise": {
            "简介": "运动实验数据，包含不同时间、饮食和运动状态下的脉搏变化。",
            "适用场景": "时间变化、分类比较、实验数据分析",
            "推荐图形": "折线图、柱状图、箱线图"
        }
    }
    return descriptions.get(name, {"简介": "无说明", "适用场景": "无", "推荐图形": "无"})


def get_column_types(df: pd.DataFrame):
    """识别列类型"""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    category_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]
    return numeric_cols, category_cols, datetime_cols


def describe_numeric(series: pd.Series) -> pd.DataFrame:
    """数值变量描述统计"""
    s = series.dropna()
    if s.empty:
        return pd.DataFrame({"统计量": [], "值": []})

    stats = {
        "样本数": s.count(),
        "均值": s.mean(),
        "中位数": s.median(),
        "标准差": s.std(),
        "最小值": s.min(),
        "25%分位数": s.quantile(0.25),
        "50%分位数": s.quantile(0.50),
        "75%分位数": s.quantile(0.75),
        "最大值": s.max(),
        "偏度": s.skew(),
        "峰度": s.kurt()
    }
    return pd.DataFrame({
        "统计量": list(stats.keys()),
        "值": list(stats.values())
    })


def describe_category(series: pd.Series) -> pd.DataFrame:
    """分类变量频数统计"""
    vc = series.value_counts(dropna=False)
    pct = series.value_counts(dropna=False, normalize=True) * 100
    return pd.DataFrame({
        "类别": vc.index.astype(str),
        "频数": vc.values,
        "占比(%)": pct.values.round(2)
    })


def safe_sort_dataframe(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    try:
        return df.sort_values(by=x_col)
    except Exception:
        return df


def get_chart_guide():
    """不同图形对应目的与数据类型"""
    return pd.DataFrame({
        "可视化方法": ["折线图", "柱状图", "饼图", "散点图", "直方图", "箱线图"],
        "主要目的": [
            "观察趋势与变化过程",
            "比较不同类别的数量或统计量差异",
            "展示各类别占整体的比例结构",
            "分析两个变量之间的关系、聚类与异常点",
            "分析单变量分布特征",
            "比较分布、中位数、四分位数和异常值"
        ],
        "适用数据类型": [
            "时间序列、顺序变量 + 数值变量",
            "分类变量 + 数值变量 / 分类变量 + 频数",
            "单一分类变量 + 占比/频数",
            "两个数值变量，可叠加分类变量",
            "单个数值变量",
            "数值变量，可按分类变量分组"
        ],
        "典型问题": [
            "某指标随时间如何变化？",
            "不同类别之间谁更大？差多少？",
            "各类别各占多少比例？",
            "两个变量是否相关？是否有异常值？",
            "变量是否偏态？是否近似正态？",
            "不同组数据分布是否不同？是否存在离群值？"
        ]
    })


def try_parse_datetime(df: pd.DataFrame):
    """尝试自动识别时间列"""
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                converted = pd.to_datetime(df[col], errors="raise")
                if len(df[col]) > 0 and converted.notna().sum() / len(df[col]) > 0.8:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df


# =========================
# 标题
# =========================
st.title("Python 数据可视化教学平台")
st.markdown("""
本平台基于 **Streamlit + Plotly** 构建，用于演示常见数据可视化方法。  
当前版本支持：
- 从 GitHub 仓库 data 文件夹在线读取数据
- 上传本地 CSV / Excel 数据
- 数据集及变量说明
- 可视化方法—研究目的—数据类型对照说明
""")

# =========================
# 侧边栏：数据集选择
# =========================
st.sidebar.header("一、数据集设置")

data_source = st.sidebar.radio("数据来源", ["GitHub数据集", "上传本地数据"])

if data_source == "GitHub数据集":
    dataset_name = st.sidebar.selectbox(
        "选择数据集",
        ["iris", "tips", "penguins", "titanic", "flights", "diamonds", "exercise"]
    )
    df = load_dataset(dataset_name).copy()
    dataset_desc = get_dataset_description(dataset_name)
    dataset_label = dataset_name
else:
    uploaded_file = st.sidebar.file_uploader("上传本地数据文件", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        df = try_parse_datetime(df)
        dataset_name = "本地上传数据"
        dataset_desc = {
            "简介": "用户上传的本地数据，可用于自定义可视化分析。",
            "适用场景": "数据探索、课堂演示、自定义分析",
            "推荐图形": "根据变量类型自动选择"
        }
        dataset_label = uploaded_file.name
    else:
        st.info("请在左侧上传 CSV 或 Excel 文件。")
        st.stop()

if df.empty:
    st.warning("数据为空，请检查 GitHub 仓库中的 data 文件夹及 CSV 文件。")
    st.stop()

# 删除全空行
df = df.dropna(how="all")
numeric_cols, category_cols, datetime_cols = get_column_types(df)
all_cols = df.columns.tolist()

# =========================
# 数据集说明
# =========================
st.header("1. 数据集说明")

c1, c2 = st.columns([1, 2])
with c1:
    st.metric("数据集名称", dataset_label)
    st.metric("样本数", df.shape[0])
    st.metric("变量数", df.shape[1])
with c2:
    st.write(f"**简介：** {dataset_desc['简介']}")
    st.write(f"**适用场景：** {dataset_desc['适用场景']}")
    st.write(f"**推荐图形：** {dataset_desc['推荐图形']}")

# =========================
# 变量说明
# =========================
st.header("2. 变量说明")

var_info = pd.DataFrame({
    "变量名": df.columns,
    "数据类型": [str(df[col].dtype) for col in df.columns],
    "变量类别": [
        "数值变量" if col in numeric_cols else ("时间变量" if col in datetime_cols else "分类变量")
        for col in df.columns
    ],
    "缺失值个数": [df[col].isnull().sum() for col in df.columns],
    "非空样本数": [df[col].notnull().sum() for col in df.columns],
    "示例值": [str(df[col].dropna().iloc[0]) if df[col].dropna().shape[0] > 0 else "无" for col in df.columns]
})
st.dataframe(var_info, use_container_width=True)

# =========================
# 数据概览
# =========================
st.header("3. 数据概览")

col_a, col_b, col_c = st.columns(3)
col_a.metric("数值变量数", len(numeric_cols))
col_b.metric("分类变量数", len(category_cols))
col_c.metric("缺失值总数", int(df.isnull().sum().sum()))

with st.expander("查看原始数据"):
    st.dataframe(df, use_container_width=True)

with st.expander("查看缺失值统计"):
    null_df = pd.DataFrame({
        "字段名": df.columns,
        "缺失值个数": df.isnull().sum().values,
        "缺失率(%)": (df.isnull().sum().values / len(df) * 100).round(2)
    })
    st.dataframe(null_df, use_container_width=True)

# =========================
# 可视化方法说明
# =========================
st.header("4. 不同可视化方法对应的研究目的与数据类型")
st.dataframe(get_chart_guide(), use_container_width=True)

# =========================
# 数据特征描述
# =========================
st.header("5. 数据类型与分布特征描述")

tab1, tab2, tab3 = st.tabs(["数值变量", "分类变量", "相关性分析"])

with tab1:
    if numeric_cols:
        selected_num = st.selectbox("选择数值变量", numeric_cols, key="selected_num")
        st.subheader(f"数值变量：{selected_num}")
        st.dataframe(describe_numeric(df[selected_num]), use_container_width=True)

        fig_num_hist = px.histogram(
            df,
            x=selected_num,
            nbins=20,
            marginal="box",
            title=f"{selected_num} 的分布特征"
        )
        fig_num_hist.update_layout(height=450)
        st.plotly_chart(fig_num_hist, use_container_width=True)
    else:
        st.info("当前数据集中没有数值变量。")

with tab2:
    if category_cols:
        selected_cat = st.selectbox("选择分类变量", category_cols, key="selected_cat")
        st.subheader(f"分类变量：{selected_cat}")
        st.dataframe(describe_category(df[selected_cat]), use_container_width=True)

        fig_cat_bar = px.bar(
            describe_category(df[selected_cat]),
            x="类别",
            y="频数",
            text="频数",
            title=f"{selected_cat} 的类别频数统计"
        )
        fig_cat_bar.update_layout(height=450)
        st.plotly_chart(fig_cat_bar, use_container_width=True)
    else:
        st.info("当前数据集中没有分类变量。")

with tab3:
    if len(numeric_cols) >= 2:
        corr_method = st.selectbox("选择相关系数方法", ["pearson", "spearman"], key="corr_method")
        corr_df = df[numeric_cols].corr(method=corr_method)

        st.dataframe(corr_df.round(3), use_container_width=True)

        fig_corr = px.imshow(
            corr_df.round(3),
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title=f"数值变量相关系数矩阵（{corr_method}）"
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("至少需要两个数值变量才能进行相关性分析。")

# =========================
# 可视化分析模块
# =========================
st.header("6. 可视化分析实验")

st.sidebar.header("二、图表设置")
chart_type = st.sidebar.selectbox(
    "选择图表类型",
    ["折线图", "柱状图", "饼图", "散点图", "直方图", "箱线图"]
)

chart_guide_df = get_chart_guide()
selected_guide = chart_guide_df[chart_guide_df["可视化方法"] == chart_type].iloc[0]
st.info(
    f"**研究目的：** {selected_guide['主要目的']}  \n"
    f"**适用数据类型：** {selected_guide['适用数据类型']}  \n"
    f"**典型问题：** {selected_guide['典型问题']}"
)

# 通用样式参数
st.sidebar.header("三、图表样式设置")
figure_height = st.sidebar.slider("图形高度", 400, 900, 550, 50)
show_legend = st.sidebar.checkbox("显示图例", value=True)
template_name = st.sidebar.selectbox(
    "选择主题样式",
    ["plotly", "plotly_white", "plotly_dark", "ggplot2", "simple_white"]
)

fig = None

if chart_type == "折线图":
    if not numeric_cols:
        st.warning("折线图至少需要一个数值变量。")
    else:
        st.sidebar.subheader("折线图参数")
        x_candidates = datetime_cols + all_cols
        x_col = st.sidebar.selectbox("选择 X 轴", x_candidates, key="line_x")
        y_col = st.sidebar.selectbox("选择 Y 轴", numeric_cols, key="line_y")
        line_mode = st.sidebar.selectbox("线型模式", ["lines", "lines+markers"], key="line_mode")
        line_dash = st.sidebar.selectbox("线条样式", ["solid", "dot", "dash", "longdash", "dashdot"], key="line_dash")

        if x_col in category_cols:
            agg_func = st.sidebar.selectbox("统计方式", ["mean", "median", "sum", "count"], key="line_agg")
            if agg_func == "count":
                plot_df = df.groupby(x_col)[y_col].count().reset_index()
            else:
                plot_df = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
        else:
            plot_df = df[[x_col, y_col]].dropna()

        plot_df = safe_sort_dataframe(plot_df, x_col)

        fig = px.line(
            plot_df,
            x=x_col,
            y=y_col,
            markers=("markers" in line_mode),
            template=template_name,
            title=f"{chart_type}：{y_col} 随 {x_col} 的变化"
        )
        fig.update_traces(line=dict(dash=line_dash))
        fig.update_layout(height=figure_height, showlegend=show_legend)

        st.subheader("统计参量")
        st.dataframe(describe_numeric(df[y_col]), use_container_width=True)

elif chart_type == "柱状图":
    if not numeric_cols:
        st.warning("柱状图建议至少有一个数值变量。")
    else:
        st.sidebar.subheader("柱状图参数")
        x_col = st.sidebar.selectbox("选择分类变量", category_cols if category_cols else all_cols, key="bar_x")
        y_col = st.sidebar.selectbox("选择数值变量", numeric_cols, key="bar_y")
        agg_func = st.sidebar.selectbox("统计方式", ["mean", "median", "sum", "count"], key="bar_agg")

        if agg_func == "count":
            plot_df = df.groupby(x_col)[y_col].count().reset_index()
        else:
            plot_df = df.groupby(x_col)[y_col].agg(agg_func).reset_index()

        fig = px.bar(
            plot_df,
            x=x_col,
            y=y_col,
            text=y_col,
            template=template_name,
            title=f"{chart_type}：不同 {x_col} 的 {y_col}"
        )
        fig.update_layout(height=figure_height, showlegend=show_legend)

        st.subheader("分组统计结果")
        st.dataframe(plot_df, use_container_width=True)

elif chart_type == "饼图":
    st.sidebar.subheader("饼图参数")
    pie_col = st.sidebar.selectbox("选择分类变量", category_cols if category_cols else all_cols, key="pie_col")
    hole_size = st.sidebar.slider("中心空洞大小（0=饼图，>0=环形图）", 0.0, 0.8, 0.0, 0.1)

    plot_df = df[pie_col].value_counts(dropna=False).reset_index()
    plot_df.columns = [pie_col, "频数"]

    fig = px.pie(
        plot_df,
        names=pie_col,
        values="频数",
        hole=hole_size,
        template=template_name,
        title=f"{chart_type}：{pie_col} 的占比结构"
    )
    fig.update_layout(height=figure_height, showlegend=show_legend)

    st.subheader("类别占比统计")
    st.dataframe(describe_category(df[pie_col]), use_container_width=True)

elif chart_type == "散点图":
    if len(numeric_cols) < 2:
        st.warning("散点图至少需要两个数值变量。")
    else:
        st.sidebar.subheader("散点图参数")
        x_col = st.sidebar.selectbox("选择 X 轴", numeric_cols, key="scatter_x")
        y_col = st.sidebar.selectbox(
            "选择 Y 轴",
            [col for col in numeric_cols if col != x_col] if len(numeric_cols) > 1 else numeric_cols,
            key="scatter_y"
        )
        color_col = st.sidebar.selectbox("选择分类变量（颜色）", ["无"] + category_cols, key="scatter_color")
        size_col = st.sidebar.selectbox("选择数值变量（点大小，可选）", ["无"] + numeric_cols, key="scatter_size")
        opacity = st.sidebar.slider("透明度", 0.2, 1.0, 0.8, 0.1)
        add_trendline = st.sidebar.checkbox("添加趋势线", value=False)

        plot_kwargs = {
            "data_frame": df,
            "x": x_col,
            "y": y_col,
            "opacity": opacity,
            "template": template_name,
            "title": f"{chart_type}：{x_col} 与 {y_col} 的关系"
        }

        if color_col != "无":
            plot_kwargs["color"] = color_col
        if size_col != "无":
            plot_kwargs["size"] = size_col
        if add_trendline:
            plot_kwargs["trendline"] = "ols"

        fig = px.scatter(**plot_kwargs)
        fig.update_layout(height=figure_height, showlegend=show_legend)

        st.subheader("双变量统计参量")
        corr_pearson = df[[x_col, y_col]].dropna().corr(method="pearson").iloc[0, 1]
        corr_spearman = df[[x_col, y_col]].dropna().corr(method="spearman").iloc[0, 1]
        cov_value = df[[x_col, y_col]].dropna().cov().iloc[0, 1]

        pair_stats = pd.DataFrame({
            "统计量": ["Pearson相关系数", "Spearman相关系数", "协方差"],
            "值": [corr_pearson, corr_spearman, cov_value]
        })
        st.dataframe(pair_stats, use_container_width=True)

elif chart_type == "直方图":
    if not numeric_cols:
        st.warning("直方图需要数值变量。")
    else:
        st.sidebar.subheader("直方图参数")
        hist_col = st.sidebar.selectbox("选择数值变量", numeric_cols, key="hist_col")
        bins = st.sidebar.slider("分箱数量（bins）", 5, 60, 20, 1)
        color_col = st.sidebar.selectbox("选择分类变量（可选）", ["无"] + category_cols, key="hist_color")
        histnorm = st.sidebar.selectbox("纵轴统计方式", ["", "percent", "probability", "density"], key="hist_norm")
        marginal_type = st.sidebar.selectbox("边缘图类型", ["box", "violin", "rug", None], key="hist_marginal")

        if color_col == "无":
            fig = px.histogram(
                df,
                x=hist_col,
                nbins=bins,
                histnorm=histnorm if histnorm != "" else None,
                marginal=marginal_type,
                template=template_name,
                title=f"{chart_type}：{hist_col} 的分布"
            )
        else:
            fig = px.histogram(
                df,
                x=hist_col,
                color=color_col,
                nbins=bins,
                histnorm=histnorm if histnorm != "" else None,
                marginal=marginal_type,
                barmode="overlay",
                opacity=0.7,
                template=template_name,
                title=f"{chart_type}：{hist_col} 的分布"
            )

        fig.update_layout(height=figure_height, showlegend=show_legend)

        st.subheader("单变量统计参量")
        st.dataframe(describe_numeric(df[hist_col]), use_container_width=True)

elif chart_type == "箱线图":
    if not numeric_cols:
        st.warning("箱线图需要数值变量。")
    else:
        st.sidebar.subheader("箱线图参数")
        y_col = st.sidebar.selectbox("选择数值变量", numeric_cols, key="box_y")
        x_col = st.sidebar.selectbox("选择分类变量（可选）", ["无"] + category_cols, key="box_x")
        color_col = st.sidebar.selectbox("选择着色变量（可选）", ["无"] + category_cols, key="box_color")
        points_option = st.sidebar.selectbox("是否显示散点", ["outliers", "all", False], key="box_points")
        orientation = st.sidebar.selectbox("方向", ["纵向", "横向"], key="box_orientation")

        plot_kwargs = {
            "data_frame": df,
            "template": template_name,
            "points": points_option,
            "title": f"{chart_type}：{y_col} 的分布"
        }

        if orientation == "纵向":
            plot_kwargs["y"] = y_col
            if x_col != "无":
                plot_kwargs["x"] = x_col
            if color_col != "无":
                plot_kwargs["color"] = color_col
        else:
            plot_kwargs["x"] = y_col
            if x_col != "无":
                plot_kwargs["y"] = x_col
            if color_col != "无":
                plot_kwargs["color"] = color_col
            plot_kwargs["title"] = f"{chart_type}：{y_col} 的分布（横向）"

        fig = px.box(**plot_kwargs)
        fig.update_layout(height=figure_height, showlegend=show_legend)

        st.subheader("单变量统计参量")
        st.dataframe(describe_numeric(df[y_col]), use_container_width=True)

        if x_col != "无":
            st.subheader("分组统计参量")
            grouped = df.groupby(x_col)[y_col].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
            st.dataframe(grouped, use_container_width=True)

# =========================
# 图形显示
# =========================
if fig is not None:
    st.subheader("图表展示")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 图表说明
# =========================
st.header("7. 当前图形的教学解释")

explanation_text = {
    "折线图": """
- 适合观察变量随时间、顺序或连续变量的变化趋势。  
- 若横轴为分类变量，通常展示各类别统计量的变化。  
- 常见于时间序列、实验过程和阶段性变化分析。  
""",
    "柱状图": """
- 适合比较不同类别之间的数量或统计量差异。  
- 可用于均值、中位数、频数、总量等比较。  
- 常见于分类数据展示与组间对比。  
""",
    "饼图": """
- 适合展示整体内部构成比例。  
- 当类别较少时效果较好，类别过多时可读性下降。  
- 常见于占比分析和结构分析。  
""",
    "散点图": """
- 适合分析两个数值变量之间的关系。  
- 可通过颜色表示类别，通过点大小表示第三变量。  
- 有助于识别聚类、趋势和异常点。  
""",
    "直方图": """
- 适合观察单变量的分布形态。  
- 能帮助判断偏态、集中趋势与离散程度。  
- bins 参数变化会影响图形细节。  
""",
    "箱线图": """
- 适合展示中位数、四分位范围及异常值。  
- 可用于比较不同类别之间分布差异。  
- 特别适合发现离群值与数据波动范围。  
"""
}

st.markdown(explanation_text.get(chart_type, "当前图表用于数据分析与可视化展示。"))

# =========================
# 教学总结
# =========================
st.header("8. 教学提示")
st.markdown("""
1. **先识别数据类型，再选择图表类型。**  
2. **先明确研究目的，再决定参数设置。**  
3. **图形分析应与统计参量结合，不能只看图不看数。**  
4. **同一种图表适用于不同研究目的时，关键在于变量选择和统计方式设定。**  
5. **上传本地数据后，优先查看变量说明，再决定采用哪种可视化方法。**
""")