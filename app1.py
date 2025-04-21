# -*- coding: utf-8 -*- # 指定编码为 UTF-8
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 页面基础配置 ---
st.set_page_config(
    page_title="盐城二手房智能分析器",  # 页面标题
    page_icon="🏠",                  # 页面图标
    layout="wide",                   # 页面布局：宽屏
    initial_sidebar_state="auto" # 侧边栏初始状态：展开
)

# --- 常量定义：模型和资源文件路径 ---
# 获取脚本所在的目录
try:
    # 当作为脚本运行时，这会生效
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 在 __file__ 未定义的环境（如 Streamlit sharing）中的回退方案
    CURRENT_DIR = os.getcwd()

# 定义各个资源文件的完整路径
MARKET_MODEL_PATH = os.path.join(CURRENT_DIR, 'market_segment_lgbm_model.joblib')   # 市场细分模型
PRICE_LEVEL_MODEL_PATH = os.path.join(CURRENT_DIR, 'price_level_rf_model.joblib')  # 价格水平模型
REGRESSION_MODEL_PATH = os.path.join(CURRENT_DIR, 'unit_price_rf_model.joblib')    # 均价预测（回归）模型
SCALER_PATH = os.path.join(CURRENT_DIR, 'regression_scaler.joblib')             # 回归模型使用的缩放器
FEATURE_NAMES_PATH = os.path.join(CURRENT_DIR, 'feature_names.joblib')           # 包含各模型所需特征名称的文件
MAPPINGS_PATH = os.path.join(CURRENT_DIR, 'mappings.joblib')                   # 包含分类特征编码映射的文件

# --- ***** 新增：定义均价预测模型所需的固定特征列表 ***** ---
# ***** 注意：这里的特征列表必须与训练回归模型和Scaler时使用的特征完全一致，包括顺序 *****
REQUIRED_REGRESSION_FEATURES = ['所属区域', '房龄', '面积(㎡)', '楼层数', '建造时间', '室', '厅', '卫']
print(f"代码指定均价预测特征: {REQUIRED_REGRESSION_FEATURES}") # 打印日志：代码中指定的均价预测特征

# --- 加载资源函数 (使用缓存) ---
@st.cache_resource # 使用 Streamlit 缓存装饰器，避免重复加载资源
def load_resources():
    """加载所有必要的资源文件 (模型, scaler, 特征名, 映射关系)。"""
    resources = {} # 初始化用于存储加载资源的字典
    all_files_exist = True # 标志位：所有必需文件是否存在
    # 定义所有必需的文件路径列表
    required_files = [MARKET_MODEL_PATH, PRICE_LEVEL_MODEL_PATH, REGRESSION_MODEL_PATH,
                      SCALER_PATH, FEATURE_NAMES_PATH, MAPPINGS_PATH]
    missing_files = [] # 用于存储缺失文件名的列表
    # 检查每个必需文件是否存在
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 未找到。") # 打印错误日志
            missing_files.append(os.path.basename(file_path)) # 将缺失的文件名添加到列表
            all_files_exist = False # 更新标志位
    # 如果有文件缺失，打印错误信息并返回
    if not all_files_exist:
        print(f"错误：缺少文件 {missing_files}。请确保所有 .joblib 文件与 app.py 在同一目录。")
        return None, missing_files # 返回 None 和缺失文件列表

    # 尝试加载所有资源文件
    try:
        resources['market_model'] = joblib.load(MARKET_MODEL_PATH)          # 加载市场细分模型
        resources['price_level_model'] = joblib.load(PRICE_LEVEL_MODEL_PATH) # 加载价格水平模型
        resources['regression_model'] = joblib.load(REGRESSION_MODEL_PATH)   # 加载均价预测模型
        resources['scaler'] = joblib.load(SCALER_PATH)                     # 加载缩放器
        resources['feature_names'] = joblib.load(FEATURE_NAMES_PATH)       # 加载特征名称
        resources['mappings'] = joblib.load(MAPPINGS_PATH)                 # 加载映射关系
        print("所有资源加载成功。") # 打印成功日志

        # --- 验证 feature_names.joblib 中的回归特征 ---
        # 从加载的特征名称中获取 'regression' 部分
        loaded_reg_features = resources.get('feature_names', {}).get('regression')
        if loaded_reg_features:
            print(f"从 {os.path.basename(FEATURE_NAMES_PATH)} 加载的 'regression' 特征: {loaded_reg_features}")
            # 比较加载的特征与代码中定义的特征是否一致
            if set(loaded_reg_features) != set(REQUIRED_REGRESSION_FEATURES):
                 print(f"警告: 从 {os.path.basename(FEATURE_NAMES_PATH)} 加载的 'regression' 特征与代码中指定的 ({REQUIRED_REGRESSION_FEATURES}) 不完全匹配。将优先使用代码中指定的列表。")
                 # ***** 关键：检查 Scaler 是否与代码指定的特征匹配 *****
                 # 检查加载的 scaler 是否有 n_features_in_ 属性，并且其值与代码中定义的特征数量是否匹配
                 if hasattr(resources['scaler'], 'n_features_in_') and resources['scaler'].n_features_in_ != len(REQUIRED_REGRESSION_FEATURES):
                      error_msg = f"严重错误: Scaler (来自 {os.path.basename(SCALER_PATH)}) 期望 {resources['scaler'].n_features_in_} 个特征, 但代码指定了 {len(REQUIRED_REGRESSION_FEATURES)} 个回归特征 ({REQUIRED_REGRESSION_FEATURES})。请确保 Scaler 与指定的特征列表一致。"
                      print(error_msg) # 打印严重错误日志
                      # 返回 None 模拟加载失败，因为存在不匹配
                      return None, [error_msg]
                 else:
                    # 如果特征不匹配但 Scaler 匹配（或无法检查Scaler），则仅打印警告
                    print(f"从 {os.path.basename(FEATURE_NAMES_PATH)} 加载的 'regression' 特征与代码指定一致。") # 修正日志信息
            else:
                 # 如果特征列表完全一致
                 print(f"从 {os.path.basename(FEATURE_NAMES_PATH)} 加载的 'regression' 特征与代码指定一致。")
        else:
            # 如果 feature_names.joblib 中没有 'regression' 键
            print(f"警告: 在 {os.path.basename(FEATURE_NAMES_PATH)} 中未找到 'regression' 特征列表。将使用代码中指定的列表: {REQUIRED_REGRESSION_FEATURES}")
             # ***** 关键：同样检查 Scaler *****
             # 即使没有从文件加载特征列表，也要检查 Scaler 是否与代码中定义的特征列表匹配
            if hasattr(resources['scaler'], 'n_features_in_') and resources['scaler'].n_features_in_ != len(REQUIRED_REGRESSION_FEATURES):
                error_msg = f"严重错误: Scaler (来自 {os.path.basename(SCALER_PATH)}) 期望 {resources['scaler'].n_features_in_} 个特征, 但代码指定了 {len(REQUIRED_REGRESSION_FEATURES)} 个回归特征 ({REQUIRED_REGRESSION_FEATURES})。请确保 Scaler 与指定的特征列表一致。"
                print(error_msg) # 打印严重错误日志
                return None, [error_msg] # 返回 None 模拟加载失败

        return resources, None # 成功加载所有资源，返回资源字典和 None
    except Exception as e:
        # 捕获加载过程中可能出现的任何其他异常
        print(f"加载资源时发生错误: {e}") # 打印错误日志
        return None, [f"加载错误: {e}"] # 返回 None 和错误信息列表

# 调用加载函数
resources, load_error_info = load_resources()

# --- 辅助函数 ---
def format_mapping_options_for_selectbox(name_to_code_mapping):
    """为 Streamlit Selectbox 准备选项和格式化函数所需的数据, 增加 '无' 选项。

    Args:
        name_to_code_mapping (dict): 从 mapping 文件加载的原始名称到编码的字典。

    Returns:
        dict: 一个新的字典，键是编码 (或 None)，值是用于在下拉框中显示的字符串。
              包含一个 {None: "无 (不适用)"} 的条目。
    """
    # 检查输入是否为字典类型
    if not isinstance(name_to_code_mapping, dict):
        print(f"[格式化错误] 输入非字典: {type(name_to_code_mapping)}")
        return {} # 出错时返回空字典

    # 创建包含 "无" 选项的新字典
    code_to_display_string = {None: "无 (不适用)"} # 首先添加 'None' 选项

    try:
        # 在添加其他选项前，对原始映射项进行排序
        try:
            # 尝试按整数编码排序
            sorted_items = sorted(name_to_code_mapping.items(), key=lambda item: int(item[1]))
        except ValueError:
             # 如果无法将所有编码转换为整数，则按字符串编码排序
             # print(f"[格式化警告] 无法将所有 code 转换为 int 进行排序，将按字符串排序: {name_to_code_mapping}")
             sorted_items = sorted(name_to_code_mapping.items(), key=lambda item: str(item[1]))

        # 遍历排序后的项，添加到新字典中
        for name, code in sorted_items:
            try:
                # Selectbox 的选项通常需要基本类型（int 或 str）
                code_key = int(code)
            except ValueError:
                # 如果不能转换为整数，则保持为字符串
                code_key = str(code)

            name_str = str(name) # 确保名称是字符串
            # 在下拉框中只显示名称
            code_to_display_string[code_key] = f"{name_str}"

        return code_to_display_string # 返回格式化后的字典

    except (TypeError, KeyError, Exception) as e: # 捕获处理过程中可能发生的更广泛的错误
        print(f"[格式化错误] 转换/排序映射时出错 ({name_to_code_mapping}): {e}")
        # 如果排序/转换失败，回退：只返回包含 'None' 选项的字典
        return {None: "无 (不适用)"}


# --- Streamlit 用户界面主要部分 ---
st.title("🏠 盐城二手房智能分析与预测") # 设置页面主标题
# 使用 Markdown 添加介绍性文本
st.markdown("""
欢迎使用盐城二手房分析工具！请在左侧边栏输入房产特征，我们将为您提供三个维度的预测：
1.  **市场细分预测**: 判断房产属于低端、中端还是高端市场。
2.  **价格水平预测**: 判断房产单价是否高于其所在区域的平均水平。
3.  **房产均价预测**: 预测房产的每平方米单价（元/㎡）。
""")
st.markdown("---") # 添加分割线

# --- 应用启动时资源加载失败或映射缺失的处理 ---
if not resources:
     # 如果资源加载失败 (resources 为 None)
     st.error("❌ **应用程序初始化失败！**") # 显示错误消息
     if load_error_info:
         # 如果有具体的加载错误信息
         st.warning(f"无法加载必要的资源文件。错误详情:")
         for error in load_error_info:
             st.markdown(f"*   `{error}`") # 逐条显示错误信息
             # 为 Scaler 不匹配错误提供具体指导
             if "Scaler" in error and "特征" in error:
                 st.info(f"💡 **提示:** 上述错误表明用于均价预测的特征缩放器 (`{os.path.basename(SCALER_PATH)}`) 与代码中指定的特征列表 (`{REQUIRED_REGRESSION_FEATURES}`) 不匹配。您需要：\n    1. 确认代码中的 `REQUIRED_REGRESSION_FEATURES` 列表是正确的。\n    2. 使用 **完全相同** 的特征和 **顺序** 重新训练并保存 `regression_scaler.joblib` 文件。")
     else:
         # 如果没有具体的错误信息，但资源加载失败
         st.warning("无法找到一个或多个必需的资源文件。")
     # 提供检查步骤
     st.markdown(f"""
        请检查以下几点：
        *   确认以下所有 `.joblib` 文件都与 `app.py` 文件在 **同一个** 目录下:
            *   `{os.path.basename(MARKET_MODEL_PATH)}`
            *   `{os.path.basename(PRICE_LEVEL_MODEL_PATH)}`
            *   `{os.path.basename(REGRESSION_MODEL_PATH)}`
            *   `{os.path.basename(SCALER_PATH)}`
            *   `{os.path.basename(FEATURE_NAMES_PATH)}`
            *   `{os.path.basename(MAPPINGS_PATH)}`
        *   确保 `{os.path.basename(MAPPINGS_PATH)}` 和 `{os.path.basename(FEATURE_NAMES_PATH)}` 文件内容有效。
        *   检查运行 Streamlit 的终端是否有更详细的错误信息。
     """)
     st.stop() # 停止应用执行

# --- 如果资源加载成功 ---
# 从加载的 resources 字典中获取各个组件
mappings = resources['mappings']                   # 获取映射关系
feature_names_loaded = resources.get('feature_names', {}) # 安全地获取特征名称字典
market_model = resources['market_model']         # 获取市场细分模型
price_level_model = resources['price_level_model']   # 获取价格水平模型
regression_model = resources['regression_model']   # 获取均价预测模型
scaler = resources['scaler']                       # 获取缩放器

# 检查核心映射和特征列表是否存在且为预期类型
required_mappings = ['方位', '楼层', '所属区域', '房龄', '市场类别', '是否高于区域均价'] # 需要检查的映射键
required_features_in_file = ['market', 'price_level'] # 需要在特征文件中检查的键（回归特征单独处理）
valid_resources = True # 标志位：资源文件内容是否有效
missing_or_invalid = [] # 存储缺失或无效项的列表

# 检查必需的映射是否存在且为字典类型
for key in required_mappings:
    if key not in mappings or not isinstance(mappings.get(key), dict):
        missing_or_invalid.append(f"映射 '{key}' (来自 {os.path.basename(MAPPINGS_PATH)})")
        valid_resources = False

# 检查必需的特征列表是否存在且为列表类型
for key in required_features_in_file:
    if key not in feature_names_loaded or not isinstance(feature_names_loaded.get(key), list):
        missing_or_invalid.append(f"特征列表 '{key}' (来自 {os.path.basename(FEATURE_NAMES_PATH)})")
        valid_resources = False

# 检查特征文件中的 'regression' 键是否存在及类型（即使我们稍后会覆盖它）
if 'regression' not in feature_names_loaded:
     # 如果文件中没有 'regression' 键，打印信息
     print(f"信息: 'regression' 键未在 {os.path.basename(FEATURE_NAMES_PATH)} 中找到。将使用代码中定义的特征列表。")
elif not isinstance(feature_names_loaded.get('regression'), list):
     # 如果 'regression' 键存在但不是列表，记录为无效
     missing_or_invalid.append(f"特征列表 'regression' (来自 {os.path.basename(FEATURE_NAMES_PATH)}) 格式无效 (应为列表)")
     valid_resources = False

# 如果有任何资源无效，显示错误并停止
if not valid_resources:
    st.error(f"❌ 资源文件内容不完整或格式错误。缺少或无效的项目:")
    for item in missing_or_invalid:
        st.markdown(f"*   {item}") # 逐条显示无效项
    st.stop() # 停止应用执行

# --- 侧边栏输入控件 ---
st.sidebar.header("🏘️ 房产特征输入") # 侧边栏标题

# --- ***** 修改：字典，将内部特征名映射到用户界面标签 ***** ---
# 定义一个字典，用于将内部使用的特征名称映射到用户界面上显示的标签
feature_to_label = {
    # 选择项特征的标签
    '方位': "房屋方位:",
    '楼层': "楼层位置:",
    '所属区域': "所属区域:",
    '房龄': "房龄:",
    # 数值项特征的标签
    '总价(万)': "总价 (万):",
    '面积(㎡)': "面积 (㎡):",
    '建造时间': "建造时间 (年份):",
    '楼层数': "总楼层数:",
    '室': "室:",
    '厅': "厅:",
    '卫': "卫:"
}

# 初始化用于存储用户输入的字典
selectbox_inputs = {}     # 存储下拉框选择的值 (编码或 None)
selectbox_labels_map = {} # 用于将内部键映射回显示标签 (如果需要)
all_select_valid = True  # 标志位：所有下拉框是否都成功加载了选项

st.sidebar.subheader("选择项特征") # 侧边栏子标题
# 封装创建下拉框控件的逻辑
def create_selectbox(internal_key, help_text, key_suffix):
    """创建一个 Streamlit 下拉选择框。

    Args:
        internal_key (str): 在 mappings 和 feature_to_label 中使用的内部特征键。
        help_text (str): 显示在控件旁边的帮助提示文本。
        key_suffix (str): 用于构造 Streamlit 控件唯一 key 的后缀。

    Returns:
        int or str or None: 用户选择的值（通常是编码），如果选择了 "无 (不适用)" 则返回 None。
                            如果加载出错，也可能返回 None。
    """
    global all_select_valid # 允许修改全局标志位
    label = feature_to_label.get(internal_key, internal_key) # 从映射字典获取显示标签
    try:
        options_map = mappings[internal_key] # 获取该特征的名称到编码的映射
        # 生成包含 "无" 选项的显示字典
        display_map = format_mapping_options_for_selectbox(options_map)

        # 检查是否有有效的选项（除了 "无"）
        if not display_map or len(display_map) <= 1:
             st.sidebar.warning(f"'{label}' 缺少有效选项 (除了'无')。请检查 {os.path.basename(MAPPINGS_PATH)} 中的 '{internal_key}'。")
             if not display_map:
                 # 如果连 display_map 都是空的，提供一个回退
                 display_map = {None: "无 (加载失败)"}

        options_codes = list(display_map.keys()) # 获取所有选项的键（编码和 None）

        # 确定默认选项的索引 - 尽量避免将 "无" 作为默认值
        default_index = 0 # 默认索引为 0 ("无")
        if len(options_codes) > 1: # 如果至少有一个有效选项
            # 为特定特征设置更合理的默认值（示例）
            common_defaults = {'楼层': 1, '房龄': 2} # 假设 1 代表中间楼层, 2 代表 6-10年
            target_default_code = common_defaults.get(internal_key) # 获取该特征的目标默认编码

            # 如果目标默认编码存在于选项中
            if target_default_code is not None and target_default_code in options_codes:
                try:
                    # 找到该编码的索引
                    default_index = options_codes.index(target_default_code)
                except ValueError:
                    # 如果找不到（理论上不应发生，但做保护），打印警告并使用第一个有效选项
                    print(f"Warning: Default code {target_default_code} for {internal_key} not found in options {options_codes}. Defaulting.")
                    default_index = 1 # 默认使用第一个非 'None' 选项
            # 如果没有特定默认值，根据选项数量选择一个默认值
            elif len(options_codes) > 3: # 选项较多时，选择中间附近的选项
                 default_index = (len(options_codes) -1) // 2 + 1 # 'None'之后的中间索引
            elif len(options_codes) >= 2: # 只有一个有效选项时
                default_index = 1 # 选择那个有效选项

        # 创建 Streamlit selectbox 控件
        selected_value = st.sidebar.selectbox(
            label,                           # 控件标签
            options=options_codes,           # 选项列表（编码和 None）
            index=default_index,             # 默认选中项的索引
            format_func=lambda x: display_map.get(x, f"未知 ({x})"), # 格式化函数，显示名称而非编码
            key=f"{key_suffix}_select",      # 控件的唯一 key
            help=help_text                   # 帮助提示
        )
        selectbox_labels_map[internal_key] = label # 存储内部键到标签的映射
        return selected_value # 返回用户选择的值
    except Exception as e:
        # 捕获创建 selectbox 过程中的任何错误
        st.sidebar.error(f"加载 '{label}' 选项时出错: {e}")
        print(f"Error details for loading {label}: {e}") # 在控制台打印详细错误
        all_select_valid = False # 设置标志位表明有下拉框加载失败
        return None # 返回 None 表示出错

# 使用封装的函数创建各个下拉框控件
selectbox_inputs['方位'] = create_selectbox('方位', "选择房屋的主要朝向。选择 '无' 如果不确定或不适用。", "orientation")
selectbox_inputs['楼层'] = create_selectbox('楼层', "选择房屋所在楼层的大致位置。选择 '无' 如果不确定或不适用。", "floor_level")
selectbox_inputs['所属区域'] = create_selectbox('所属区域', "选择房产所在的行政区域或板块。选择 '无' 如果不确定或不适用。", "district")
selectbox_inputs['房龄'] = create_selectbox('房龄', "选择房屋的建造年限范围。选择 '无' 如果不确定或不适用。", "age")

# --- ***** 修改：数值输入控件，增加 "无" 选项 ***** ---
st.sidebar.subheader("数值项特征") # 侧边栏子标题
numeric_inputs = {}         # 存储最终的数值输入值 (可能是数值或 None)
numeric_input_states = {}   # 存储每个数值特征是选择 "输入数值" 还是 "无 (不适用)"

# 定义数值输入的默认值 (仅在选择 "输入数值" 时使用)
default_numeric_values = {
    '总价(万)': 120.0,
    '面积(㎡)': 95.0,
    '建造时间': 2015,
    '楼层数': 18,
    '室': 3,
    '厅': 2,
    '卫': 1
}

# 定义数值输入的参数 (最小值、最大值、步长、格式、帮助文本)
numeric_params = {
    '总价(万)': {'min_value': 0.0, 'max_value': 10000.0, 'step': 5.0, 'format': "%.1f", 'help': "输入房产的总价，单位万元。选择 '无' 表示不适用。"},
    '面积(㎡)': {'min_value': 1.0, 'max_value': 2000.0, 'step': 1.0, 'format': "%.1f", 'help': "输入房产的建筑面积，单位平方米。选择 '无' 表示不适用。"},
    '建造时间': {'min_value': 1900, 'max_value': 2025, 'step': 1, 'format': "%d", 'help': "输入房屋的建造年份。选择 '无' 表示不适用。"},
    '楼层数': {'min_value': 1, 'max_value': 100, 'step': 1, 'format': "%d", 'help': "输入楼栋的总楼层数。选择 '无' 表示不适用。"},
    '室': {'min_value': 0, 'max_value': 20, 'step': 1, 'format': "%d", 'help': "输入卧室数量。选择 '无' 表示不适用。"},
    '厅': {'min_value': 0, 'max_value': 10, 'step': 1, 'format': "%d", 'help': "输入客厅/餐厅数量。选择 '无' 表示不适用。"},
    '卫': {'min_value': 0, 'max_value': 10, 'step': 1, 'format': "%d", 'help': "输入卫生间数量。选择 '无' 表示不适用。"}
}

# 遍历 feature_to_label 字典，为数值特征创建组合输入控件
for key, label in feature_to_label.items():
    if key in numeric_params: # 检查当前项是否是需要处理的数值特征
        param = numeric_params[key]         # 获取该特征的参数
        default_val = default_numeric_values[key] # 获取默认值
        # 为控件 key 创建一个简单的后缀 (移除特殊字符)
        key_suffix = key.replace('(','').replace(')','').replace('㎡','')

        # 创建一个下拉框，让用户选择是输入数值还是选择 "无"
        numeric_input_states[key] = st.sidebar.selectbox(
            label,                             # 使用 feature_to_label 定义的标签
            options=["输入数值", "无 (不适用)"], # 选项
            index=0,                           # 默认选择 "输入数值"
            key=f"{key_suffix}_selector",      # 控件唯一 key
            help=param['help']                 # 帮助提示
        )

        # 根据用户的选择，决定是否显示数值输入框
        if numeric_input_states[key] == "输入数值":
            # 如果用户选择 "输入数值"，则显示 st.number_input 控件
            numeric_inputs[key] = st.sidebar.number_input(
                f"输入 {label}", # 输入框前的提示性文本（可选）
                min_value=param['min_value'],    # 最小值
                max_value=param['max_value'],    # 最大值
                value=default_val,               # 默认值
                step=param['step'],              # 步长
                format=param['format'],          # 显示格式
                key=f"{key_suffix}_value",       # 控件唯一 key
                label_visibility="collapsed"     # 隐藏 number_input 的标签，因为前面 selectbox 已有标签
            )
        else:
            # 如果用户选择 "无 (不适用)"，则将该特征的值设为 None
            numeric_inputs[key] = None
            # 可选：可以显示一个禁用的占位符文本框，或者什么都不显示
            # st.sidebar.text_input(f"{label}", value="不适用", disabled=True, key=f"{key_suffix}_value_disabled")


# --- 预测触发按钮 ---
st.sidebar.markdown("---") # 侧边栏分割线
# 根据是否有下拉框加载失败来决定按钮是否可用
predict_button_disabled = not all_select_valid
# 设置按钮的帮助文本
predict_button_help = "点击这里根据输入的特征进行预测分析" if all_select_valid else "部分下拉框选项加载失败，无法进行预测。请检查资源文件或错误信息。"

# 创建预测按钮
if st.sidebar.button("🚀 开始分析预测", type="primary", use_container_width=True, help=predict_button_help, disabled=predict_button_disabled):

    # --- ***** 修改：整合输入时处理 None 值 ***** ---
    # 将下拉框的输入和数值输入整合到一个字典中
    all_inputs = {**selectbox_inputs} # 首先复制下拉框的输入
    # 遍历数值输入的状态
    for key, state in numeric_input_states.items():
        if state == "无 (不适用)":
            # 如果用户选择了 "无"，将对应的值设为 None
            all_inputs[key] = None
        else:
            # 如果用户选择了 "输入数值"，从 Streamlit 的 session_state 中获取 number_input 的值
            key_suffix = key.replace('(','').replace(')','').replace('㎡','')
            # 使用与 number_input 控件相同的 key 来获取其当前值
            all_inputs[key] = st.session_state[f"{key_suffix}_value"]

    print("Combined inputs for prediction:", all_inputs) # 打印整合后的输入数据，用于调试

    # --- 初始化结果变量 ---
    market_pred_label = "等待计算..."      # 市场细分预测结果标签
    price_level_pred_label = "等待计算..." # 价格水平预测结果标签
    price_level_pred_code = -99          # 价格水平预测结果编码 (-99 表示未计算或错误)
    unit_price_pred = -1.0               # 均价预测结果 (-1.0 表示未计算或错误)
    error_messages = []                  # 存储运行时错误信息
    # 存储每个模型是否因数据不足而无法预测的标志
    insufficient_data_flags = {'market': False, 'price_level': False, 'regression': False}

    # --- ***** 修改：Helper Function to Check Input Sufficiency (Handles None) ***** ---
    def check_sufficiency(model_key, required_feature_list):
        """检查特定模型所需的所有特征是否都有有效的 (非 None) 输入值。

        Args:
            model_key (str): 模型的标识符 (如 'market', 'price_level', 'regression')。
            required_feature_list (list): 该模型必需的特征名称列表。

        Returns:
            bool: 如果所有必需特征都有非 None 值，则返回 True，否则返回 False。
                  同时会更新全局的 insufficient_data_flags。
        """
        missing_for_model = [] # 存储当前模型缺失的特征名称（用户界面标签）
        # 遍历模型所需的每个特征
        for feat in required_feature_list:
            # 检查特征是否存在于整合后的输入字典中
            if feat not in all_inputs:
                 # 这是一个严重的配置错误：模型需要的特征在 UI 上根本没有定义
                 print(f"严重警告: 模型 '{model_key}' 需要的特征 '{feat}' 在UI输入中未定义!")
                 missing_for_model.append(f"{feature_to_label.get(feat, feat)} (UI未定义)")
            # 检查特征的值是否为 None
            elif all_inputs[feat] is None:
                # 如果值为 None (用户选择了 "无" 或下拉框加载失败等)
                missing_for_model.append(feature_to_label.get(feat, feat)) # 使用显示标签记录缺失项

        # 如果有任何缺失的特征
        if missing_for_model:
            print(f"模型 '{model_key}' 数据不足，缺少: {missing_for_model}") # 打印日志
            insufficient_data_flags[model_key] = True # 设置对应模型的不足标志
            return False # 返回 False 表示数据不足
        return True # 所有必需特征都有值，返回 True

    # --- 1. 市场细分预测 ---
    # 获取市场细分模型所需的特征列表
    market_features_needed = feature_names_loaded.get('market', [])
    if not market_features_needed:
         # 如果特征列表文件没有 'market' 键
         st.warning("警告: 未在 feature_names.joblib 中找到 'market' 模型的特征列表，无法进行市场细分预测。")
         insufficient_data_flags['market'] = True # 标记为数据不足（配置缺失）
         market_pred_label = "配置缺失"         # 设置特定状态标签
    elif check_sufficiency('market', market_features_needed):
        # 如果特征列表存在且数据充足
        try:
            # 准备模型输入数据 (只包含需要的特征)
            input_data_market = {feat: all_inputs[feat] for feat in market_features_needed}
            # 创建 DataFrame 并确保列顺序与训练时一致
            input_df_market = pd.DataFrame([input_data_market])[market_features_needed]
            # 使用模型进行预测
            market_pred_code = market_model.predict(input_df_market)[0]
            # 获取市场类别的编码到名称的映射
            market_output_map_raw = mappings.get('市场类别', {})
            # 将预测编码转换为正确的类型 (通常是 int) 以便在映射中查找
            market_pred_key = int(market_pred_code) if isinstance(market_pred_code, (int, np.integer, float)) else str(market_pred_code)
            # 从映射中获取预测结果的标签
            market_pred_label = market_output_map_raw.get(market_pred_key, f"未知编码 ({market_pred_key})")
        except Exception as e:
            # 捕获预测过程中可能发生的错误
            msg = f"市场细分模型预测时发生错误: {e}"
            print(msg) # 打印错误日志
            error_messages.append(msg) # 记录错误信息
            market_pred_label = "预测失败" # 设置失败状态标签
    else:
        # 如果 check_sufficiency 返回 False (数据不足)
        market_pred_label = "数据不足"

    # --- 2. 价格水平预测 ---
    # 获取价格水平模型所需的特征列表
    price_level_features_needed = feature_names_loaded.get('price_level', [])
    if not price_level_features_needed:
        # 如果特征列表文件没有 'price_level' 键
        st.warning("警告: 未在 feature_names.joblib 中找到 'price_level' 模型的特征列表，无法进行价格水平预测。")
        insufficient_data_flags['price_level'] = True # 标记为数据不足（配置缺失）
        price_level_pred_label = "配置缺失"         # 设置特定状态标签
    elif check_sufficiency('price_level', price_level_features_needed):
        # 如果特征列表存在且数据充足
        try:
            # 准备模型输入数据
            input_data_price_level = {feat: all_inputs[feat] for feat in price_level_features_needed}
            # 创建 DataFrame 并确保列顺序
            input_df_price_level = pd.DataFrame([input_data_price_level])[price_level_features_needed]
            # 使用模型进行预测
            price_level_pred_code_raw = price_level_model.predict(input_df_price_level)[0]
            # 获取价格水平的编码到名称的映射
            price_level_output_map_raw = mappings.get('是否高于区域均价', {})

            # 将预测编码转换为整数，并存储预测编码 (0 或 1)
            if isinstance(price_level_pred_code_raw, (int, np.integer, float)):
                 price_level_pred_key = int(price_level_pred_code_raw)
                 price_level_pred_code = price_level_pred_key # 存储 0 或 1
            else:
                 # 如果预测结果不是数字，视为错误/未知
                 price_level_pred_key = str(price_level_pred_code_raw)
                 price_level_pred_code = -99 # 使用错误/未知编码

            # 从映射中获取预测结果的标签
            price_level_pred_label = price_level_output_map_raw.get(price_level_pred_key, f"未知编码 ({price_level_pred_key})")

        except Exception as e:
            # 捕获预测过程中的错误
            msg = f"价格水平模型预测时发生错误: {e}"
            print(msg) # 打印错误日志
            error_messages.append(msg) # 记录错误信息
            price_level_pred_label = "预测失败" # 设置失败状态标签
            price_level_pred_code = -99      # 确保是错误编码
    else:
        # 如果 check_sufficiency 返回 False (数据不足)
        price_level_pred_label = "数据不足"
        price_level_pred_code = -99 # 使用错误/未知编码表示数据不足

    # --- 3. 均价预测 (回归) ---
    # ***** 使用代码中定义的 REQUIRED_REGRESSION_FEATURES *****
    regression_features_needed = REQUIRED_REGRESSION_FEATURES # 直接使用代码中定义的列表
    print(f"执行均价预测，使用特征: {regression_features_needed}") # 打印日志

    # 检查回归模型所需的数据是否充足
    if check_sufficiency('regression', regression_features_needed):
        try:
            # 准备回归模型的输入数据
            input_data_reg = {feat: all_inputs[feat] for feat in regression_features_needed}
            # 创建 DataFrame，并严格按照 REQUIRED_REGRESSION_FEATURES 的顺序排列列
            input_df_reg = pd.DataFrame([input_data_reg])[regression_features_needed]
            print("均价预测模型输入 DataFrame (原始):", input_df_reg) # 打印原始输入

            # 应用缩放器 - 缩放器必须是使用相同特征和顺序训练的
            try:
                 input_df_reg_scaled = scaler.transform(input_df_reg)
                 print("均价预测模型输入 DataFrame (缩放后):", input_df_reg_scaled) # 打印缩放后的输入
            except ValueError as ve:
                 # 捕获缩放器应用时的 ValueError
                 print(f"缩放器错误: {ve}")
                 # 检查错误消息是否与特征数量或名称不匹配有关
                 if "feature_names mismatch" in str(ve) or "number of features" in str(ve) or "X has" in str(ve):
                      # 获取缩放器期望的特征数量
                      n_scaler_feats = getattr(scaler, 'n_features_in_', '未知数量')
                      # 构建详细的错误信息
                      error_detail = f"缩放器期望 {n_scaler_feats} 个特征, 但提供了 {input_df_reg.shape[1]} 个 ({regression_features_needed})。请确保 'regression_scaler.joblib' 使用相同的特征和顺序进行训练。"
                      # 抛出包含详细信息的 ValueError，这将显示在 Streamlit 界面上
                      raise ValueError(f"缩放器与提供的特征不匹配。{error_detail}") from ve
                 else:
                     # 如果是其他类型的 ValueError，重新抛出
                     raise

            # 使用回归模型进行预测
            unit_price_pred_raw = regression_model.predict(input_df_reg_scaled)[0]
            # 将预测结果转换为浮点数，并确保不小于 0
            unit_price_pred = max(0, float(unit_price_pred_raw))
            print(f"均价预测结果: {unit_price_pred}") # 打印预测结果

        except Exception as e:
            # 捕获预测或缩放过程中的任何其他错误
            msg = f"均价预测模型预测时发生错误: {e}"
            print(msg) # 打印错误日志
            error_messages.append(msg) # 记录错误信息
            unit_price_pred = -1.0 # 设置为错误状态值
    else:
        # 如果 check_sufficiency 返回 False (数据不足)
        unit_price_pred = -1.0 # 设置为错误/数据不足状态值
        # 确保在 check_sufficiency 失败时，insufficient_data_flags 被正确设置
        insufficient_data_flags['regression'] = True

    # --- 结果显示区域 ---
    st.markdown("---") # 分割线
    st.subheader("📈 预测结果分析") # 子标题

    # 定义用于显示的颜色
    market_color = "#1f77b4"          # 市场细分标题颜色 (蓝色)
    price_level_base_color = "#ff7f0e" # 价格水平标题颜色 (橙色)
    unit_price_color = "#2ca02c"      # 均价预测标题颜色 (绿色)
    insufficient_data_color = "#7f7f7f" # 数据不足文本颜色 (灰色)
    error_color = "#d62728"          # 错误/失败文本颜色 (红色)
    config_missing_color = "#ffbb78" # 配置缺失文本颜色 (浅橙色)

    # 创建三列来显示结果
    col1, col2, col3 = st.columns(3)

    # 辅助函数，用于创建统一风格的结果显示块
    def display_result(title, title_color, result_text, result_color):
        """在当前列中显示一个结果块。

        Args:
            title (str): 结果块的标题。
            title_color (str): 标题的 CSS 颜色。
            result_text (str): 要显示的预测结果文本。
            result_color (str): 结果文本的 CSS 颜色。
        """
        # 使用 Markdown 和 HTML 设置标题样式（居中）
        st.markdown(f"<h5 style='color: {title_color}; margin-bottom: 5px; text-align: center;'>{title}</h5>", unsafe_allow_html=True)
        # 使用 Markdown 和 HTML 设置结果文本样式（大字体、粗体、居中）
        st.markdown(f"<p style='font-size: 28px; font-weight: bold; color: {result_color}; margin-bottom: 10px; text-align: center;'>{result_text}</p>", unsafe_allow_html=True)

    # --- 在第一列显示市场细分结果 ---
    with col1:
        title = "市场细分" # 结果块标题
        # 根据预测状态确定显示的文本和颜色
        if market_pred_label == "配置缺失":
             display_text = "特征配置缺失"
             display_color = config_missing_color
        elif insufficient_data_flags['market'] or market_pred_label == "数据不足":
            # 如果标记为数据不足或标签为"数据不足"
            display_text = "数据不足"
            display_color = insufficient_data_color
        elif market_pred_label == "预测失败":
            display_text = "预测失败"
            display_color = error_color
        else:
            # 成功预测
            display_text = market_pred_label
            display_color = market_color # 成功时使用标题颜色
        # 调用辅助函数显示结果
        display_result(title, market_color, display_text, display_color)

    # --- 在第二列显示价格水平结果 ---
    with col2:
        title = "价格水平 (相对区域)" # 结果块标题
        # 根据预测状态确定显示的文本和颜色
        if price_level_pred_label == "配置缺失":
            display_text = "特征配置缺失"
            display_color = config_missing_color
        elif insufficient_data_flags['price_level'] or price_level_pred_label == "数据不足":
            display_text = "数据不足"
            display_color = insufficient_data_color
        elif price_level_pred_label == "预测失败" or price_level_pred_code == -99:
             # 将"预测失败"标签和错误代码 -99 视为同一种失败/未知状态
             display_text = "预测失败/未知"
             display_color = error_color
        elif price_level_pred_code == 1: # 预测为高于区域均价
            display_text = price_level_pred_label # 显示映射的标签 (例如 "高于区域均价")
            display_color = "#E74C3C"          # 使用红色表示偏高
        elif price_level_pred_code == 0: # 预测为不高于区域均价
            display_text = price_level_pred_label # 显示映射的标签 (例如 "低于或持平区域均价")
            display_color = "#2ECC71"          # 使用绿色表示不偏高
        else:
            # 理论上不应到达这里的回退情况
            display_text = "未知状态"
            display_color = insufficient_data_color
        # 调用辅助函数显示结果
        display_result(title, price_level_base_color, display_text, display_color)

    # --- 在第三列显示均价预测结果 ---
    with col3:
        title = "均价预测" # 结果块标题
        # ***** 修改：直接在结果中添加单位，移除下方小字标签 *****
        # 根据预测状态确定显示的文本和颜色
        if insufficient_data_flags['regression']:
            display_text = "数据不足"
            display_color = insufficient_data_color
        elif unit_price_pred == -1.0: # 覆盖了预测错误和因数据不足导致的 -1.0 状态
            display_text = "预测失败/数据不足" # 合并状态的显示文本
            display_color = error_color      # 对合并状态使用错误颜色
        else:
            # 成功预测，格式化数字并添加单位
            display_text = f"{unit_price_pred:,.0f} 元/㎡" # 添加千位分隔符，无小数，带单位
            display_color = unit_price_color # 成功时使用标题颜色
        # 调用辅助函数显示结果
        display_result(title, unit_price_color, display_text, display_color)


    # --- 显示错误信息或成功/警告消息 ---
    if error_messages:
        # 如果在预测过程中捕获了运行时错误
        st.markdown("---") # 分割线
        st.error("执行过程中遇到以下运行时错误：") # 显示错误提示
        for i, msg in enumerate(error_messages):
            # 显示对用户安全的通用错误消息
            st.markdown(f"{i+1}. 分析时出现问题，请检查输入或联系管理员。")
            # 在控制台打印详细错误信息以供调试
            print(f"Detailed Error {i+1}: {msg}")
            # 如果错误与 Scaler 相关，提供额外的提示
            if "缩放器与提供的特征不匹配" in msg:
                 st.warning(f"💡 **提示 (错误 {i+1}):** 检测到均价预测所需的特征与加载的缩放器 (`{os.path.basename(SCALER_PATH)}`) 不匹配。请确保代码中定义的特征列表 (`REQUIRED_REGRESSION_FEATURES`) 与用于训练和保存缩放器的特征列表完全一致（包括顺序）。")

    # 在所有预测尝试完成后，检查最终状态标志
    has_insufficient_data = any(insufficient_data_flags.values()) # 是否有任何模型因数据不足而失败
    # 是否有任何预测失败 (运行时错误 或 预测结果标签/代码表示失败)
    has_errors = bool(error_messages) or market_pred_label == "预测失败" or price_level_pred_label == "预测失败" or unit_price_pred == -1.0

    # 根据最终状态显示不同的总结信息
    # 情况 1: 所有预测成功完成 (没有数据不足，没有错误，没有配置缺失)
    if not has_insufficient_data and not has_errors and market_pred_label != "配置缺失" and price_level_pred_label != "配置缺失":
        st.success("✅ 所有分析预测完成！") # 显示成功消息
        st.markdown("---")
        st.info("💡 **提示:** 模型预测结果是基于历史数据和输入特征的估计，仅供参考。实际交易价格受市场供需、具体房况、谈判等多种因素影响。") # 提供免责声明和提示
    # 情况 2: 部分预测因数据不足或配置缺失未能完成
    elif has_insufficient_data or market_pred_label == "配置缺失" or price_level_pred_label == "配置缺失":
        st.warning("⚠️ 部分预测因输入数据不足或配置缺失未能完成。请在侧边栏提供所有必需的特征信息（避免选择 '无 (不适用)'）。") # 显示警告消息
        st.markdown("---")
        st.info("💡 **提示:** 模型预测结果是基于历史数据和输入特征的估计，仅供参考。实际交易价格受市场供需、具体房况、谈判等多种因素影响。") # 提供免责声明和提示
    # 情况 3: 发生预测失败，但没有捕获到具体的 error_messages (例如，模型内部逻辑问题但未抛出异常)
    elif has_errors and not error_messages:
         st.error("❌ 部分预测失败。请检查输入或联系管理员。") # 显示通用失败消息
         st.markdown("---")
         st.info("💡 **提示:** 模型预测结果是基于历史数据和输入特征的估计，仅供参考。实际交易价格受市场供需、具体房况、谈判等多种因素影响。") # 提供免责声明和提示
    # 情况 4: 如果 error_messages 不为空，则上面的错误块已经显示了信息，这里不再重复显示


# --- 页脚信息 ---
st.sidebar.markdown("---") # 侧边栏分割线
st.sidebar.caption("模型信息: LightGBM & RandomForest") # 显示模型信息
st.sidebar.caption("数据来源: 安居客")          # 澄清数据来源
st.sidebar.caption("开发者: 凌欢")                     # 显示开发者信息