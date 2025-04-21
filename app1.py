# -*- coding: utf-8 -*- # 指定编码为 UTF-8
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 页面基础配置 ---
st.set_page_config(
    page_title="盐城市二手房智能分析器",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- 常量定义：模型和资源文件路径 ---
# 获取脚本所在目录
try:
    # 在作为脚本运行时有效
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 在 __file__ 未定义的 环境（如 Streamlit sharing）中的备用方案
    CURRENT_DIR = os.getcwd()

MARKET_MODEL_PATH = os.path.join(CURRENT_DIR, 'market_segment_lgbm_model.joblib')
PRICE_LEVEL_MODEL_PATH = os.path.join(CURRENT_DIR, 'price_level_rf_model.joblib')
REGRESSION_MODEL_PATH = os.path.join(CURRENT_DIR, 'unit_price_rf_model.joblib')
SCALER_PATH = os.path.join(CURRENT_DIR, 'regression_scaler.joblib')
FEATURE_NAMES_PATH = os.path.join(CURRENT_DIR, 'feature_names.joblib')
MAPPINGS_PATH = os.path.join(CURRENT_DIR, 'mappings.joblib')

# --- ***** 新增：定义均价预测模型所需的固定特征列表 ***** ---
# ***** 注意：这里的特征列表必须与训练回归模型和Scaler时使用的特征完全一致，包括顺序 *****
REQUIRED_REGRESSION_FEATURES = ['所属区域', '房龄', '面积(㎡)', '楼层数', '建造时间', '室', '厅', '卫']
print(f"代码指定均价预测特征: {REQUIRED_REGRESSION_FEATURES}") # 记录此规范

# --- 加载资源函数 (使用缓存) ---
@st.cache_resource
def load_resources():
    """加载所有必要的资源文件 (模型, scaler, 特征名, 映射关系)。"""
    resources = {}
    all_files_exist = True
    required_files = [MARKET_MODEL_PATH, PRICE_LEVEL_MODEL_PATH, REGRESSION_MODEL_PATH,
                      SCALER_PATH, FEATURE_NAMES_PATH, MAPPINGS_PATH]
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 未找到。")
            missing_files.append(os.path.basename(file_path)) # 仅显示文件名
            all_files_exist = False
    if not all_files_exist:
        print(f"错误：缺少文件 {missing_files}。请确保所有 .joblib 文件与 app.py 在同一目录。")
        return None, missing_files
    try:
        resources['market_model'] = joblib.load(MARKET_MODEL_PATH)
        resources['price_level_model'] = joblib.load(PRICE_LEVEL_MODEL_PATH)
        resources['regression_model'] = joblib.load(REGRESSION_MODEL_PATH)
        resources['scaler'] = joblib.load(SCALER_PATH)
        resources['feature_names'] = joblib.load(FEATURE_NAMES_PATH)
        resources['mappings'] = joblib.load(MAPPINGS_PATH)
        print("所有资源加载成功。")

        # --- 验证 feature_names.joblib 中的回归特征 ---
        loaded_reg_features = resources.get('feature_names', {}).get('regression')
        if loaded_reg_features:
            print(f"从 {os.path.basename(FEATURE_NAMES_PATH)} 加载的 'regression' 特征: {loaded_reg_features}")
            if set(loaded_reg_features) != set(REQUIRED_REGRESSION_FEATURES):
                 print(f"警告: 从 {os.path.basename(FEATURE_NAMES_PATH)} 加载的 'regression' 特征与代码中指定的 ({REQUIRED_REGRESSION_FEATURES}) 不完全匹配。将优先使用代码中指定的列表。")
                 # ***** 关键：检查 Scaler 是否与代码指定的特征匹配 *****
                 if hasattr(resources['scaler'], 'n_features_in_') and resources['scaler'].n_features_in_ != len(REQUIRED_REGRESSION_FEATURES):
                      error_msg = f"严重错误: Scaler (来自 {os.path.basename(SCALER_PATH)}) 期望 {resources['scaler'].n_features_in_} 个特征, 但代码指定了 {len(REQUIRED_REGRESSION_FEATURES)} 个回归特征 ({REQUIRED_REGRESSION_FEATURES})。请确保 Scaler 与指定的特征列表一致。"
                      print(error_msg)
                      # 返回 None 以模拟此不匹配导致的加载失败
                      return None, [error_msg]
                 else:
                    print(f"从 {os.path.basename(FEATURE_NAMES_PATH)} 加载的 'regression' 特征与代码指定一致。")
        else:
            print(f"警告: 在 {os.path.basename(FEATURE_NAMES_PATH)} 中未找到 'regression' 特征列表。将使用代码中指定的列表: {REQUIRED_REGRESSION_FEATURES}")
             # ***** 关键：同样检查 Scaler *****
            if hasattr(resources['scaler'], 'n_features_in_') and resources['scaler'].n_features_in_ != len(REQUIRED_REGRESSION_FEATURES):
                error_msg = f"严重错误: Scaler (来自 {os.path.basename(SCALER_PATH)}) 期望 {resources['scaler'].n_features_in_} 个特征, 但代码指定了 {len(REQUIRED_REGRESSION_FEATURES)} 个回归特征 ({REQUIRED_REGRESSION_FEATURES})。请确保 Scaler 与指定的特征列表一致。"
                print(error_msg)
                return None, [error_msg]

        return resources, None
    except Exception as e:
        print(f"加载资源时发生错误: {e}")
        return None, [f"加载错误: {e}"]

resources, load_error_info = load_resources()

# --- 辅助函数 ---
def format_mapping_options_for_selectbox(name_to_code_mapping):
    """为 Streamlit Selectbox 准备选项和格式化函数所需的数据, 增加 '无' 选项。"""
    if not isinstance(name_to_code_mapping, dict):
        print(f"[格式化错误] 输入非字典: {type(name_to_code_mapping)}")
        return {} # 出错时返回空字典

    code_to_display_string = {None: "无 (不适用)"} # 首先添加 '无' 选项

    try:
        # 添加前对原始映射项进行排序
        try:
            # 尝试按整数代码排序
            sorted_items = sorted(name_to_code_mapping.items(), key=lambda item: int(item[1]))
        except ValueError:
             # 如果整数转换失败，则回退到按字符串代码排序
            # print(f"[格式化警告] 无法将所有 code 转换为 int 进行排序，将按字符串排序: {name_to_code_mapping}")
             sorted_items = sorted(name_to_code_mapping.items(), key=lambda item: str(item[1]))

        for name, code in sorted_items:
            try:
                code_key = int(code) # Selectbox 选项通常需要原始类型
            except ValueError:
                code_key = str(code) # 如果无法转换为整数，则保留为字符串

            name_str = str(name)
            code_to_display_string[code_key] = f"{name_str}" # 只显示名称

        return code_to_display_string

    except (TypeError, KeyError, Exception) as e: # 捕获处理过程中的更广泛错误
        print(f"[格式化错误] 转换/排序映射时出错 ({name_to_code_mapping}): {e}")
        # 备用方案：如果排序/转换失败，仅返回 '无' 选项
        return {None: "无 (不适用)"}


# --- Streamlit 用户界面主要部分 ---
st.title("🏠 盐城市二手房智能分析与预测")
st.markdown("""
欢迎使用盐城市二手房分析工具！请在左侧边栏输入房产特征，我们将为您提供三个维度的预测：
1.  **市场细分预测**: 判断房产属于低端、中端还是高端市场
2.  **价格水平预测**: 判断房产单价是否高于其所在区域的平均水平
3.  **房产均价预测**: 预测房产的每平方米单价（元/㎡）
""")
st.markdown("---")

# --- 应用启动时资源加载失败或映射缺失的处理 ---
if not resources:
     st.error("❌ **应用程序初始化失败！**")
     if load_error_info:
         st.warning(f"无法加载必要的资源文件。错误详情:")
         for error in load_error_info:
             st.markdown(f"*   `{error}`")
             # 为缩放器不匹配提供具体指导
             if "Scaler" in error and "特征" in error:
                 st.info(f"💡 **提示:** 上述错误表明用于均价预测的特征缩放器 (`{os.path.basename(SCALER_PATH)}`) 与代码中指定的特征列表 (`{REQUIRED_REGRESSION_FEATURES}`) 不匹配。您需要：\n    1. 确认代码中的 `REQUIRED_REGRESSION_FEATURES` 列表是正确的。\n    2. 使用 **完全相同** 的特征和 **顺序** 重新训练并保存 `regression_scaler.joblib` 文件。")
     else:
         st.warning("无法找到一个或多个必需的资源文件。")
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
     st.stop()

# --- 如果资源加载成功 ---
mappings = resources['mappings']
feature_names_loaded = resources.get('feature_names', {}) # 使用 .get 以确保安全
market_model = resources['market_model']
price_level_model = resources['price_level_model']
regression_model = resources['regression_model']
scaler = resources['scaler']

# 检查核心映射和特征列表是否存在且为预期类型
required_mappings = ['方位', '楼层', '所属区域', '房龄', '市场类别', '是否高于区域均价']
required_features_in_file = ['market', 'price_level'] # 回归特征单独处理
valid_resources = True
missing_or_invalid = []

for key in required_mappings:
    if key not in mappings or not isinstance(mappings.get(key), dict):
        missing_or_invalid.append(f"映射 '{key}' (来自 {os.path.basename(MAPPINGS_PATH)})")
        valid_resources = False

for key in required_features_in_file:
    # feature_names 的值应该是一个列表
    if key not in feature_names_loaded or not isinstance(feature_names_loaded.get(key), list):
        missing_or_invalid.append(f"特征列表 '{key}' (来自 {os.path.basename(FEATURE_NAMES_PATH)})")
        valid_resources = False
# 检查回归键是否存在，即使稍后覆盖它，它也可能指示问题
if 'regression' not in feature_names_loaded:
     print(f"信息: 'regression' 键未在 {os.path.basename(FEATURE_NAMES_PATH)} 中找到。将使用代码中定义的特征列表。")
elif not isinstance(feature_names_loaded.get('regression'), list):
     missing_or_invalid.append(f"特征列表 'regression' (来自 {os.path.basename(FEATURE_NAMES_PATH)}) 格式无效 (应为列表)")
     valid_resources = False


if not valid_resources:
    st.error(f"❌ 资源文件内容不完整或格式错误。缺少或无效的项目:")
    for item in missing_or_invalid:
        st.markdown(f"*   {item}")
    st.stop()

# --- 侧边栏输入控件 ---
st.sidebar.header("🏘️ 房产特征输入")

# --- ***** 修改：字典，将内部特征名映射到用户界面标签 ***** ---
feature_to_label = {
    # 选择项
    '方位': "房屋方位:",
    '楼层': "楼层位置:",
    '所属区域': "所属区域:",
    '房龄': "房龄:",
    # 数值项
    '总价(万)': "总价 (万):",
    '面积(㎡)': "面积 (㎡):",
    '建造时间': "建造时间 (年份):",
    '楼层数': "总楼层数:",
    '室': "室:",
    '厅': "厅:",
    '卫': "卫:"
}

selectbox_inputs = {}
selectbox_labels_map = {} # 用于在需要时将内部键映射回显示标签
all_select_valid = True # 跟踪所有下拉框是否正确加载选项

st.sidebar.subheader("选择项特征")
# 封装下拉框创建逻辑
def create_selectbox(internal_key, help_text, key_suffix):
    global all_select_valid # 允许修改全局标志
    label = feature_to_label.get(internal_key, internal_key) # 从映射中获取标签
    try:
        options_map = mappings[internal_key]
        # 生成包含 '无' 选项的显示映射
        display_map = format_mapping_options_for_selectbox(options_map)

        if not display_map or len(display_map) <= 1: # 应包含 '无' 和至少一个其他选项
             st.sidebar.warning(f"'{label}' 缺少有效选项 (除了'无')。请检查 {os.path.basename(MAPPINGS_PATH)} 中的 '{internal_key}'。")
             if not display_map:
                 display_map = {None: "无 (加载失败)"} # 提供备用方案

        options_codes = list(display_map.keys()) # 键包括 None 和实际代码

        # 确定默认索引 - 尽量避免将 '无' 作为默认值
        default_index = 0 # 如果没有其他选项或逻辑适用，则默认为 '无'
        if len(options_codes) > 1:
            common_defaults = {'楼层': 1, '房龄': 2} # 示例：默认为中间楼层，6-10年
            target_default_code = common_defaults.get(internal_key)

            if target_default_code is not None and target_default_code in options_codes:
                try:
                    default_index = options_codes.index(target_default_code)
                except ValueError:
                    print(f"警告: 找不到用于 {internal_key} 的默认代码 {target_default_code}，选项为 {options_codes}。使用默认值。")
                    default_index = 1 # 默认为第一个非 '无' 的选项
            # 基于选项数量的更智能的默认索引
            elif len(options_codes) > 3: # 选项较多时，选择中间附近的
                 default_index = (len(options_codes) -1) // 2 + 1 # '无' 之后的索引
            elif len(options_codes) >= 2: # 除了 '无' 之外只有一个选项
                default_index = 1 # 选择第一个实际选项

        selected_value = st.sidebar.selectbox(
            label,
            options=options_codes,
            index=default_index,
            format_func=lambda x: display_map.get(x, f"未知 ({x})"),
            key=f"{key_suffix}_select",
            help=help_text
        )
        selectbox_labels_map[internal_key] = label # 存储键到标签的映射
        return selected_value
    except Exception as e:
        st.sidebar.error(f"加载 '{label}' 选项时出错: {e}")
        print(f"加载 {label} 出错的详细信息: {e}") # 在控制台打印详细错误
        all_select_valid = False
        return None

# 使用函数创建下拉选择框
selectbox_inputs['方位'] = create_selectbox('方位', "选择房屋的主要朝向。选择 '无' 如果不确定或不适用。", "orientation")
selectbox_inputs['楼层'] = create_selectbox('楼层', "选择房屋所在楼层的大致位置。选择 '无' 如果不确定或不适用。", "floor_level")
selectbox_inputs['所属区域'] = create_selectbox('所属区域', "选择房产所在的行政区域或板块。选择 '无' 如果不确定或不适用。", "district")
selectbox_inputs['房龄'] = create_selectbox('房龄', "选择房屋的建造年限范围。选择 '无' 如果不确定或不适用。", "age")

# --- ***** 修改：数值输入控件，增加 "无" 选项 ***** ---
st.sidebar.subheader("数值项特征")
numeric_inputs = {}
numeric_input_states = {} # 用于存储状态 ('输入数值' 或 '无')

# 定义默认数值（仅在选择 '输入数值' 时使用）
default_numeric_values = {
    '总价(万)': 120.0,
    '面积(㎡)': 100.0,
    '建造时间': 2018,
    '楼层数': 30,
    '室': 3,
    '厅': 2,
    '卫': 2
}

# 定义数值输入参数
numeric_params = {
    '总价(万)': {'min_value': 0.0, 'max_value': 10000.0, 'step': 5.0, 'format': "%.1f", 'help': "输入房产的总价，单位万元。留空或选择 '无' 表示不适用。"},
    '面积(㎡)': {'min_value': 1.0, 'max_value': 2000.0, 'step': 1.0, 'format': "%.1f", 'help': "输入房产的建筑面积，单位平方米。留空或选择 '无' 表示不适用。"},
    '建造时间': {'min_value': 1900, 'max_value': 2025, 'step': 1, 'format': "%d", 'help': "输入房屋的建造年份。留空或选择 '无' 表示不适用。"},
    '楼层数': {'min_value': 1, 'max_value': 100, 'step': 1, 'format': "%d", 'help': "输入楼栋的总楼层数。留空或选择 '无' 表示不适用。"},
    '室': {'min_value': 0, 'max_value': 20, 'step': 1, 'format': "%d", 'help': "输入卧室数量。留空或选择 '无' 表示不适用。"},
    '厅': {'min_value': 0, 'max_value': 10, 'step': 1, 'format': "%d", 'help': "输入客厅/餐厅数量。留空或选择 '无' 表示不适用。"},
    '卫': {'min_value': 0, 'max_value': 10, 'step': 1, 'format': "%d", 'help': "输入卫生间数量。留空或选择 '无' 表示不适用。"}
}

# 为数值特征创建组合输入小部件
for key, label in feature_to_label.items():
    if key in numeric_params: # 检查是否是我们处理的数值特征
        param = numeric_params[key]
        default_val = default_numeric_values[key]
        key_suffix = key.replace('(','').replace(')','').replace('㎡','') # 创建一个简单的键后缀

        # 用于选择输入值或指定 '无' 的选择器
        numeric_input_states[key] = st.sidebar.selectbox(
            label, # 使用 feature_to_label 中定义的标签
            options=["输入数值", "无"],
            index=0,  # 默认为 "输入数值"
            key=f"{key_suffix}_selector",
            help=param['help']
        )

        # 条件性地显示数字输入框
        if numeric_input_states[key] == "输入数值":
            numeric_inputs[key] = st.sidebar.number_input(
                f"输入 {label}", # 稍微修改标签以更清晰
                min_value=param['min_value'],
                max_value=param['max_value'],
                value=default_val,
                step=param['step'],
                format=param['format'],
                key=f"{key_suffix}_value",
                label_visibility="collapsed" # 隐藏标签，因为它已由选择器隐含
            )
        else:
            # 如果选择了 "无 (不适用)"，则为此特征存储 None
            numeric_inputs[key] = None
            # 可选地，显示禁用的占位符或不显示任何内容
            # st.sidebar.text_input(f"{label}", value="不适用", disabled=True, key=f"{key_suffix}_value_disabled")


# --- 预测触发按钮 ---
st.sidebar.markdown("---")
predict_button_disabled = not all_select_valid # 即使某些数值为 None 仍可预测
predict_button_help = "点击这里根据输入的特征进行预测分析" if all_select_valid else "部分下拉框选项加载失败，无法进行预测。请检查资源文件或错误信息。"

if st.sidebar.button("🚀 开始分析预测", type="primary", use_container_width=True, help=predict_button_help, disabled=predict_button_disabled):

    # --- ***** 修改：整合输入时处理 None 值 ***** ---
    # 从下拉选择框输入开始
    all_inputs = {**selectbox_inputs}
    # 添加数值输入，尊重来自选择器的 'None' 状态
    for key, state in numeric_input_states.items():
        if state == "无":
            all_inputs[key] = None # 如果选择了 '无' 则存储 None
        else:
            # 从相应的 number_input 小部件检索值
            key_suffix = key.replace('(','').replace(')','').replace('㎡','')
            all_inputs[key] = st.session_state[f"{key_suffix}_value"] # 使用其键获取值

    print("Combined inputs for prediction:", all_inputs) # 调试输出

    # --- 初始化结果变量 ---
    market_pred_label = "等待计算..."
    price_level_pred_label = "等待计算..."
    price_level_pred_code = -99 # 对 '未预测' 或 '错误' 使用不同的代码
    unit_price_pred = -1.0 # 对 '未预测' 或 '错误' 使用 -1.0
    error_messages = []
    insufficient_data_flags = {'market': False, 'price_level': False, 'regression': False}

    # --- ***** 修改：Helper Function to Check Input Sufficiency (Handles None) ***** ---
    def check_sufficiency(model_key, required_feature_list):
        """检查模型所需的所有特征是否具有有效（非 None）值。"""
        missing_for_model = []
        for feat in required_feature_list:
            # 检查特征是否存在于组合输入中，以及其值是否为 None
            if feat not in all_inputs:
                 # 这是一个关键配置错误 - 所需特征未在 UI 中定义！
                 print(f"严重警告: 模型 '{model_key}' 需要的特征 '{feat}' 在UI输入中未定义!")
                 missing_for_model.append(f"{feature_to_label.get(feat, feat)} (UI未定义)")
            elif all_inputs[feat] is None:
                # 特征存在，但其值为 None（用户选择了 '无' 或加载失败）
                missing_for_model.append(feature_to_label.get(feat, feat)) # 使用显示标签

        if missing_for_model:
            print(f"模型 '{model_key}' 数据不足，缺少: {missing_for_model}")
            insufficient_data_flags[model_key] = True
            return False # 数据不足
        return True # 数据充足


    # --- 1. 市场细分预测 ---
    market_features_needed = feature_names_loaded.get('market', [])
    if not market_features_needed:
         st.warning("警告: 未在 feature_names.joblib 中找到 'market' 模型的特征列表，无法进行市场细分预测。")
         insufficient_data_flags['market'] = True # 标记为不足
         market_pred_label = "配置缺失" # 特定状态
    elif check_sufficiency('market', market_features_needed):
        try:
            # 仅筛选此模型所需的非 None 输入
            input_data_market = {feat: all_inputs[feat] for feat in market_features_needed}
            input_df_market = pd.DataFrame([input_data_market])[market_features_needed] # 确保顺序
            market_pred_code = market_model.predict(input_df_market)[0]
            market_output_map_raw = mappings.get('市场类别', {})
            # 确保预测代码被视为正确的类型以进行映射查找
            market_pred_key = int(market_pred_code) if isinstance(market_pred_code, (int, np.integer, float)) else str(market_pred_code)
            market_pred_label = market_output_map_raw.get(market_pred_key, f"未知编码 ({market_pred_key})")
        except Exception as e:
            msg = f"市场细分模型预测时发生错误: {e}"
            print(msg)
            error_messages.append(msg)
            market_pred_label = "预测失败" # 指示运行时错误
    else:
        # check_sufficiency 返回 False
        market_pred_label = "数据不足"

    # --- 2. 价格水平预测 ---
    price_level_features_needed = feature_names_loaded.get('price_level', [])
    if not price_level_features_needed:
        st.warning("警告: 未在 feature_names.joblib 中找到 'price_level' 模型的特征列表，无法进行价格水平预测。")
        insufficient_data_flags['price_level'] = True
        price_level_pred_label = "配置缺失"
    elif check_sufficiency('price_level', price_level_features_needed):
        try:
            input_data_price_level = {feat: all_inputs[feat] for feat in price_level_features_needed}
            input_df_price_level = pd.DataFrame([input_data_price_level])[price_level_features_needed] # 确保顺序
            price_level_pred_code_raw = price_level_model.predict(input_df_price_level)[0]
            price_level_output_map_raw = mappings.get('是否高于区域均价', {})

            # 确定映射的键类型并存储代码
            if isinstance(price_level_pred_code_raw, (int, np.integer, float)):
                 price_level_pred_key = int(price_level_pred_code_raw)
                 price_level_pred_code = price_level_pred_key # 存储 0 或 1
            else:
                 price_level_pred_key = str(price_level_pred_code_raw)
                 price_level_pred_code = -99 # 错误/未知代码

            price_level_pred_label = price_level_output_map_raw.get(price_level_pred_key, f"未知编码 ({price_level_pred_key})")

        except Exception as e:
            msg = f"价格水平模型预测时发生错误: {e}"
            print(msg)
            error_messages.append(msg)
            price_level_pred_label = "预测失败"
            price_level_pred_code = -99 # 确保错误代码
    else:
        # check_sufficiency 返回 False
        price_level_pred_label = "数据不足"
        price_level_pred_code = -99 # 如果需要，指示数据不足状态

    # --- 3. 均价预测 (回归) ---
    # ***** 使用代码中定义的 REQUIRED_REGRESSION_FEATURES *****
    regression_features_needed = REQUIRED_REGRESSION_FEATURES
    print(f"执行均价预测，使用特征: {regression_features_needed}") # 记录正在使用的特征

    if check_sufficiency('regression', regression_features_needed):
        try:
            # 使用 REQUIRED_REGRESSION_FEATURES 列表准备数据
            input_data_reg = {feat: all_inputs[feat] for feat in regression_features_needed}
            # 创建列顺序与 REQUIRED_REGRESSION_FEATURES 完全一致的 DataFrame
            input_df_reg = pd.DataFrame([input_data_reg])[regression_features_needed]
            print("均价预测模型输入 DataFrame (原始):", input_df_reg)

            # 应用缩放器 - 必须与训练时使用的特征和顺序匹配
            try:
                 input_df_reg_scaled = scaler.transform(input_df_reg)
                 print("均价预测模型输入 DataFrame (缩放后):", input_df_reg_scaled)
            except ValueError as ve:
                 print(f"缩放器错误: {ve}")
                 # 检查错误消息是否关于特征名称/数量不匹配
                 if "feature_names mismatch" in str(ve) or "number of features" in str(ve) or "X has" in str(ve):
                      n_scaler_feats = getattr(scaler, 'n_features_in_', '未知数量')
                      error_detail = f"缩放器期望 {n_scaler_feats} 个特征, 但提供了 {input_df_reg.shape[1]} 个 ({regression_features_needed})。请确保 'regression_scaler.joblib' 使用相同的特征和顺序进行训练。"
                      raise ValueError(f"缩放器与提供的特征不匹配。{error_detail}") from ve
                 else:
                     raise # 重新引发其他缩放器错误

            unit_price_pred_raw = regression_model.predict(input_df_reg_scaled)[0]
            unit_price_pred = max(0, float(unit_price_pred_raw)) # 确保非负浮点数
            print(f"均价预测结果: {unit_price_pred}")

        except Exception as e:
            msg = f"均价预测模型预测时发生错误: {e}"
            print(msg)
            error_messages.append(msg)
            unit_price_pred = -1.0 # 标记为错误
    else:
        # check_sufficiency 返回 False
        unit_price_pred = -1.0 # 标记为数据不足/错误状态
        # 如果 check_sufficiency 失败，确保正确设置标志
        insufficient_data_flags['regression'] = True

    # --- 结果显示区域 ---
    st.markdown("---")
    st.subheader("📈 预测结果分析")

    # 定义颜色
    market_color = "#1f77b4"  # 蓝色
    price_level_base_color = "#ff7f0e" # 橙色（用于标题）
    unit_price_color = "#2ca02c" # 绿色
    insufficient_data_color = "#7f7f7f" # 灰色
    error_color = "#d62728" # 红色
    config_missing_color = "#ffbb78" # 浅橙色，用于配置问题


    col1, col2, col3 = st.columns(3)

    # 用于创建一致结果显示块的辅助函数
    def display_result(title, title_color, result_text, result_color):
        st.markdown(f"<h2 style='color: {title_color}; margin-bottom: 5px; text-align: center;'>{title}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 26px; font-weight: bold; color: {result_color}; margin-bottom: 10px; text-align: center;'>{result_text}</p>", unsafe_allow_html=True)


    with col1: # 市场细分
        title = "市场细分"
        if market_pred_label == "配置缺失":
             display_text = "特征配置缺失"
             display_color = config_missing_color
        elif insufficient_data_flags['market'] or market_pred_label == "数据不足":
            display_text = "数据不足"
            display_color = insufficient_data_color
        elif market_pred_label == "预测失败":
            display_text = "预测失败"
            display_color = error_color
        else:
            display_text = market_pred_label
            display_color = market_color # 对结果使用标题颜色
        display_result(title, market_color, display_text, display_color)


    with col2: # 价格水平
        title = "价格水平 (相对区域)"
        if price_level_pred_label == "配置缺失":
            display_text = "特征配置缺失"
            display_color = config_missing_color
        elif insufficient_data_flags['price_level'] or price_level_pred_label == "数据不足":
            display_text = "数据不足"
            display_color = insufficient_data_color
        elif price_level_pred_label == "预测失败" or price_level_pred_code == -99:
             # 将 -99 代码（错误或初始状态）视为与显式失败标签相同
             display_text = "预测失败/未知" # 组合状态
             display_color = error_color
        elif price_level_pred_code == 1: # 高于平均水平
            display_text = price_level_pred_label
            display_color = "#ff7f0e" # 红色表示更高
        elif price_level_pred_code == 0: # 不高于平均水平
            display_text = price_level_pred_label
            display_color = "#ff7f0e" # 绿色表示不高于
        else: # 当前逻辑不应发生，但包含备用方案
            display_text = "未知状态"
            display_color = insufficient_data_color
        display_result(title, price_level_base_color, display_text, display_color)


    with col3: # 均价预测
        title = "均价预测"
        # ***** 修改：直接在结果中添加单位，移除下方小字标签 *****
        if insufficient_data_flags['regression']:
            display_text = "数据不足"
            display_color = insufficient_data_color
        elif unit_price_pred == -1.0: # 涵盖预测错误和检查失败时的初始 '不足' 状态
            display_text = "预测失败/数据不足"
            display_color = error_color # 对此组合状态使用错误颜色
        else:
            # 格式化成功预测的价格（带单位）
            display_text = f"{unit_price_pred:,.0f} 元/㎡"
            display_color = unit_price_color # 对结果使用标题颜色
        display_result(title, unit_price_color, display_text, display_color)


    # --- 显示错误或成功/警告消息 ---
    if error_messages:
        st.markdown("---")
        st.error("执行过程中遇到以下运行时错误：")
        for i, msg in enumerate(error_messages):
            # 向用户显示更安全的消息，记录详细信息
            st.markdown(f"{i+1}. 分析时出现问题，请检查输入或联系管理员。")
            print(f"Detailed Error {i+1}: {msg}") # 记录实际错误以供调试
            if "缩放器与提供的特征不匹配" in msg: # 为缩放器问题提供具体指导
                 st.warning(f"💡 **提示 (错误 {i+1}):** 检测到均价预测所需的特征与加载的缩放器 (`{os.path.basename(SCALER_PATH)}`) 不匹配。请确保代码中定义的特征列表 (`REQUIRED_REGRESSION_FEATURES`) 与用于训练和保存缩放器的特征列表完全一致（包括顺序）。")

    # 在预测后检查标志以提供准确的状态摘要
    has_insufficient_data = any(insufficient_data_flags.values())
    has_errors = bool(error_messages) or market_pred_label == "预测失败" or price_level_pred_label == "预测失败" or unit_price_pred == -1.0

    # 根据结果显示摘要消息
    if not has_insufficient_data and not has_errors and market_pred_label != "配置缺失" and price_level_pred_label != "配置缺失":
        st.success("✅ 所有分析预测完成！")
        st.markdown("---")
        st.info("💡 **提示:** 模型预测结果是基于历史数据和输入特征的估计，仅供参考。实际交易价格受市场供需、具体房况、谈判等多种因素影响。")
    elif has_insufficient_data or market_pred_label == "配置缺失" or price_level_pred_label == "配置缺失":
        st.warning("⚠️ 部分预测因输入数据不足或配置缺失未能完成。请在侧边栏提供所有必需的特征信息（避免选择'无'）")
        st.markdown("---")
        st.info("💡 **提示:** 模型预测结果是基于历史数据和输入特征的估计，仅供参考。实际交易价格受市场供需、具体房况、谈判等多种因素影响。")
    elif has_errors and not error_messages: # 处理预测失败但未抛出上述异常的情况
         st.error("❌ 部分预测失败。请检查输入或联系管理员。")
         st.markdown("---")
         st.info("💡 **提示:** 模型预测结果是基于历史数据和输入特征的估计，仅供参考。实际交易价格受市场供需、具体房况、谈判等多种因素影响")
    # 如果 error_messages 不为空，则上面的错误块已显示。


# --- 页脚信息 ---
st.sidebar.markdown("---")
st.sidebar.caption("模型信息: LightGBM & RandomForest")
st.sidebar.caption("数据来源: 安居客") # 如果是示例，请说明数据来源
st.sidebar.caption("开发者: 凌欢")