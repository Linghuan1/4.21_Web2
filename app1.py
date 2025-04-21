# -*- coding: utf-8 -*- # 指定编码为 UTF-8
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 页面基础配置 ---
st.set_page_config(
    page_title="盐城二手房智能分析器",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 常量定义：模型和资源文件路径 ---
# Assume files are in the same directory as the script
# --- 常量定义：模型和资源文件路径 ---
# Get the directory where the script is located
try:
    # This works when running as a script
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for environments where __file__ is not defined (like Streamlit sharing)
    CURRENT_DIR = os.getcwd()

MARKET_MODEL_PATH = os.path.join(CURRENT_DIR, 'market_segment_lgbm_model.joblib')
PRICE_LEVEL_MODEL_PATH = os.path.join(CURRENT_DIR, 'price_level_rf_model.joblib')
REGRESSION_MODEL_PATH = os.path.join(CURRENT_DIR, 'unit_price_rf_model.joblib')
SCALER_PATH = os.path.join(CURRENT_DIR, 'regression_scaler.joblib')
FEATURE_NAMES_PATH = os.path.join(CURRENT_DIR, 'feature_names.joblib')
MAPPINGS_PATH = os.path.join(CURRENT_DIR, 'mappings.joblib')

# --- ***** 新增：定义均价预测模型所需的固定特征列表 ***** ---
REQUIRED_REGRESSION_FEATURES = ['所属区域', '房龄', '面积(㎡)', '楼层数', '建造时间', '室', '厅', '卫']
print(f"代码指定均价预测特征: {REQUIRED_REGRESSION_FEATURES}") # Log this specification

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
            missing_files.append(os.path.basename(file_path)) # Show only filename
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
            else:
                 print(f"从 {os.path.basename(FEATURE_NAMES_PATH)} 加载的 'regression' 特征与代码指定一致。")
        else:
            print(f"警告: 在 {os.path.basename(FEATURE_NAMES_PATH)} 中未找到 'regression' 特征列表。将使用代码中指定的列表: {REQUIRED_REGRESSION_FEATURES}")

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
        return {} # Return empty dict on error

    code_to_display_string = {None: "无 (不适用)"} # Add the 'None' option first

    try:
        # Sort the original mapping items before adding them
        try:
            # Try sorting by integer code
            sorted_items = sorted(name_to_code_mapping.items(), key=lambda item: int(item[1]))
        except ValueError:
             # Fallback to sorting by string code if int conversion fails
            #  print(f"[格式化警告] 无法将所有 code 转换为 int 进行排序，将按字符串排序: {name_to_code_mapping}")
             sorted_items = sorted(name_to_code_mapping.items(), key=lambda item: str(item[1]))

        for name, code in sorted_items:
            try:
                code_key = int(code) # Selectbox options usually need primitive types
            except ValueError:
                code_key = str(code) # Keep as string if not convertible to int

            name_str = str(name)
            code_to_display_string[code_key] = f"{name_str}" # Just show name

        return code_to_display_string

    except (TypeError, KeyError, Exception) as e: # Catch broader errors during processing
        print(f"[格式化错误] 转换/排序映射时出错 ({name_to_code_mapping}): {e}")
        # Fallback: return only the 'None' option if sorting/conversion fails
        return {None: "无 (不适用)"}


# --- Streamlit 用户界面主要部分 ---
st.title("🏠 盐城二手房智能分析与预测")
st.markdown("""
欢迎使用盐城二手房分析工具！请在左侧边栏输入房产特征，我们将为您提供三个维度的预测：
1.  **市场细分预测**: 判断房产属于低端、中端还是高端市场。
2.  **价格水平预测**: 判断房产单价是否高于其所在区域的平均水平。
3.  **房产均价预测**: 预测房产的每平方米单价（元/㎡）。
""")
st.markdown("---")

# --- 应用启动时资源加载失败或映射缺失的处理 ---
if not resources:
     st.error("❌ **应用程序初始化失败！**")
     if load_error_info:
         st.warning(f"无法加载必要的资源文件。错误详情:")
         for error in load_error_info:
             st.markdown(f"*   `{error}`")
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
# feature_names is loaded but we will prioritize REQUIRED_REGRESSION_FEATURES for regression
feature_names_loaded = resources.get('feature_names', {}) # Use .get for safety
market_model = resources['market_model']
price_level_model = resources['price_level_model']
regression_model = resources['regression_model']
scaler = resources['scaler']

# 检查核心映射和特征列表是否存在且为预期类型
required_mappings = ['方位', '楼层', '所属区域', '房龄', '市场类别', '是否高于区域均价']
# We still load feature_names, but check specific model requirements later
required_features_in_file = ['market', 'price_level'] # Regression handled separately
valid_resources = True
missing_or_invalid = []

for key in required_mappings:
    if key not in mappings or not isinstance(mappings.get(key), dict):
        missing_or_invalid.append(f"映射 '{key}' (来自 {os.path.basename(MAPPINGS_PATH)})")
        valid_resources = False

for key in required_features_in_file:
    # feature_names value should be a list
    if key not in feature_names_loaded or not isinstance(feature_names_loaded.get(key), list):
        missing_or_invalid.append(f"特征列表 '{key}' (来自 {os.path.basename(FEATURE_NAMES_PATH)})")
        valid_resources = False
# Check if regression key exists, even if we override it later, it might indicate issues
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
st.sidebar.subheader("选择项特征")
selectbox_inputs = {}
selectbox_labels_map = {} # To map internal key back to display label if needed
all_select_valid = True # Track if all dropdowns load options correctly

# 字典，将内部特征名映射到用户界面标签
feature_to_label = {
    '方位': "房屋方位:",
    '楼层': "楼层位置:",
    '所属区域': "所属区域:",
    '房龄': "房龄:",
    '总价(万)': "总价 (万):",
    '面积(㎡)': "面积 (㎡):",
    '建造时间': "建造时间 (年份):",
    '楼层数': "总楼层数:",
    '室': "室:",
    '厅': "厅:",
    '卫': "卫:"
}


# 封装下拉框创建逻辑
def create_selectbox(label, mapping_key, help_text, key_suffix):
    global all_select_valid # Allow modification of the global flag
    try:
        options_map = mappings[mapping_key]
        # Generate display map including the 'None' option
        display_map = format_mapping_options_for_selectbox(options_map)

        if not display_map or len(display_map) <= 1: # Should have 'None' + at least one other
             st.sidebar.warning(f"'{label}' 缺少有效选项 (除了'无')。请检查 {os.path.basename(MAPPINGS_PATH)} 中的 '{mapping_key}'。")
             if not display_map:
                 display_map = {None: "无 (加载失败)"} # Provide a fallback

        options_codes = list(display_map.keys()) # Keys include None and the actual codes

        # Determine default index - try to avoid 'None' as default
        default_index = 0 # Default to '无' if no other options or logic applies
        if len(options_codes) > 1:
            common_defaults = {'楼层': 1, '房龄': 2}
            target_default_code = common_defaults.get(mapping_key)

            if target_default_code is not None and target_default_code in options_codes:
                try:
                    default_index = options_codes.index(target_default_code)
                except ValueError:
                    print(f"Warning: Default code {target_default_code} for {mapping_key} not found in options {options_codes}. Defaulting.")
                    default_index = 1 # Default to the first non-'None' option
            elif len(options_codes) > 2:
                 default_index = len(options_codes) // 2
            elif len(options_codes) == 2:
                default_index = 1

        selected_value = st.sidebar.selectbox(
            label,
            options=options_codes,
            index=default_index,
            format_func=lambda x: display_map.get(x, f"未知 ({x})"),
            key=f"{key_suffix}_select",
            help=help_text
        )
        selectbox_labels_map[mapping_key] = label # Store mapping key to label
        return selected_value
    except Exception as e:
        st.sidebar.error(f"加载 '{label}' 选项时出错: {e}")
        print(f"Error details for loading {label}: {e}") # Print detailed error to console
        all_select_valid = False
        return None


selectbox_inputs['方位'] = create_selectbox(feature_to_label['方位'], '方位', "选择房屋的主要朝向。选择 '无' 如果不确定或不适用。", "orientation")
selectbox_inputs['楼层'] = create_selectbox(feature_to_label['楼层'], '楼层', "选择房屋所在楼层的大致位置。选择 '无' 如果不确定或不适用。", "floor_level")
selectbox_inputs['所属区域'] = create_selectbox(feature_to_label['所属区域'], '所属区域', "选择房产所在的行政区域或板块。选择 '无' 如果不确定或不适用。", "district")
selectbox_inputs['房龄'] = create_selectbox(feature_to_label['房龄'], '房龄', "选择房屋的建造年限范围。选择 '无' 如果不确定或不适用。", "age")

# --- 数值输入控件 ---
st.sidebar.subheader("数值项特征")
numeric_inputs = {}
numeric_inputs['总价(万)'] = st.sidebar.number_input(feature_to_label['总价(万)'], min_value=0.0, max_value=10000.0, value=120.0, step=5.0, format="%.1f", key="total_price", help="输入房产的总价，单位万元。")
numeric_inputs['面积(㎡)'] = st.sidebar.number_input(feature_to_label['面积(㎡)'], min_value=1.0, max_value=2000.0, value=95.0, step=1.0, format="%.1f", key="area_sqm", help="输入房产的建筑面积，单位平方米。")
numeric_inputs['建造时间'] = st.sidebar.number_input(feature_to_label['建造时间'], min_value=1900, max_value=2025, value=2015, step=1, format="%d", key="build_year", help="输入房屋的建造年份。")
numeric_inputs['楼层数'] = st.sidebar.number_input(feature_to_label['楼层数'], min_value=1, max_value=100, value=18, step=1, format="%d", key="floor_num", help="输入楼栋的总楼层数。")
numeric_inputs['室'] = st.sidebar.number_input(feature_to_label['室'], min_value=0, max_value=20, value=3, step=1, format="%d", key="rooms", help="输入卧室数量。")
numeric_inputs['厅'] = st.sidebar.number_input(feature_to_label['厅'], min_value=0, max_value=10, value=2, step=1, format="%d", key="halls", help="输入客厅/餐厅数量。")
numeric_inputs['卫'] = st.sidebar.number_input(feature_to_label['卫'], min_value=0, max_value=10, value=1, step=1, format="%d", key="baths", help="输入卫生间数量。")

# --- 预测触发按钮 ---
st.sidebar.markdown("---")
predict_button_disabled = not all_select_valid
predict_button_help = "点击这里根据输入的特征进行预测分析" if all_select_valid else "部分下拉框选项加载失败，无法进行预测。请检查资源文件或错误信息。"

if st.sidebar.button("🚀 开始分析预测", type="primary", use_container_width=True, help=predict_button_help, disabled=predict_button_disabled):

    all_inputs = {**selectbox_inputs, **numeric_inputs}
    print("Combined inputs for prediction:", all_inputs) # Debugging output

    # --- Initialize result variables ---
    market_pred_label = "等待计算..."
    price_level_pred_label = "等待计算..."
    price_level_pred_code = -99
    unit_price_pred = -1.0
    error_messages = []
    insufficient_data_flags = {'market': False, 'price_level': False, 'regression': False}

    # --- Helper Function to Check Input Sufficiency ---
    def check_sufficiency(model_key, required_feature_list):
        """Checks if all required features for a model are present (not None)."""
        missing_for_model = []
        for feat in required_feature_list:
            # Check if the feature is expected from a selectbox OR numeric input
            is_selectbox_feature = feat in selectbox_inputs
            is_numeric_feature = feat in numeric_inputs

            if is_selectbox_feature and all_inputs.get(feat) is None:
                # Use the label from feature_to_label mapping
                missing_for_model.append(feature_to_label.get(feat, feat))
            # No check for numeric inputs being None needed, as number_input prevents it
            elif not is_selectbox_feature and not is_numeric_feature:
                 # This case means a required feature isn't in the UI inputs
                 print(f"严重警告: 模型 '{model_key}' 需要的特征 '{feat}' 在UI输入中未定义!")
                 missing_for_model.append(f"{feat} (UI未定义)")

        if missing_for_model:
            print(f"模型 '{model_key}' 数据不足，缺少: {missing_for_model}")
            insufficient_data_flags[model_key] = True
            return False
        return True

    # --- 1. 市场细分预测 ---
    market_features_needed = feature_names_loaded.get('market', [])
    if not market_features_needed:
         st.warning("警告: 未在 feature_names.joblib 中找到 'market' 模型的特征列表，无法进行市场细分预测。")
         insufficient_data_flags['market'] = True # Mark as insufficient
         market_pred_label = "配置缺失" # Specific status
    elif check_sufficiency('market', market_features_needed):
        try:
            input_data_market = {feat: all_inputs[feat] for feat in market_features_needed}
            input_df_market = pd.DataFrame([input_data_market])[market_features_needed] # Ensure order
            market_pred_code = market_model.predict(input_df_market)[0]
            market_output_map_raw = mappings.get('市场类别', {})
            market_pred_key = int(market_pred_code) if isinstance(market_pred_code, (int, np.integer)) else str(market_pred_code)
            market_pred_label = market_output_map_raw.get(market_pred_key, f"未知编码 ({market_pred_key})")
        except Exception as e:
            msg = f"市场细分模型预测时发生错误: {e}"
            print(msg)
            error_messages.append(msg)
            market_pred_label = "预测失败" # Indicate runtime error

    # --- 2. 价格水平预测 ---
    price_level_features_needed = feature_names_loaded.get('price_level', [])
    if not price_level_features_needed:
        st.warning("警告: 未在 feature_names.joblib 中找到 'price_level' 模型的特征列表，无法进行价格水平预测。")
        insufficient_data_flags['price_level'] = True
        price_level_pred_label = "配置缺失"
    elif check_sufficiency('price_level', price_level_features_needed):
        try:
            input_data_price_level = {feat: all_inputs[feat] for feat in price_level_features_needed}
            input_df_price_level = pd.DataFrame([input_data_price_level])[price_level_features_needed] # Ensure order
            price_level_pred_code_raw = price_level_model.predict(input_df_price_level)[0]
            price_level_output_map_raw = mappings.get('是否高于区域均价', {})

            if isinstance(price_level_pred_code_raw, (int, np.integer)):
                 price_level_pred_key = int(price_level_pred_code_raw)
                 price_level_pred_code = price_level_pred_key
            else:
                 price_level_pred_key = str(price_level_pred_code_raw)
                 price_level_pred_code = -99

            price_level_pred_label = price_level_output_map_raw.get(price_level_pred_key, f"未知编码 ({price_level_pred_key})")

        except Exception as e:
            msg = f"价格水平模型预测时发生错误: {e}"
            print(msg)
            error_messages.append(msg)
            price_level_pred_label = "预测失败"
            price_level_pred_code = -99

    # --- 3. 均价预测 (回归) ---
    # ***** 使用代码中定义的 REQUIRED_REGRESSION_FEATURES *****
    regression_features_needed = REQUIRED_REGRESSION_FEATURES
    print(f"执行均价预测，使用特征: {regression_features_needed}") # Log features being used

    if check_sufficiency('regression', regression_features_needed):
        try:
            # Prepare data using the REQUIRED_REGRESSION_FEATURES list
            input_data_reg = {}
            for feat in regression_features_needed:
                 if feat not in all_inputs:
                     # This should ideally be caught by check_sufficiency, but double-check
                     raise ValueError(f"内部错误: 必需的回归特征 '{feat}' 未在 'all_inputs' 中找到。")
                 input_data_reg[feat] = all_inputs[feat]

            # Create DataFrame with columns in the exact order of REQUIRED_REGRESSION_FEATURES
            input_df_reg = pd.DataFrame([input_data_reg])[regression_features_needed]
            print("均价预测模型输入 DataFrame (原始):", input_df_reg)

            # Apply scaler
            # Ensure the scaler was trained with features in the *same order*
            try:
                 input_df_reg_scaled = scaler.transform(input_df_reg)
                 print("均价预测模型输入 DataFrame (缩放后):", input_df_reg_scaled)
            except ValueError as ve:
                 print(f"缩放器错误: {ve}")
                 # Check if the error message is about feature names/number mismatch
                 if "feature_names mismatch" in str(ve) or "number of features" in str(ve):
                     raise ValueError(f"缩放器与提供的特征 ({regression_features_needed}) 不匹配。请确保 'regression_scaler.joblib' 使用相同的特征和顺序进行训练。") from ve
                 else:
                     raise # Re-raise other scaler errors

            unit_price_pred_raw = regression_model.predict(input_df_reg_scaled)[0]
            unit_price_pred = max(0, float(unit_price_pred_raw)) # Ensure non-negative float
            print(f"均价预测结果: {unit_price_pred}")

        except Exception as e:
            msg = f"均价预测模型预测时发生错误: {e}"
            print(msg)
            error_messages.append(msg)
            unit_price_pred = -1.0 # Mark as error

    # --- 结果显示区域 ---
    st.markdown("---")
    st.subheader("📈 预测结果分析")

    # Define colors
    market_color = "#1f77b4"  # Blue
    price_level_base_color = "#ff7f0e" # Orange (for title)
    unit_price_color = "#2ca02c" # Green
    insufficient_data_color = "#7f7f7f" # Grey
    error_color = "#d62728" # Red
    config_missing_color = "#ffbb78" # Light orange for config issue


    col1, col2, col3 = st.columns(3)

    with col1: # Market Segment
        st.markdown(f"<h5 style='color: {market_color}; margin-bottom: 5px;'>市场细分</h5>", unsafe_allow_html=True)
        if market_pred_label == "配置缺失":
             display_text = "特征配置缺失"
             display_color = config_missing_color
        elif insufficient_data_flags['market']:
            display_text = "数据不足，无法判断"
            display_color = insufficient_data_color
        elif market_pred_label == "预测失败":
            display_text = "预测失败"
            display_color = error_color
        else:
            display_text = market_pred_label
            display_color = market_color # Use title color for result
        st.markdown(f"<p style='font-size: 28px; font-weight: bold; color: {display_color}; margin-bottom: 10px;'>{display_text}</p>", unsafe_allow_html=True)

    with col2: # Price Level
        st.markdown(f"<h5 style='color: {price_level_base_color}; margin-bottom: 5px;'>价格水平 (相对区域)</h5>", unsafe_allow_html=True)
        if price_level_pred_label == "配置缺失":
            display_text = "特征配置缺失"
            display_color = config_missing_color
        elif insufficient_data_flags['price_level']:
            display_text = "数据不足，无法判断"
            display_color = insufficient_data_color
        elif price_level_pred_label == "预测失败" or price_level_pred_code == -99 :
             display_text = "预测失败"
             display_color = error_color
        elif price_level_pred_code == 1:
            display_text = price_level_pred_label
            display_color = "#E74C3C" # Red for higher
        elif price_level_pred_code == 0:
            display_text = price_level_pred_label
            display_color = "#2ECC71" # Green for not higher
        else:
            display_text = "未知状态" # Fallback
            display_color = insufficient_data_color
        st.markdown(f"<p style='font-size: 28px; font-weight: bold; color: {display_color}; margin-bottom: 10px;'>{display_text}</p>", unsafe_allow_html=True)

    with col3: # Unit Price Prediction
        st.markdown(f"<h5 style='color: {unit_price_color}; margin-bottom: 5px;'>均价预测</h5>", unsafe_allow_html=True)
        label_text = "预测单价 (元/㎡)" # Default label
        value_html = "" # Initialize value html

        if insufficient_data_flags['regression']:
            display_text = "数据不足，无法判断"
            display_color = insufficient_data_color
            value_html = f"<p style='font-size: 28px; font-weight: bold; color: {display_color}; margin-bottom: 10px;'>{display_text}</p>"
        elif unit_price_pred == -1.0:
            display_text = "预测失败"
            display_color = error_color
            value_html = f"<p style='font-size: 28px; font-weight: bold; color: {display_color}; margin-bottom: 10px;'>{display_text}</p>"
        else:
            display_text = f"{unit_price_pred:,.0f}" # Formatted number
            display_color = unit_price_color # Use title color for result
            # Mimic st.metric look with markdown
            value_html = f"<p style='font-size: 28px; font-weight: bold; color: {display_color}; margin-bottom: 10px;'>{display_text}</p>"

        # Display label first
        st.markdown(f"<p style='font-size: small; color: grey; margin-bottom: 0px;'>{label_text}</p>", unsafe_allow_html=True)
        # Display the value/status using markdown
        st.markdown(value_html, unsafe_allow_html=True)


    # --- Display errors or success message ---
    if error_messages:
        st.markdown("---")
        st.error("执行过程中遇到以下运行时错误：")
        for i, msg in enumerate(error_messages):
            # Be careful about displaying raw exception messages which might contain sensitive info
            # For production, log detailed errors and show generic messages to the user
            st.markdown(f"{i+1}. 分析时出现问题，请检查输入或联系管理员。") # Safer message
            print(f"Detailed Error {i+1}: {msg}") # Log the actual error
    elif not any(insufficient_data_flags.values()):
        st.success("✅ 分析预测完成！")
        st.markdown("---")
        st.info("💡 **提示:** 模型预测结果是基于历史数据和输入特征的估计，仅供参考。实际交易价格受市场供需、具体房况、谈判等多种因素影响。")
    elif any(insufficient_data_flags.values()):
        st.warning("⚠️ 部分预测因输入数据不足或配置缺失未能完成。请在侧边栏提供所有必需的特征信息（避免选择 '无'）。")
        st.markdown("---")
        st.info("💡 **提示:** 模型预测结果是基于历史数据和输入特征的估计，仅供参考。实际交易价格受市场供需、具体房况、谈判等多种因素影响。")


# --- 页脚信息 ---
st.sidebar.markdown("---")
st.sidebar.caption("模型信息: LightGBM & RandomForest")
st.sidebar.caption("数据来源: 安居客")
st.sidebar.caption("开发者: 凌欢")