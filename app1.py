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
# ***** 注意：这里的特征列表必须与训练回归模型和Scaler时使用的特征完全一致，包括顺序 *****
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
                 # ***** 关键：检查 Scaler 是否与代码指定的特征匹配 *****
                 if hasattr(resources['scaler'], 'n_features_in_') and resources['scaler'].n_features_in_ != len(REQUIRED_REGRESSION_FEATURES):
                      error_msg = f"严重错误: Scaler (来自 {os.path.basename(SCALER_PATH)}) 期望 {resources['scaler'].n_features_in_} 个特征, 但代码指定了 {len(REQUIRED_REGRESSION_FEATURES)} 个回归特征 ({REQUIRED_REGRESSION_FEATURES})。请确保 Scaler 与指定的特征列表一致。"
                      print(error_msg)
                      # Returning None simulates a load failure specifically for this mismatch
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
             # Provide specific guidance for scaler mismatch
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
feature_names_loaded = resources.get('feature_names', {}) # Use .get for safety
market_model = resources['market_model']
price_level_model = resources['price_level_model']
regression_model = resources['regression_model']
scaler = resources['scaler']

# 检查核心映射和特征列表是否存在且为预期类型
required_mappings = ['方位', '楼层', '所属区域', '房龄', '市场类别', '是否高于区域均价']
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
selectbox_labels_map = {} # To map internal key back to display label if needed
all_select_valid = True # Track if all dropdowns load options correctly

st.sidebar.subheader("选择项特征")
# 封装下拉框创建逻辑
def create_selectbox(internal_key, help_text, key_suffix):
    global all_select_valid # Allow modification of the global flag
    label = feature_to_label.get(internal_key, internal_key) # Get label from map
    try:
        options_map = mappings[internal_key]
        # Generate display map including the 'None' option
        display_map = format_mapping_options_for_selectbox(options_map)

        if not display_map or len(display_map) <= 1: # Should have 'None' + at least one other
             st.sidebar.warning(f"'{label}' 缺少有效选项 (除了'无')。请检查 {os.path.basename(MAPPINGS_PATH)} 中的 '{internal_key}'。")
             if not display_map:
                 display_map = {None: "无 (加载失败)"} # Provide a fallback

        options_codes = list(display_map.keys()) # Keys include None and the actual codes

        # Determine default index - try to avoid 'None' as default
        default_index = 0 # Default to '无' if no other options or logic applies
        if len(options_codes) > 1:
            common_defaults = {'楼层': 1, '房龄': 2} # Example: Default to middle floor, 6-10 years
            target_default_code = common_defaults.get(internal_key)

            if target_default_code is not None and target_default_code in options_codes:
                try:
                    default_index = options_codes.index(target_default_code)
                except ValueError:
                    print(f"Warning: Default code {target_default_code} for {internal_key} not found in options {options_codes}. Defaulting.")
                    default_index = 1 # Default to the first non-'None' option
            # Smarter default index based on number of options
            elif len(options_codes) > 3: # More options, pick near middle
                 default_index = (len(options_codes) -1) // 2 + 1 # index after None
            elif len(options_codes) >= 2: # Only one option besides 'None'
                default_index = 1 # Pick the first actual option

        selected_value = st.sidebar.selectbox(
            label,
            options=options_codes,
            index=default_index,
            format_func=lambda x: display_map.get(x, f"未知 ({x})"),
            key=f"{key_suffix}_select",
            help=help_text
        )
        selectbox_labels_map[internal_key] = label # Store mapping key to label
        return selected_value
    except Exception as e:
        st.sidebar.error(f"加载 '{label}' 选项时出错: {e}")
        print(f"Error details for loading {label}: {e}") # Print detailed error to console
        all_select_valid = False
        return None

# Create selectboxes using the function
selectbox_inputs['方位'] = create_selectbox('方位', "选择房屋的主要朝向。选择 '无' 如果不确定或不适用。", "orientation")
selectbox_inputs['楼层'] = create_selectbox('楼层', "选择房屋所在楼层的大致位置。选择 '无' 如果不确定或不适用。", "floor_level")
selectbox_inputs['所属区域'] = create_selectbox('所属区域', "选择房产所在的行政区域或板块。选择 '无' 如果不确定或不适用。", "district")
selectbox_inputs['房龄'] = create_selectbox('房龄', "选择房屋的建造年限范围。选择 '无' 如果不确定或不适用。", "age")

# --- ***** 修改：数值输入控件，增加 "无" 选项 ***** ---
st.sidebar.subheader("数值项特征")
numeric_inputs = {}
numeric_input_states = {} # To store the state ('输入数值' or '无')

# Define default numeric values (used only if '输入数值' is selected)
default_numeric_values = {
    '总价(万)': 120.0,
    '面积(㎡)': 95.0,
    '建造时间': 2015,
    '楼层数': 18,
    '室': 3,
    '厅': 2,
    '卫': 1
}

# Define numeric input parameters
numeric_params = {
    '总价(万)': {'min_value': 0.0, 'max_value': 10000.0, 'step': 5.0, 'format': "%.1f", 'help': "输入房产的总价，单位万元。留空或选择 '无' 表示不适用。"},
    '面积(㎡)': {'min_value': 1.0, 'max_value': 2000.0, 'step': 1.0, 'format': "%.1f", 'help': "输入房产的建筑面积，单位平方米。留空或选择 '无' 表示不适用。"},
    '建造时间': {'min_value': 1900, 'max_value': 2025, 'step': 1, 'format': "%d", 'help': "输入房屋的建造年份。留空或选择 '无' 表示不适用。"},
    '楼层数': {'min_value': 1, 'max_value': 100, 'step': 1, 'format': "%d", 'help': "输入楼栋的总楼层数。留空或选择 '无' 表示不适用。"},
    '室': {'min_value': 0, 'max_value': 20, 'step': 1, 'format': "%d", 'help': "输入卧室数量。留空或选择 '无' 表示不适用。"},
    '厅': {'min_value': 0, 'max_value': 10, 'step': 1, 'format': "%d", 'help': "输入客厅/餐厅数量。留空或选择 '无' 表示不适用。"},
    '卫': {'min_value': 0, 'max_value': 10, 'step': 1, 'format': "%d", 'help': "输入卫生间数量。留空或选择 '无' 表示不适用。"}
}

# Create combined input widgets for numeric features
for key, label in feature_to_label.items():
    if key in numeric_params: # Check if it's a numeric feature we handle
        param = numeric_params[key]
        default_val = default_numeric_values[key]
        key_suffix = key.replace('(','').replace(')','').replace('㎡','') # Create a simple key suffix

        # Selector to choose between entering a value or specifying 'None'
        numeric_input_states[key] = st.sidebar.selectbox(
            label, # Use the label defined in feature_to_label
            options=["输入数值", "无 (不适用)"],
            index=0,  # Default to "输入数值"
            key=f"{key_suffix}_selector",
            help=param['help']
        )

        # Conditionally display the number input
        if numeric_input_states[key] == "输入数值":
            numeric_inputs[key] = st.sidebar.number_input(
                f"输入 {label}", # Slightly modify label for clarity
                min_value=param['min_value'],
                max_value=param['max_value'],
                value=default_val,
                step=param['step'],
                format=param['format'],
                key=f"{key_suffix}_value",
                label_visibility="collapsed" # Hide label as it's implied by selector
            )
        else:
            # If "无 (不适用)" is selected, store None for this feature
            numeric_inputs[key] = None
            # Optionally, display a disabled placeholder or nothing
            # st.sidebar.text_input(f"{label}", value="不适用", disabled=True, key=f"{key_suffix}_value_disabled")


# --- 预测触发按钮 ---
st.sidebar.markdown("---")
predict_button_disabled = not all_select_valid # Can still predict if some numeric are None
predict_button_help = "点击这里根据输入的特征进行预测分析" if all_select_valid else "部分下拉框选项加载失败，无法进行预测。请检查资源文件或错误信息。"

if st.sidebar.button("🚀 开始分析预测", type="primary", use_container_width=True, help=predict_button_help, disabled=predict_button_disabled):

    # --- ***** 修改：整合输入时处理 None 值 ***** ---
    # Start with selectbox inputs
    all_inputs = {**selectbox_inputs}
    # Add numeric inputs, respecting the 'None' state from the selectors
    for key, state in numeric_input_states.items():
        if state == "无 (不适用)":
            all_inputs[key] = None # Store None if '无' was selected
        else:
            # Retrieve the value from the corresponding number_input widget
            key_suffix = key.replace('(','').replace(')','').replace('㎡','')
            all_inputs[key] = st.session_state[f"{key_suffix}_value"] # Get value using its key

    print("Combined inputs for prediction:", all_inputs) # Debugging output

    # --- Initialize result variables ---
    market_pred_label = "等待计算..."
    price_level_pred_label = "等待计算..."
    price_level_pred_code = -99 # Use a distinct code for 'not predicted' or 'error'
    unit_price_pred = -1.0 # Use -1.0 for 'not predicted' or 'error'
    error_messages = []
    insufficient_data_flags = {'market': False, 'price_level': False, 'regression': False}

    # --- ***** 修改：Helper Function to Check Input Sufficiency (Handles None) ***** ---
    def check_sufficiency(model_key, required_feature_list):
        """Checks if all required features for a model have valid (non-None) values."""
        missing_for_model = []
        for feat in required_feature_list:
            # Check if the feature exists in the combined inputs and if its value is None
            if feat not in all_inputs:
                 # This is a critical configuration error - required feature not in UI
                 print(f"严重警告: 模型 '{model_key}' 需要的特征 '{feat}' 在UI输入中未定义!")
                 missing_for_model.append(f"{feature_to_label.get(feat, feat)} (UI未定义)")
            elif all_inputs[feat] is None:
                # Feature exists, but its value is None (user selected '无' or it failed loading)
                missing_for_model.append(feature_to_label.get(feat, feat)) # Use display label

        if missing_for_model:
            print(f"模型 '{model_key}' 数据不足，缺少: {missing_for_model}")
            insufficient_data_flags[model_key] = True
            return False # Data is insufficient
        return True # Data is sufficient


    # --- 1. 市场细分预测 ---
    market_features_needed = feature_names_loaded.get('market', [])
    if not market_features_needed:
         st.warning("警告: 未在 feature_names.joblib 中找到 'market' 模型的特征列表，无法进行市场细分预测。")
         insufficient_data_flags['market'] = True # Mark as insufficient
         market_pred_label = "配置缺失" # Specific status
    elif check_sufficiency('market', market_features_needed):
        try:
            # Filter only non-None inputs needed for this model
            input_data_market = {feat: all_inputs[feat] for feat in market_features_needed}
            input_df_market = pd.DataFrame([input_data_market])[market_features_needed] # Ensure order
            market_pred_code = market_model.predict(input_df_market)[0]
            market_output_map_raw = mappings.get('市场类别', {})
            # Ensure prediction code is treated as the correct type for map lookup
            market_pred_key = int(market_pred_code) if isinstance(market_pred_code, (int, np.integer, float)) else str(market_pred_code)
            market_pred_label = market_output_map_raw.get(market_pred_key, f"未知编码 ({market_pred_key})")
        except Exception as e:
            msg = f"市场细分模型预测时发生错误: {e}"
            print(msg)
            error_messages.append(msg)
            market_pred_label = "预测失败" # Indicate runtime error
    else:
        # check_sufficiency returned False
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
            input_df_price_level = pd.DataFrame([input_data_price_level])[price_level_features_needed] # Ensure order
            price_level_pred_code_raw = price_level_model.predict(input_df_price_level)[0]
            price_level_output_map_raw = mappings.get('是否高于区域均价', {})

            # Determine the key type for the map and store the code
            if isinstance(price_level_pred_code_raw, (int, np.integer, float)):
                 price_level_pred_key = int(price_level_pred_code_raw)
                 price_level_pred_code = price_level_pred_key # Store 0 or 1
            else:
                 price_level_pred_key = str(price_level_pred_code_raw)
                 price_level_pred_code = -99 # Error/Unknown code

            price_level_pred_label = price_level_output_map_raw.get(price_level_pred_key, f"未知编码 ({price_level_pred_key})")

        except Exception as e:
            msg = f"价格水平模型预测时发生错误: {e}"
            print(msg)
            error_messages.append(msg)
            price_level_pred_label = "预测失败"
            price_level_pred_code = -99 # Ensure error code
    else:
        # check_sufficiency returned False
        price_level_pred_label = "数据不足"
        price_level_pred_code = -99 # Indicate insufficient data state if needed

    # --- 3. 均价预测 (回归) ---
    # ***** 使用代码中定义的 REQUIRED_REGRESSION_FEATURES *****
    regression_features_needed = REQUIRED_REGRESSION_FEATURES
    print(f"执行均价预测，使用特征: {regression_features_needed}") # Log features being used

    if check_sufficiency('regression', regression_features_needed):
        try:
            # Prepare data using the REQUIRED_REGRESSION_FEATURES list
            input_data_reg = {feat: all_inputs[feat] for feat in regression_features_needed}
            # Create DataFrame with columns in the exact order of REQUIRED_REGRESSION_FEATURES
            input_df_reg = pd.DataFrame([input_data_reg])[regression_features_needed]
            print("均价预测模型输入 DataFrame (原始):", input_df_reg)

            # Apply scaler - MUST match the features and order used for training
            try:
                 input_df_reg_scaled = scaler.transform(input_df_reg)
                 print("均价预测模型输入 DataFrame (缩放后):", input_df_reg_scaled)
            except ValueError as ve:
                 print(f"缩放器错误: {ve}")
                 # Check if the error message is about feature names/number mismatch
                 if "feature_names mismatch" in str(ve) or "number of features" in str(ve) or "X has" in str(ve):
                      n_scaler_feats = getattr(scaler, 'n_features_in_', '未知数量')
                      error_detail = f"缩放器期望 {n_scaler_feats} 个特征, 但提供了 {input_df_reg.shape[1]} 个 ({regression_features_needed})。请确保 'regression_scaler.joblib' 使用相同的特征和顺序进行训练。"
                      raise ValueError(f"缩放器与提供的特征不匹配。{error_detail}") from ve
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
    else:
        # check_sufficiency returned False
        unit_price_pred = -1.0 # Mark as insufficient data / error state
        # Ensure the flag is set correctly if check_sufficiency failed
        insufficient_data_flags['regression'] = True

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

    # Helper to create consistent result display block
    def display_result(title, title_color, result_text, result_color):
        st.markdown(f"<h5 style='color: {title_color}; margin-bottom: 5px; text-align: center;'>{title}</h5>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 28px; font-weight: bold; color: {result_color}; margin-bottom: 10px; text-align: center;'>{result_text}</p>", unsafe_allow_html=True)


    with col1: # Market Segment
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
            display_color = market_color # Use title color for result
        display_result(title, market_color, display_text, display_color)


    with col2: # Price Level
        title = "价格水平 (相对区域)"
        if price_level_pred_label == "配置缺失":
            display_text = "特征配置缺失"
            display_color = config_missing_color
        elif insufficient_data_flags['price_level'] or price_level_pred_label == "数据不足":
            display_text = "数据不足"
            display_color = insufficient_data_color
        elif price_level_pred_label == "预测失败" or price_level_pred_code == -99:
             # Treat -99 code (error or initial state) same as explicit failure label
             display_text = "预测失败/未知" # Combined state
             display_color = error_color
        elif price_level_pred_code == 1: # Higher than average
            display_text = price_level_pred_label
            display_color = "#E74C3C" # Red for higher
        elif price_level_pred_code == 0: # Not higher than average
            display_text = price_level_pred_label
            display_color = "#2ECC71" # Green for not higher
        else: # Should not happen with current logic, but include a fallback
            display_text = "未知状态"
            display_color = insufficient_data_color
        display_result(title, price_level_base_color, display_text, display_color)


    with col3: # Unit Price Prediction
        title = "均价预测"
        # ***** 修改：直接在结果中添加单位，移除下方小字标签 *****
        if insufficient_data_flags['regression']:
            display_text = "数据不足"
            display_color = insufficient_data_color
        elif unit_price_pred == -1.0: # Covers prediction errors and initial 'insufficient' state if check failed
            display_text = "预测失败/数据不足"
            display_color = error_color # Use error color for this combined state
        else:
            # Format successfully predicted price WITH unit
            display_text = f"{unit_price_pred:,.0f} 元/㎡"
            display_color = unit_price_color # Use title color for result
        display_result(title, unit_price_color, display_text, display_color)


    # --- Display errors or success/warning message ---
    if error_messages:
        st.markdown("---")
        st.error("执行过程中遇到以下运行时错误：")
        for i, msg in enumerate(error_messages):
            # Show a safer message to the user, log details
            st.markdown(f"{i+1}. 分析时出现问题，请检查输入或联系管理员。")
            print(f"Detailed Error {i+1}: {msg}") # Log the actual error for debugging
            if "缩放器与提供的特征不匹配" in msg: # Provide specific guidance for scaler issues
                 st.warning(f"💡 **提示 (错误 {i+1}):** 检测到均价预测所需的特征与加载的缩放器 (`{os.path.basename(SCALER_PATH)}`) 不匹配。请确保代码中定义的特征列表 (`REQUIRED_REGRESSION_FEATURES`) 与用于训练和保存缩放器的特征列表完全一致（包括顺序）。")

    # Check flags AFTER predictions to give accurate status summary
    has_insufficient_data = any(insufficient_data_flags.values())
    has_errors = bool(error_messages) or market_pred_label == "预测失败" or price_level_pred_label == "预测失败" or unit_price_pred == -1.0

    # Display summary message based on outcomes
    if not has_insufficient_data and not has_errors and market_pred_label != "配置缺失" and price_level_pred_label != "配置缺失":
        st.success("✅ 所有分析预测完成！")
        st.markdown("---")
        st.info("💡 **提示:** 模型预测结果是基于历史数据和输入特征的估计，仅供参考。实际交易价格受市场供需、具体房况、谈判等多种因素影响。")
    elif has_insufficient_data or market_pred_label == "配置缺失" or price_level_pred_label == "配置缺失":
        st.warning("⚠️ 部分预测因输入数据不足或配置缺失未能完成。请在侧边栏提供所有必需的特征信息（避免选择 '无 (不适用)'）。")
        st.markdown("---")
        st.info("💡 **提示:** 模型预测结果是基于历史数据和输入特征的估计，仅供参考。实际交易价格受市场供需、具体房况、谈判等多种因素影响。")
    elif has_errors and not error_messages: # Handle cases where prediction failed without throwing exception shown above
         st.error("❌ 部分预测失败。请检查输入或联系管理员。")
         st.markdown("---")
         st.info("💡 **提示:** 模型预测结果是基于历史数据和输入特征的估计，仅供参考。实际交易价格受市场供需、具体房况、谈判等多种因素影响。")
    # If error_messages is not empty, the error block above already displayed.


# --- 页脚信息 ---
st.sidebar.markdown("---")
st.sidebar.caption("模型信息: LightGBM & RandomForest")
st.sidebar.caption("数据来源: 安居客") # Clarify data source if it's example
st.sidebar.caption("开发者: 凌欢")