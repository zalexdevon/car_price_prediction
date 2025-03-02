import os
from box.exceptions import BoxValueError
import yaml
from regressor import (
    logger,
)  # sửa <classifier> thành tên source code project tương ứng
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import pickle
import pandas as pd
import plotly.express as px
from regressor.Mylib import myfuncs
import pandas as pd
import os
from regressor import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from regressor.Mylib import myfuncs
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import ParameterSampler
from sklearn import metrics
from sklearn.base import clone
from regressor.utils import common
from sklearn.linear_model import Ridge, Lasso, ElasticNet

BASE_MODELS = {
    "RID": Ridge(),
    "LAS": Lasso(),
    "ELA": ElasticNet(),
    "RF": RandomForestRegressor(random_state=42),
    "ET": ExtraTreesRegressor(random_state=42),
    "GB": GradientBoostingRegressor(random_state=42),
    "SGD": GradientBoostingRegressor(random_state=42),
    "XGB": XGBRegressor(random_state=42),
    "LGB": LGBMRegressor(verbose=-1, random_state=42),
}


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (Path): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: _description_
    """
    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        verbose (bool, optional): ignore if multiple dirs is to be created. Defaults to True.

    """

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, "wb") as f:
        f.write(imgdata)


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())


def create_param_grid_from_param_grid_for_transformer_and_for_model(
    param_transformer: dict, param_model: dict
):
    a = param_transformer
    b = param_model

    c_key = []
    c_value = []
    if a is not None:
        a_key = ["transform__" + item for item in a.keys()]
        c_key = a_key
        c_value = list(a.values())

    b_key = ["model__" + item for item in b.keys()]

    c_key += b_key
    c_value += list(b.values())

    return dict(zip(c_key, c_value))


def unindent_all_lines(content):
    content = content.strip("\n")
    lines = content.split("\n")
    lines = [item.strip() for item in lines]
    content_processed = "\n".join(lines)

    return content_processed


def insert_br_html_at_the_end_of_line(lines):
    return [f"{item} <br>" for item in lines]


def get_monitor_desc(param_grid_model_desc: dict):
    result = ""

    for key, value in param_grid_model_desc.items():
        key_processed = process_param_name(key)
        line = f"{key_processed}: {value}<br>"
        result += line

    return result


def process_param_name(name):
    if len(name) == 3:
        return name

    if len(name) > 3:
        return name[:3]

    return name + "_" * (3 - len(name))


# file_path = "artifacts/monitor_desc/monitor_desc.txt"
# print(get_monitor_desc(file_path))


def plot_monitor(monitor):
    target_val_value = 50  # giá trị mục tiêu của val
    min_y_value = 0  # giá trị nhỏ nhất của trục y
    max_val_value = 1000  # giá trị lớn nhất của val
    max_y_value = 1000 + 200  # giá trị lớn nhất của trục y
    dtick_y_value = 50  # thước chia trục y

    monitor_descs = [item[0] for item in monitor]
    train_scores = [item[1] for item in monitor]
    val_scores = [item[2] for item in monitor]

    for i in range(len(train_scores)):
        if train_scores[i] > max_val_value:
            train_scores[i] = max_val_value

        if val_scores[i] > max_val_value:
            val_scores[i] = max_val_value

    x_values = list(range(1, len(train_scores) + 1))

    df = pd.DataFrame({"x": x_values, "train": train_scores, "val": val_scores})
    df_long = df.melt(
        id_vars=["x"], value_vars=["train", "val"], var_name="Category", value_name="y"
    )

    fig = px.line(
        df_long,
        x="x",
        y="y",
        color="Category",
        markers=True,
        color_discrete_map={
            "train": "gray",
            "val": "blue",
        },
        hover_data={"x": False, "Category": False},
    )

    for i in range(len(x_values)):
        text = monitor_descs[i]

        fig.add_annotation(
            text=text,
            x=x_values[i],
            y=max_val_value,
            xref="x",
            yref="y",
            showarrow=False,
            font=dict(family="Consolas", size=7, color="red"),
            ax=0,
            align="left",
            yanchor="bottom",
        )

    fig.add_hline(y=max_val_value, line_dash="solid", line_color="black", line_width=2)

    fig.add_hline(
        y=target_val_value, line_dash="dash", line_color="green", line_width=2
    )

    fig.update_layout(
        autosize=False,
        width=100 * (len(x_values) + 2) + 30,
        height=600,
        margin=dict(l=30, r=10, t=10, b=0),
        xaxis=dict(
            title="",
            range=[
                0,
                len(x_values) + 2,
            ],
            tickmode="linear",
        ),
        yaxis=dict(
            title="",
            range=[min_y_value, max_y_value],
            dtick=dtick_y_value,
        ),
        showlegend=False,
    )

    html_path = "monitor_plot.html"
    fig.write_html(html_path, config={"displayModeBar": False})


def get_param_grid_model(param_grid_model: dict):
    values = param_grid_model.values()

    values = [myfuncs.get_range_for_param(str(item)) for item in values]

    return dict(zip(list(param_grid_model.keys()), values))


def get_base_model(model_name: str):
    """Get the Model object from model_name <br>


    Args:
        model_name (str): format = model_name_real_blabla

    """

    model_name_real = model_name.split("_")[0]

    return BASE_MODELS[model_name_real]


def sub_param_for_yaml_file(src_path: str, des_path: str, replace_dict: dict):
    """Substitue params in src_path and save in des_path

    Args:
        replace_dict (dict): key: item needed to replace, value: item to replace
        VD:
        ```python
        replace_dict = {
            "${P}": data_transformation,
            "${T}": model_name,
            "${E}": evaluation,
        }

        ```
    """

    with open(src_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    config_str = yaml.dump(config_data, default_flow_style=False)

    for key, value in replace_dict.items():
        config_str = config_str.replace(key, value)

    with open(des_path, "w", encoding="utf-8") as file:
        file.write(config_str)

    print(f"Đã thay thế các tham số trong {src_path} lưu vào {des_path}")


def get_real_column_name(column):
    """After using ColumnTransformer, the column name has format = bla__Age, so only take Age"""

    start_index = column.find("__") + 2
    column = column[start_index:]
    return column


def get_real_column_name_from_get_feature_names_out(columns):
    """Take the exact name from the list retrieved by method get_feature_names_out() of ColumnTransformer"""

    return [get_real_column_name(item) for item in columns]
