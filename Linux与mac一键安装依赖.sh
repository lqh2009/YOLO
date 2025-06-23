#!/bin/bash

echo "开始安装依赖..."

# 检查Python版本
if command -v python3 &> /dev/null
then
    PYTHON_CMD=python3
    PIP_CMD=pip3
else
    echo "未检测到Python 3,请先安装Python 3.7或更高版本"
    exit 1
fi

# 获取Python版本
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "检测到Python版本: $PYTHON_VERSION"

# 检查Python版本是否至少为3.7
if [ "$(printf '%s\n' "3.7" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.7" ]; then
    echo "Python版本必须至少为3.7。请升级您的Python版本。"
    exit 1
fi

# 检查是否安装了pip
if ! command -v $PIP_CMD &> /dev/null
then
    echo "未检测到pip,正在安装pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $PYTHON_CMD get-pip.py
    rm get-pip.py
fi

# 安装依赖
echo "正在安装项目依赖..."
$PIP_CMD install -r requirements.txt

# 检查安装是否成功
if [ $? -eq 0 ]; then
    echo "依赖安装成功!"
    echo "您现在可以运行 '$PYTHON_CMD object_detection_app.py' 来启动应用程序"
else
    echo "依赖安装失败,请检查错误信息并重试"
fi