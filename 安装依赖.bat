@echo off
chcp 65001 >nul
echo 正在安装依赖...

REM 设置Python解释器路径
set PYTHON_PATH=.\py311\python.exe

REM 检查Python解释器是否存在
if not exist %PYTHON_PATH% (
    echo 错误: 未找到Python解释器,请确保py311文件夹中包含python.exe
    pause
    exit /b 1
)

REM 使用指定的Python解释器安装requirements.txt中的依赖
%PYTHON_PATH% -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo 安装依赖时出错,请检查错误信息并重试。
    pause
    exit /b 1
)

echo 依赖安装完成!

REM 检测CUDA是否可用
echo 正在检测CUDA是否可用...
%PYTHON_PATH% -c "import torch; print('CUDA是否可用:', torch.cuda.is_available()); print('可用的CUDA设备数量:', torch.cuda.device_count()); print('当前CUDA设备:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"

pause