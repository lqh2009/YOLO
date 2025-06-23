@echo off
chcp 65001 >nul
echo 正在启动对象检测应用程序...

REM 设置Python解释器路径
set PYTHON_PATH=.\py311\python.exe

REM 检查Python解释器是否存在
if not exist %PYTHON_PATH% (
    echo 错误: 未找到Python解释器,请确保py311文件夹中包含python.exe
    pause
    exit /b 1
)

REM 使用指定的Python解释器启动应用程序
%PYTHON_PATH% app.py

if %errorlevel% neq 0 (
    echo 启动应用程序时出错,请检查错误信息并重试。
    pause
    exit /b 1
)

pause