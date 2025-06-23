# 物体检测应用

这是一个基于Python的物体检测应用程序。

## 文件说明

- `object_detection_app.py`: 主应用程序文件
- `object_detection.py`: 物体检测核心功能
- `app_parameters.json`: 应用程序参数配置文件
- `requirements.txt`: 项目依赖列表
- `install_dependencies.bat`: Windows系统安装依赖脚本
- `install_dependencies.sh`: Linux/Mac系统安装依赖脚本
- `一键安装依赖.bat`: Windows系统一键安装依赖脚本
- `一键启动.bat`: Windows系统一键启动应用脚本
- `object_detection_app.spec`: PyInstaller打包配置文件
- `app.log`: 应用程序日志文件

## 安装说明

### Windows用户

1. 双击运行 `一键安装依赖.bat` 文件安装所需依赖。
2. 安装完成后,双击 `一键启动.bat` 文件启动应用程序。

### Linux/Mac用户

1. 打开终端,进入项目目录。
2. 运行以下命令给安装脚本添加执行权限:
   ```sh
   chmod +x Linux与mac一键安装依赖.sh
   ```
3. 运行安装脚本:
   ```sh
   ./Linux与mac一键安装依赖.sh
   ```
4. 安装完成后,脚本会提示您如何启动应用程序。通常是运行:
   ```sh
   python3 object_detection_app.py
   ```

## 使用说明

1. 启动应用程序后,会打开一个图形界面。
2. 点击"选择图片"按钮,选择要进行物体检测的图片。
3. 程序会自动进行物体检测,并在图片上标注检测到的物体。
4. 检测结果会显示在界面上。

## 注意事项

- 确保您的系统已安装Python 3.7或更高版本。
- 脚本会自动检测并使用系统上可用的Python 3版本。
- 如遇到任何问题,请查看 `app.log` 文件了解详细错误信息。
- 可以通过修改 `app_parameters.json` 文件来调整应用程序的参数设置。

官方GitHub地址：https://github.com/ultralytics/ultralytics
应用制作：@文抑青年--B站主页：https://space.bilibili.com/259012968
