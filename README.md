**说明文档**<br>
**如何运行**<br>
app里自行设置本地IP，train_ui运行后UI界面内增量训练内容到base AI model，自行准备xxx.jsonl格式的数据文件，model_ui加载对应model路径，启动后与app交互。<br>
**环境**
需要安装的PIP：<br>
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129<br>
pip install transformers datasets peft accelerate<br>
其余具体看：requirements.txt<br>
