> 程序使用说明：

软件平台： linux下python 3.4以上

目录需要cd到当前程序脚本所在目录

a) mnist_main.py  包含训练与测试的主函数，运行可直接显示训练过程；linux命令行执行命令： 
 
```bash
>>> python mnist_main.py
```

b) train.py  训练函数，用于训练并保存模型至当前目录的model文件夹下；执行命令:
```bash
>>> python train.py
```
c) test.py  测试函数，用于测试所有测试样本并打印准确率；执行命令:
```bash
>>> python test.py
```

d) inference.py  测试单个图像的预测结果。执行包括一个指定一张图像的路径参数，
   输出预测结果。执行命令如第0张图：
```
>>> python inference.py 'image/0.jpg'
```
