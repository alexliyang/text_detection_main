 # ctpn_new

### 2018.3.24 更新

这里是我们将要重构代码的地方，每个人的模块代码将存放在这里，在向仓库提交前请确保在根目录中的.gitignore文件中中添加了 **ctpn_tensorflow/***，使得git忽略你在原实现下的代码改动。然后将代码push，必要时PR。之后这份README请大家共同维护，在较大的更新时请在这里添加你所做的工作的描述，减少大家理解代码的时间。

之后将在这里上传基本的项目结构，大家有任何的建议都可以在群里说，Happy Hacking😄

### 2018.3.25 更新
新的项目结构
```
ctpn_new--|___ctpn 
          |___data_process 读取训练数据并处理成需要的形式，返回roidb对象
          |___input_layer 网络结果第一层，给核心网络提供每轮batch所需的数据
          |___network 核心网络 基类base_network和子类ctpn_network
          |___lib 运行时所需要的某些cython扩展
          |___prepare 数据预处理脚本，结果直接输出到dataset
          |___run 训练数据和测试数据程序的入口
          |___config.yml 全局配置 唯一配置
**所有程序的运行都在ctpn_new目录下运行，书写时请注意包和模块的路径问题**
```