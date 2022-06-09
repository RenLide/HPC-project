# HPC-project
通过Makefile对代码进行编译。

使用make explicit或make implicit即可生成相应对可执行文件。

使用make clean可以清除所有生成的中间文件和可执行文件。

使用explicit.LSF提交TaiYi脚本，通过“-r”和“-np”等参数对程序进行相应控制。

程序执行完成或者强制中断后，均可生成对应对HDF5文件，可用于分析计算结果和代码性能以及用于继续计算。

完整代码及使用说明文档的github仓库链接为https://github.com/RenLide/HPC-project.git。

TaiYi服务器中项目代码文件夹位置为/work/mae-renld/HPC/project。
