Jax版本必须和cuda toolkit和cudnn版本符合，且cudnn版本需和pytorch需要的版本符合。新的jaxlib已不再支持cuda11，需使用cuda12以上系列，且仅支持cudnn>=9.5。
**处于未知原因，jaxlib0.4.29+cuda12.cudnn91会在运行时出现段错误，无法使用**
现有可运行版本使用cuda toolkit 12.6 + cudnn9.10.2.21 + pytorch2.8.0 + jax0.6.1[cuda12]
* 安装后系统可换回cuda toolkit 12.1继续运行，但安装时版本必须为12.6（或更高）！

直接安装[tapnet](https://github.com/google-deepmind/tapnet)，之后请将hugginface下载的causal_bootstapir_checkpoint.pt放置与本目录下的checkpoints目录，就可以直接在此repo使用python运行`point_track_ros.py`文件

如在conda环境中运行，请确保conda中的libstdc++和ROS2使用版本一致或更新，可使用`conda install -c conda-forge "libstdcxx-ng>=12" "libgcc-ng>=12"`更新