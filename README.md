# 简介
* 此仓库为c++实现yolo5的batch多线程推理, 大体改自https://github.com/leafqycc/rknn-cpp-Multithreading
* 主要函数为./main.cc ./postprocess.cc(line 197) ./include/rknnPool.hpp(line 168,194,203)

# 更新说明


# 使用说明
### 演示
  * 系统需安装有**OpenCV**
  * 运行build-linux_RK3588.sh
  * 可切换至root用户运行performance.sh定频提高性能和稳定性
  * 编译完成后进入install运行命令./rknn_yolov5_demo **模型所在路径** **视频所在路径/摄像头序号**

### 部署应用
  * 修改include/rknnPool.hpp中的rknn_lite类
  * 修改inclue/rknnPool.hpp中的rknnPool类的构造函数

# 多线程模型帧率测试
* 使用performance.sh进行CPU/NPU定频尽量减少误差

# 补充
* 异常处理尚未完善, 目前仅支持rk3588/rk3588s下的运行

# Acknowledgements
* https://github.com/rockchip-linux/rknpu2
* https://github.com/senlinzhan/dpool
* https://github.com/ultralytics/yolov5
* https://github.com/airockchip/rknn_model_zoo
* https://github.com/rockchip-linux/rknn-toolkit2
* https://github.com/leafqycc/rknn-cpp-Multithreading
