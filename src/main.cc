// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <sys/time.h>
#include <thread>
#include <queue>
#include <vector>
#define _BASETSD_H

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rknnPool.hpp"
#include "ThreadPool.hpp"
using std::queue;
using std::time;
using std::time_t;
using std::vector;
int main(int argc, char **argv)
{
  char *model_name = NULL;
  model_name = (char *)argv[1]; // 参数二，模型所在路径
  printf("模型名称:\t%s\n", model_name);

  cv::VideoCapture capture1;
  capture1.open("/dev/video1", cv::CAP_V4L2);
  capture1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  capture1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  cv::namedWindow("Camera FPS 1");

  cv::VideoCapture capture2;
  capture2.open("/dev/video3", cv::CAP_V4L2);
  capture2.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  capture2.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  cv::namedWindow("Camera FPS 2");

  cv::VideoCapture capture3;
  capture3.open("/dev/video5", cv::CAP_V4L2);
  capture3.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  capture3.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  cv::namedWindow("Camera FPS 3");

  cv::VideoCapture capture4;
  capture4.open("/dev/video7", cv::CAP_V4L2);
  capture4.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  capture4.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  cv::namedWindow("Camera FPS 4");
  // 设置线程数
  int n = 8, frames = 0;
  printf("线程数:\t%d\n", n);
  // 类似于多个rk模型的集合?
  vector<rknn_lite *> rkpool;
  // 线程池
  dpool::ThreadPool pool(n);
  // 线程队列
  queue<std::future<int>> futs;

  //初始化
  for (int i = 0; i < n; i++)
  {
    rknn_lite *ptr = new rknn_lite(model_name, i % 3);
    rkpool.push_back(ptr);
    capture1 >> ptr->ori_img;
    capture2 >> ptr->ori_img2;
    capture3 >> ptr->ori_img3;
    capture4 >> ptr->ori_img4;
    futs.push(pool.submit(&rknn_lite::interf, &(*ptr)));
  }

  struct timeval time;
  gettimeofday(&time, nullptr);
  auto initTime = time.tv_sec * 1000 + time.tv_usec / 1000;

  gettimeofday(&time, nullptr);
  long tmpTime, lopTime = time.tv_sec * 1000 + time.tv_usec / 1000;

  while (capture1.isOpened() && capture2.isOpened() && capture3.isOpened() && capture4.isOpened())
  {
    if (futs.front().get() != 0)
      break;
    futs.pop();
    cv::imshow("Camera FPS 1", rkpool[frames % n]->ori_img);
    cv::imshow("Camera FPS 2", rkpool[frames % n]->ori_img2);
    cv::imshow("Camera FPS 3", rkpool[frames % n]->ori_img3);
    cv::imshow("Camera FPS 4", rkpool[frames % n]->ori_img4);
    if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出
      break;

    if(!capture1.read(rkpool[frames % n]->ori_img) || !capture2.read(rkpool[frames % n]->ori_img2) 
    || !capture3.read(rkpool[frames % n]->ori_img3) || !capture4.read(rkpool[frames % n]->ori_img4))
      break;

    futs.push(pool.submit(&rknn_lite::interf, &(*rkpool[frames++ % n])));

    if(frames % 60 == 0){
        gettimeofday(&time, nullptr);
        tmpTime = time.tv_sec * 1000 + time.tv_usec / 1000;
        printf("60帧平均帧率:\t%f帧\n", 60000.0 / (float)(tmpTime - lopTime));
        lopTime = tmpTime;
    }
  }

  gettimeofday(&time, nullptr);
  printf("\n平均帧率:\t%f帧\n", float(frames) / (float)(time.tv_sec * 1000 + time.tv_usec / 1000 - initTime + 0.0001) * 1000.0);

  // 释放剩下的资源
  while (!futs.empty())
  {
    if (futs.front().get())
      break;
    futs.pop();
  }
  for (int i = 0; i < n; i++)
    delete rkpool[i];
  capture1.release();
  capture2.release();
  capture3.release();
  capture4.release();
  cv::destroyAllWindows();
  return 0;
}
