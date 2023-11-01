#ifndef _rknnPool_H
#define _rknnPool_H

#include <queue>
#include <vector>
#include <iostream>
#include "rga.h"
#include "im2d.h"
#include "RgaUtils.h"
#include "rknn_api.h"
#include "postprocess.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "ThreadPool.hpp"
using cv::Mat;
using std::queue;
using std::vector;

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static void dump_tensor_attr(rknn_tensor_attr* attr);

class rknn_lite
{
private:
    rknn_context rkModel;
    unsigned char *model_data;
    rknn_sdk_version version;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    rknn_input inputs[1];
    int ret;
    int channel = 3;
    int width = 0;
    int height = 0;
public:
    Mat ori_img;
    int interf();
    rknn_lite(char *dst, int n);
    ~rknn_lite();
};

rknn_lite::rknn_lite(char *model_name, int n)
{
    /* Create the neural network */
    printf("Loading mode...\n");
    int model_data_size = 0;
    // 读取模型文件数据
    model_data = load_model(model_name, &model_data_size);
    // 通过模型文件初始化rknn类
    ret = rknn_init(&rkModel, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }
    // 
    rknn_core_mask core_mask;
    if (n == 0)
        core_mask = RKNN_NPU_CORE_0;
    else if(n == 1)
        core_mask = RKNN_NPU_CORE_1;
    else
        core_mask = RKNN_NPU_CORE_2;
    int ret = rknn_set_core_mask(rkModel, core_mask);
    if (ret < 0)
    {
        printf("rknn_init core error ret=%d\n", ret);
        exit(-1);
    }

    // 初始化rknn类的版本
    ret = rknn_query(rkModel, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }

    // 获取模型的输入参数
    ret = rknn_query(rkModel, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }
    // 设置输入数组
    input_attrs = new rknn_tensor_attr[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(rkModel, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&input_attrs[i]);
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            exit(-1);
        }
    }

    // 设置输出数组
    output_attrs = new rknn_tensor_attr[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs) );
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(rkModel, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&output_attrs[i]);
    }

    // 设置输入参数
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    memset(inputs, 0, sizeof(inputs));
    for(int i = 0; i < 1; i++)
    {
        inputs[i].index = 0;
        inputs[i].type = RKNN_TENSOR_UINT8;
        inputs[i].size = width * height * channel *4;
        inputs[i].fmt = RKNN_TENSOR_NHWC;
        inputs[i].pass_through = 0;
    }
}

rknn_lite::~rknn_lite()
{
    ret = rknn_destroy(rkModel);
    delete[] input_attrs;
    delete[] output_attrs;
    if (model_data)
        free(model_data);
}

int rknn_lite::interf()
{
    // cv::cvtColor(ori_img, ori_img, cv::COLOR_YUV2BGR_NV12);
    cv::cvtColor(ori_img, ori_img, cv::COLOR_RGB2BGR);
    int img_width = ori_img.cols;
    int img_height = ori_img.rows;
    cv::Mat img = ori_img;
    // init rga context
    // rga是rk自家的绘图库,绘图效率高于OpenCV
    rga_buffer_t src;
    rga_buffer_t dst;
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));

    // 申明存储batch_size张图片的内存地址以及单张图片的内存地址，为了简化，就没有输入四张不一样的图像，如果是四张不一样的，resize_buf应该申请四份
    void *resize_buf = nullptr;
    int img_size = input_attrs[0].dims[1] * input_attrs[0].dims[2] * input_attrs[0].dims[3] * sizeof(uint8_t);
    unsigned char *in_data_batch = NULL;
    in_data_batch = (unsigned char *)malloc(BATCH_SIZE * img_size);
    if (!in_data_batch)
    {
        return -1;
    }
    // You may not need resize when src resulotion equals to dst resulotion
    // 如果输入图像不是指定格式
    if (img_width !=  width || img_height !=  height)
    {
        resize_buf = malloc( height *  width *  channel);
        memset(resize_buf, 0x00,  height *  width *  channel);

        src = wrapbuffer_virtualaddr((void *)img.data, img_width, img_height, RK_FORMAT_RGB_888);
        dst = wrapbuffer_virtualaddr((void *)resize_buf,  width,  height, RK_FORMAT_RGB_888);
        ret = imcheck(src, dst, src_rect, dst_rect);
        if (IM_STATUS_NOERROR !=  ret)
        {
            printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS) ret));
            exit(-1);
        }
        IM_STATUS STATUS = imresize(src, dst);
        cv::Mat resize_img(cv::Size( width,  height), CV_8UC3, resize_buf);
        // 将四张图片依次存入in_data_batch的连续内存中
        for (int i = 0; i < BATCH_SIZE; ++i)
        {
            unsigned char *in_data_ptr = in_data_batch + img_size * i;
            memcpy(in_data_ptr, resize_buf, img_size);
        }
            inputs[0].buf = in_data_batch;
        }
        else{
            // 将四张图片依次存入in_data_batch的连续内存中
            for (int i = 0; i < BATCH_SIZE; ++i)
            {
                unsigned char *in_data_ptr = in_data_batch + img_size * i;
                memcpy(in_data_ptr, img.data, img_size);
            }
            inputs[0].buf = in_data_batch;
            }

    // 设置rknn的输入数据
    rknn_inputs_set( rkModel,  io_num.n_input,  inputs);

    // 设置输出
    rknn_output outputs[ io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i <  io_num.n_output; i++)
        outputs[i].want_float = 0;
    // 调用npu进行推演
     ret = rknn_run( rkModel, NULL);
    // 获取npu的推演输出结果
     ret = rknn_outputs_get( rkModel,  io_num.n_output, outputs, NULL);
    // 总之就是绘图部分
    // post process
    // width是模型需要的输入宽度, img_width是图片的实际宽度
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    
    float scale_w = (float) width / img_width;
    float scale_h = (float) height / img_height;

    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    for (int i = 0; i <  io_num.n_output; ++i)
    {
        out_scales.push_back( output_attrs[i].scale);
        out_zps.push_back( output_attrs[i].zp);
    }
    rknn_output outputs2[ io_num.n_output];
    post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf,  height,  width,
                 box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

    // Draw Objects
    char text[256];
    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        if (!strncmp(det_result->name,"person",6))
        {
          int x1 = det_result->box.left;
          int y1 = det_result->box.top;
          std::vector<int> kuang = {det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom};
          
          sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
          rectangle(ori_img, cv::Point(x1, y1), cv::Point(det_result->box.right, det_result->box.bottom), cv::Scalar(0, 0, 255, 0), 3);
          putText(ori_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
        }
    }
     ret = rknn_outputs_release( rkModel,  io_num.n_output, outputs);
    if (resize_buf)
    {
        free(resize_buf);
        free(in_data_batch);
    }
    return 0;
}
static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

#endif
