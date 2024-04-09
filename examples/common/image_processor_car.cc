#include "image_processor_car.h"
#include "../liveview/ffmpeg_stream_decoder.h"

#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include "fastdeploy/vision.h"
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <string>
#include <queue>
#include <vector>
#include "logger.h"
#include "util_misc.h"

extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
}


using namespace cv;
using namespace dnn;
using namespace std;
using namespace edge_sdk;

namespace edge_app {


    std::string model_file = "/home/flision/cwudis/Edge-SDK-master/examples/common/data/car/inference.pdmodel";
    std::string params_file = "/home/flision/cwudis/Edge-SDK-master/examples/common/data/car/inference.pdiparams";
    std::string config_file = "/home/flision/cwudis/Edge-SDK-master/examples/common/data/car/inference.yml";

    std::map<int, std::string> label_map2 = {
    {2, "car"},
    {3, "bus"},
    {4, "truck"}
    };

    int32_t ImageProcessorCar::Init() {
        //   if (GetCurrentFileDirPath(__FILE__, sizeof(cur_file_dir_path_),
         //     cur_file_dir_path_) != 0) {
         //     ERROR("get path failed");
        //      return -1;
         //  }
                   // cur_file_dir_path_ = "/home/flision/cwudis/Edge-SDK-master/examples/common"
         //  snprintf(model_file, kFilePathSizeMax,"%sdata/car/inference.pdmodel",cur_file_dir_path_);
         //  snprintf(params_file, kFilePathSizeMax,"%sdata/car/inference.pdiparams",cur_file_dir_path_);
        //   snprintf(config_file, kFilePathSizeMax,"%sdata/car/inference.yml",cur_file_dir_path_);

        DEBUG("%s, %s, %s", model_file, params_file, config_file);
        fastdeploy::RuntimeOption option;
        model = fastdeploy::vision::detection::PaddleDetectionModel(model_file, params_file, config_file);

        // cv::namedWindow(show_name_.c_str(), cv::WINDOW_NORMAL);
        // cv::resizeWindow(show_name_.c_str(), 960, 540);
        // cv::moveWindow(show_name_.c_str(), rand() & 0xFF, rand() & 0xff);
        return 0;
    }

    AVFrame* CVMatToAVFrame(const Mat& inputMat)
    {
        // 获取图像信息
        AVPixelFormat dstFormat = AV_PIX_FMT_YUV420P;
        int width = inputMat.cols;
        int height = inputMat.rows;

        // 创建 AVFrame 并设置基本参数
        AVFrame* frame = av_frame_alloc();
        if (!frame)
        {
            std::cerr << "Error allocating AVFrame" << std::endl;
            return nullptr;
        }

        frame->width = width;
        frame->height = height;
        frame->format = dstFormat;

        // 分配 AVFrame 内部缓冲区
        int ret = av_frame_get_buffer(frame, 32);
        if (ret < 0)
        {
            std::cerr << "Error allocating frame buffer" << std::endl;
            av_frame_free(&frame);
            return nullptr;
        }

        // 将图像转换为 YUV420 格式
        Mat yuvMat;
        cv::cvtColor(inputMat, yuvMat, cv::COLOR_BGR2YUV_I420);

        // 将图像数据拷贝到 AVFrame 中
        uint8_t* data = yuvMat.data;
        int frame_size = width * height;
        memcpy(frame->data[0], data, frame_size);                        // Y plane
        memcpy(frame->data[1], data + frame_size, frame_size / 4);       // U plane
        memcpy(frame->data[2], data + frame_size * 5 / 4, frame_size / 4); // V plane

        return frame;
    }

    void ImageProcessorCar::Process(const std::shared_ptr<Image> image) {

        auto draw_pred = [](string label_name, float score, float xmin, float ymin, float xmax, float ymax, Mat& frame) {


            std::string text = label_name + ", " + std::to_string(score);
            putText(frame, text, Point(xmin, ymin - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
            rectangle(frame, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 0), 2);  // 绘制检测框

            };

        auto detect = [&](cv::Mat& frame) {
            fastdeploy::vision::DetectionResult result;
            model.Predict(frame, &result);
            std::string result_str = result.Str();

            std::istringstream ss(result_str);
            std::string line;
            while (std::getline(ss, line, '\n')) {
                if (line.empty() || line.find("DetectionResult:") != std::string::npos) // 忽略空行
                    continue;

                std::istringstream iss(line);
                std::vector<float> values; // 存储从当前行中解析的值
                std::string value_str;
                while (std::getline(iss, value_str, ',')) {
                    try {
                        float value = std::stof(value_str);
                        values.push_back(value);
                    }
                    catch (const std::invalid_argument& e) {
                        std::cerr << "Invalid float value: " << value_str << std::endl;
                    }
                }

                if (values.size() != 6) // 检查解析后的值的数量是否为 6
                    continue;

                float xmin = values[0];
                float ymin = values[1];
                float xmax = values[2];
                float ymax = values[3];
                float score = values[4];
                int label_id = static_cast<int>(values[5]);

                if (score < 0.5) // 过滤可信度低于0.5的检测结果
                    continue;

                xmin = static_cast<int>(xmin);
                ymin = static_cast<int>(ymin);
                xmax = static_cast<int>(xmax);
                ymax = static_cast<int>(ymax);

                // 获取标签名称
                std::string label_name;
                auto it = label_map2.find(label_id);
                if (it != label_map2.end()) {
                    label_name = it->second; // 找到了对应的标签
                }
                else {
                    label_name = "Unknown"; // 未找到对应的标签
                }
                draw_pred(label_name, score, xmin, ymin, xmax, ymax, frame);

            }
         };

        auto rtmpPush = [&](cv::Mat& frame) {
            const char* rtmpUrl = "rtmp://192.168.200.55/live/test";
            // 注册所有FFmpeg组件
            av_register_all();
            avformat_network_init();
            
            // 打开RTMP推流地址
            AVFormatContext* outputFormatCtx = nullptr;
            AVOutputFormat* outputFormat = av_guess_format("flv", nullptr, nullptr);
            if (!outputFormat) {
                throw runtime_error("Failed to find output format!");
            }

            int ret = avformat_alloc_output_context2(&outputFormatCtx, outputFormat, nullptr, rtmpUrl);
            if (ret < 0) {
                throw runtime_error("Failed to allocate output context!");
            }

            // 添加视频流
            AVStream* outStream = avformat_new_stream(outputFormatCtx, nullptr);
            if (!outStream) {
                throw runtime_error("Failed to create new stream!");
            }

            // 设置视频流的编码参数
            AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
            if (!codec) {
                throw runtime_error("Failed to find H.264 encoder!");
            }

            AVCodecContext* codecContext = avcodec_alloc_context3(codec);
            if (!codecContext) {
                throw runtime_error("Failed to allocate codec context!");
            }

            codecContext->codec_id = codec->id;
            codecContext->codec_type = AVMEDIA_TYPE_VIDEO;
            codecContext->pix_fmt = AV_PIX_FMT_YUV420P;
            codecContext->width = frame.cols;
            codecContext->height = frame.rows;
            codecContext->time_base = { 1, 30 }; // 设置帧率为30fps
            

            // 打开视频编码器
            ret = avcodec_open2(codecContext, codec, nullptr);
            if (ret < 0) {
                throw runtime_error("Failed to open codec!");
            }

            ret = avcodec_parameters_from_context(outStream->codecpar, codecContext);
            if (ret < 0) {
                throw runtime_error("Failed to copy codec context to output stream!");
            }

            // 打开输出URL
            ret = avio_open(&outputFormatCtx->pb, rtmpUrl, AVIO_FLAG_WRITE);
            if (ret < 0) {
                cout << "Failed to open output URL: " << endl;
                throw runtime_error("Failed to open output URL!");
            }

            // 写入输出文件头
            ret = avformat_write_header(outputFormatCtx, nullptr);
            if (ret < 0) {
                cout << "Error writing packet to output: "  << endl;
                throw runtime_error("Failed to write header!");
            }

            int frame_count = 0;
            // 处理每一帧图像
            while (true) {
                detect(frame);

                AVFrame* avFrame = CVMatToAVFrame(frame);
                if (!avFrame) {
                    throw runtime_error("Failed to allocate AVFrame!");
                }

                avFrame->format = codecContext->pix_fmt;
                avFrame->width = codecContext->width;
                avFrame->height = codecContext->height;
                
                // 设置帧的时间戳
                avFrame->pts = frame_count * (codecContext->time_base.den) / (codecContext->time_base.num * 30);
                frame_count++;

                ret = avcodec_send_frame(codecContext, avFrame);
                if (ret < 0) {
                    throw runtime_error("Error sending frame to codec!");
                }

                AVPacket packet;
                memset(&packet, 0, sizeof(packet));
                packet.size = 0;

                ret = avcodec_receive_packet(codecContext, &packet);
                if (ret != 0 || packet.size > 0) {} // 如果接收失败 (ret != 0) 或者数据包大小为零 (packet.size <= 0)，表示没有接收到有效的编码数据包，需要进行错误处理并继续下一轮循环
                else {
                    av_frame_free(&avFrame);
                    av_free_packet(&packet);
                    continue;
                }
                int firstFrame = 0;
                if (packet.dts < 0 || packet.pts < 0 || packet.dts > packet.pts || firstFrame) {
                    firstFrame = 0;
                    packet.dts = packet.pts = packet.duration = 0;
                }
                    // 设置数据包的时间戳
                packet.pts = av_rescale_q(avFrame->pts, codecContext->time_base, outStream->time_base);
                packet.dts = av_rescale_q(avFrame->pts, codecContext->time_base, outStream->time_base);
                cout << "Packet PTS: " << packet.pts << ", DTS: " << packet.dts << endl;
                packet.stream_index = outStream->index;
                cout << "Output format context: " << (outputFormatCtx ? "valid" : "null") << endl;
                cout << "Packet stream index: " << packet.stream_index << ", size: " << packet.size << endl;
                ret = av_interleaved_write_frame(outputFormatCtx, &packet);
                if (ret < 0) {
                    printf("fasongshujubaochucuo\n");
                    av_frame_free(&avFrame);
                    av_free_packet(&packet);
                    continue;
                }
                av_frame_free(&avFrame);
                av_free_packet(&packet);
            }
            av_write_trailer(outputFormatCtx);
       
            

            //
            //avcodec_free_context(&codecContext);
            //avformat_free_context(outputFormatCtx);
         };


  
        auto do_process = [&] {
            Mat& frame = *image;
            
            rtmpPush(frame);
            //imshow(show_name_.c_str(), frame);
           // cv::waitKey(1);   

         };
        
        do_process();

        
    }




}  // namespace edge_app
