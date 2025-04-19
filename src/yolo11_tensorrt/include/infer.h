#ifndef INFER_H
#define INFER_H

#include <opencv2/opencv.hpp>
#include "public.h"
#include "types.h"
#include "utils.h"

#include <vision_msgs/Detection2DArray.h>
#include <sensor_msgs/Image.h>
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Float32.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <mutex>
#include <queue>
#include <thread>
using namespace nvinfer1;

inline int64_t generateKeyframeId(ros::Time stamp, int self_id)
{
    static int keyframe_count = 0;
    int t_ms = stamp.toSec() * 1000; // stamp.toSec()*1000;
    return (t_ms % 100000) * 10000 + self_id * 1000000 + keyframe_count++;
}

struct VisualImageDesc
{
    ros::Time stamp;
    cv::Mat raw_image;
    cv::Mat raw_depth_image;
    int drone_id = 1;
    int frame_id;
    VisualImageDesc(
        ros::Time _stamp,
        cv::Mat _raw_image,
        cv::Mat _raw_depth_image) : stamp(_stamp), raw_image(_raw_image), raw_depth_image(_raw_depth_image)
    {
        frame_id = generateKeyframeId(_stamp, drone_id);
    }
};
class YoloDetector
{
public:
    YoloDetector(ros::NodeHandle &nh);
    ~YoloDetector();
    void read_config_file(ros::NodeHandle &nh);
    std::vector<Detection> inference(cv::Mat &img);
    void draw_image(cv::Mat &img, std::vector<Detection> &inferResult);
    void inference_image(std::string &image_path);

    std::mutex image_mutex;
    std::queue<VisualImageDesc> image_buf;
    std::thread thread_process_image;
    std::string image_topic_;
    std::string depth_image_topic_;
    std::string image_path_;
    int use_ros_;

    void registerPub();
    void registerCallback();
    void depthImagesCallback(const sensor_msgs::ImageConstPtr img_msg, const sensor_msgs::ImageConstPtr depth);
    void process_Image_Thread();
    void detection_inference(VisualImageDesc &_current_frame);
    // cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);
    cv_bridge::CvImagePtr getImageFromMsg(const sensor_msgs::Image &img_msg);
    cv_bridge::CvImagePtr getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);

private:
    void get_engine();

    ros::NodeHandle nh_;
    image_transport::ImageTransport *it_;
    image_transport::SubscriberFilter *img_sub_, *depth_img_sub_;
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> *sync;
    ros::Timer detection_Timer_;
    ros::Timer vis_Timer_;
    ros::Timer bbox_Timer_;

    ros::Publisher detected_image_pub;
    ros::Publisher detected_boxes_pub;
    ros::Publisher detected_time_pub;

    cv::Mat received_image;
    float detect_time_;
    std::vector<Detection> res;
    bool img_received = false;
    bool img_detected = false;

    std::string config_file_;

private:
    Logger gLogger;
    std::string trtfile_path_;
    std::string onnxfile_path_;
    int gpuId_;
    int numClass_;
    float nmsThresh_;
    float confThresh_;
    int detect_image_H_, detect_image_W_;
    int detect_max_output_bbox_;
    int detect_num_box_element_;
    int use_fp16_ = 0;
    int use_int8_ = 0;
    std::string cacheFile_;
    std::string calibrationDataPath_;
    std::string output_path_;
    std::vector<std::string> vClassNames_;

    ICudaEngine *engine;
    IRuntime *runtime;
    IExecutionContext *context;

    cudaStream_t stream;

    float *outputData;
    std::vector<void *> vBufferD;
    float *transposeDevice;
    float *decodeDevice;

    int OUTPUT_CANDIDATES; // 8400: 80 * 80 + 40 * 40 + 20 * 20
};

#endif // INFER_H
