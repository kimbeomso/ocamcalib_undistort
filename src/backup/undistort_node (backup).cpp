#include "ocamcalib_undistort/ocam_functions.h"
#include "ocamcalib_undistort/Parameters.h"

#include <iostream>
#include <string>
#include <exception>
#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "opencv2/opencv.hpp"
#include <ocamcalib_undistort/PhantomVisionNetMsg.h>
#include <time.h>

#include <std_msgs/Float32MultiArray.h>
#include <pcl_ros/point_cloud.h>
#include <algorithm>

using namespace cv;
using namespace std;

typedef pcl::PointXYZRGB VIMage;
typedef pcl::PointCloud<VIMage> VPointImage;
VPointImage m_avm_data;
// VPointImage m_cam_data_right;
// VPointImage m_cam_data_left;

ros::Publisher Pub_avm;
// ros::Publisher Pub_cam_right;

ros::Publisher Pub_image_center;
ros::Publisher Pub_image_rear;
ros::Publisher Pub_image_left;
ros::Publisher Pub_image_right;
// ros::Publisher Pub_AVM_full_img;
// ros::Publisher Pub_AVM_side_img;
// ros::Publisher Pub_AVM_center_img;

// cv::Mat mapx_persp_front;
// cv::Mat mapy_persp_front;

// cv::Mat mapx_persp_rear;
// cv::Mat mapy_persp_rear;

// cv::Mat mapx_persp_left;
// cv::Mat mapy_persp_left;

// cv::Mat mapx_persp_right;
// cv::Mat mapy_persp_right;

cv::Mat AVM_left = cv::Mat::zeros(400,400, CV_8UC3);// = cv::Mat::zeros(450,450, CV_8UC3);
cv::Mat AVM_right = cv::Mat::zeros(400,400, CV_8UC3);// = cv::Mat::zeros(450,450, CV_8UC3);
cv::Mat AVM_front = cv::Mat::zeros(400,400, CV_8UC3);
cv::Mat AVM_rear = cv::Mat::zeros(400,400, CV_8UC3);

cv::Mat AVM_left_half = cv::Mat::zeros(400,400, CV_8UC3);// = cv::Mat::zeros(450,450, CV_8UC3);
cv::Mat AVM_right_half = cv::Mat::zeros(400,400, CV_8UC3);// = cv::Mat::zeros(450,450, CV_8UC3);
cv::Mat AVM_front_half = cv::Mat::zeros(400,400, CV_8UC3);
cv::Mat AVM_rear_half = cv::Mat::zeros(400,400, CV_8UC3);

cv::Mat AVM_left_quarter = cv::Mat::zeros(400,400, CV_8UC3);// = cv::Mat::zeros(450,450, CV_8UC3);
cv::Mat AVM_right_quarter = cv::Mat::zeros(400,400, CV_8UC3);// = cv::Mat::zeros(450,450, CV_8UC3);
cv::Mat AVM_front_quarter = cv::Mat::zeros(400,400, CV_8UC3);
cv::Mat AVM_rear_quarter = cv::Mat::zeros(400,400, CV_8UC3);



cv::Mat AVM_seg_front= cv::Mat::zeros(400,400, CV_8UC3);
cv::Mat AVM_seg_rear= cv::Mat::zeros(400,400, CV_8UC3);
cv::Mat AVM_seg_left= cv::Mat::zeros(400,400, CV_8UC3);
cv::Mat AVM_seg_right= cv::Mat::zeros(400,400, CV_8UC3);

cv::Mat ORG_image_front= cv::Mat::zeros(320, 200, CV_8UC3); 
cv::Mat ORG_image_rear= cv::Mat::zeros(320, 200, CV_8UC3); 
cv::Mat ORG_image_left= cv::Mat::zeros(320, 200, CV_8UC3); 
cv::Mat ORG_image_right= cv::Mat::zeros(320, 200, CV_8UC3); 

cv::Mat result = cv::Mat::zeros(1200, 400, CV_8UC3);
cv::Mat result_seg = cv::Mat::zeros(1200, 400, CV_8UC3);
cv::Mat temp = cv::Mat::zeros(1200, 800, CV_8UC3);
cv::Mat temp2 = cv::Mat::zeros(1520, 800, CV_8UC3);

cv::Mat ORG = cv::Mat::zeros(320, 200, CV_8UC3);
cv::Mat ORG2= cv::Mat::zeros(320, 200, CV_8UC3);

cv::Mat result1 = cv::Mat::zeros(400, 400, CV_8UC3);
cv::Mat result2 = cv::Mat::zeros(400, 400 , CV_8UC3);
cv::Mat result3 = cv::Mat::zeros(400, 400 , CV_8UC3); 
cv::Mat result4 = cv::Mat::zeros(400, 400 , CV_8UC3); 
cv::Mat result5 = cv::Mat::zeros(400, 400 , CV_8UC3); 
cv::Mat result6 = cv::Mat::zeros(400, 400 , CV_8UC3); 
cv::Mat full    = cv::Mat::zeros(400, 400 , CV_8UC3);
cv::Mat half    = cv::Mat::zeros(400, 400 , CV_8UC3);
cv::Mat quarter    = cv::Mat::zeros(400, 400 , CV_8UC3);

cv::Mat full_side = cv::Mat::zeros(400, 400, CV_8UC3);
cv::Mat half_side = cv::Mat::zeros(400, 400, CV_8UC3);
cv::Mat quarter_side = cv::Mat::zeros(400, 400, CV_8UC3);
cv::Mat full_cen = cv::Mat::zeros(400, 400, CV_8UC3);
cv::Mat half_cen = cv::Mat::zeros(400, 400, CV_8UC3);
cv::Mat quarter_cen = cv::Mat::zeros(400, 400, CV_8UC3);



// cv::Mat AVM_temp;

ocam_model front_model;
ocam_model  left_model;
ocam_model right_model;
ocam_model  rear_model;

char buf[256];
char buf2[256];
char buf3[256];
char buf4[256];
char buf5[256];
char buf6[256];
char buf7[256];
char buf8[256];
char buf9[256];


int idx =0;
int flag=0;
double start;
double endd;


#define M_DEG2RAD  3.1415926 / 180.0
double M_front_param[6] = {0.688 * M_DEG2RAD,  21.631 * M_DEG2RAD,   3.103* M_DEG2RAD   ,1.905,   0.033, 0.707 };
double M_left_param[6] =  {1.133 * M_DEG2RAD,  19.535 * M_DEG2RAD,   92.160* M_DEG2RAD  ,0.0,     1.034, 0.974 };
double M_right_param[6] = {3.440 * M_DEG2RAD,  18.273 * M_DEG2RAD,  -86.127* M_DEG2RAD  ,0.0,    -1.034, 0.988 };
double M_back_param[6] =  {0.752 * M_DEG2RAD,  31.238 * M_DEG2RAD,  -178.189* M_DEG2RAD ,-2.973, -0.065, 0.883 };

// void onMouse(int event, int x, int y, int flags, void* param) {
// 	switch (event) {

// 	case EVENT_LBUTTONDOWN:
		
// 		break;
// 	}
// }


Parameters getParameters(ros::NodeHandle& nh)
{
    Parameters params;

    nh.param<std::string>("camera_type", params.cameraType, "fisheye");
    // nh.param<std::string>("base_in_topic", params.inTopic, 'camera/image_center');
    // nh.param<std::string>("base_out_topic", params.outTopic, "/ocamcalib_undistorted");
    nh.param<std::string>("calibration_file_path", params.calibrationFile, "include/calib_results_left.txt"); //center.txt");

    nh.param<std::string>("transport_hint", params.transportHint, "raw");
    nh.param<double>("scale_factor", params.scaleFactor, 15);
    nh.param<int>("left_bound", params.leftBound,   0);
    nh.param<int>("right_bound", params.rightBound, 0);
    nh.param<int>("top_bound", params.topBound, 0);
    nh.param<int>("bottom_bound", params.bottomBound, 0);

    return params;
}

// void imageCallback_front(
//         const sensor_msgs::ImageConstPtr& msg,
//         const cv::Mat& mapx,
//         const cv::Mat& mapy)
// {

//     auto inImage = cv_bridge::toCvShare(msg);
//     cv::Mat undistorted(inImage->image.size(), inImage->image.type());
//     // std::cout << inImage->image.size() << std::endl;
//     cv::remap(
//             inImage->image,
//             undistorted,
//             mapx,
//             mapy,
//             cv::INTER_LINEAR,
//             cv::BORDER_CONSTANT,
//             cv::Scalar(0));

//     cv::Mat cropped(undistorted);

//     cv::Mat out(inImage->image.size(), inImage->image.type());
//     cv::resize(cropped, out, inImage->image.size());

//     cv_bridge::CvImage outImage;
//     outImage.image = out;
//     outImage.header = inImage->header;
//     outImage.encoding = inImage->encoding;

//     Pub_image_center.publish(outImage.toImageMsg());
// }

// void imageCallback_left(
//         const sensor_msgs::ImageConstPtr& msg,
//         const cv::Mat& mapx,
//         const cv::Mat& mapy)

// {
//     auto inImage = cv_bridge::toCvShare(msg);
//     cv::Mat undistorted(inImage->image.size(), inImage->image.type());
//     // std::cout << inImage->image.size() << std::endl;
//     cv::remap(
//             inImage->image,
//             undistorted,
//             mapx,
//             mapy,
//             cv::INTER_LINEAR,
//             cv::BORDER_CONSTANT,
//             cv::Scalar(0));

//     cv::Mat cropped(undistorted);

//     cv::Mat out(inImage->image.size(), inImage->image.type());
//     cv::resize(cropped, out, inImage->image.size());

//     cv_bridge::CvImage outImage;
//     outImage.image = out;
//     outImage.header = inImage->header;
//     outImage.encoding = inImage->encoding;

//     Pub_image_left.publish(outImage.toImageMsg());
// }

// void imageCallback_right(
//         const sensor_msgs::ImageConstPtr& msg,
//         const cv::Mat& mapx,
//         const cv::Mat& mapy)

// {
//     auto inImage = cv_bridge::toCvShare(msg);
//     cv::Mat undistorted(inImage->image.size(), inImage->image.type());
//     // std::cout << inImage->image.size() << std::endl;
//     cv::remap(
//             inImage->image,
//             undistorted,
//             mapx,
//             mapy,
//             cv::INTER_LINEAR,
//             cv::BORDER_CONSTANT,
//             cv::Scalar(0));

//     cv::Mat cropped(undistorted);

//     cv::Mat out(inImage->image.size(), inImage->image.type());
//     cv::resize(cropped, out, inImage->image.size());

//     cv_bridge::CvImage outImage;
//     outImage.image = out;
//     outImage.header = inImage->header;
//     outImage.encoding = inImage->encoding;

//     Pub_image_right.publish(outImage.toImageMsg());
// }
// void imageCallback_rear(
//         const sensor_msgs::ImageConstPtr& msg,
//         const cv::Mat& mapx,
//         const cv::Mat& mapy)

// {
//     auto inImage = cv_bridge::toCvShare(msg);
//     cv::Mat undistorted(inImage->image.size(), inImage->image.type());
//     // std::cout << inImage->image.size() << std::endl;
//     cv::remap(
//             inImage->image,
//             undistorted,
//             mapx,
//             mapy,
//             cv::INTER_LINEAR,
//             cv::BORDER_CONSTANT,
//             cv::Scalar(0));

//     cv::Mat cropped(undistorted);

//     cv::Mat out(inImage->image.size(), inImage->image.type());
//     cv::resize(cropped, out, inImage->image.size());

//     cv_bridge::CvImage outImage;
//     outImage.image = out;
//     outImage.header = inImage->header;
//     outImage.encoding = inImage->encoding;

//     Pub_image_rear.publish(outImage.toImageMsg());
// }
void printOcamModel(const struct ocam_model& model)
{
    std::cout << "OCamCalib model parameters" << std::endl
              << "pol: " << std::endl;
    for (int i=0; i < model.length_pol; i++)
    {
        std::cout << "\t" << model.pol[i] << "\n";
    }

    std::cout << "\ninvpol: " << std::endl;
    for (int i=0; i < model.length_invpol; i++)
    {
        std::cout << "\t" << model.invpol[i] << "\n";
    };
    std::cout << std::endl;

    std::cout << "xc:\t" << model.xc << std::endl
              << "yc:\t" << model.yc << std::endl
              << "width:\t" << model.width << std::endl
              << "height:\t" << model.height << std::endl;
}

// void callback_image (const sensor_msgs::ImageConstPtr& msg) {
//     imageCallback_front(msg, mapx_persp, mapy_persp);
// }

void seg2rgb(cv::Mat input_img, cv::Mat& output_img) {
    for (int i = 0 ; i < input_img.rows ; i++) {
        for (int j = 0 ; j < input_img.cols ; j++) {
            // if ((int)input_img.at<uchar>(i, j) != 0 && (int)input_img.at<uchar>(i, j) != 1 && (int)input_img.at<uchar>(i, j) != 2&& (int)input_img.at<uchar>(i, j) != 8&& (int)input_img.at<uchar>(i, j) != 11)
                // cout << (int)input_img.at<uchar>(i, j) << endl;
                
            switch((int)input_img.at<uchar>(i, j)){
                case 1 : // vehicle
                    output_img.at<Vec3b>(i, j)[0] = 0;
                    output_img.at<Vec3b>(i, j)[1] = 0;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                case 2 : // wheel
                    output_img.at<Vec3b>(i, j)[0] = 255;
                    output_img.at<Vec3b>(i, j)[1] = 0;
                    output_img.at<Vec3b>(i, j)[2] = 0;
                break;  
                case 3 : 
                    output_img.at<Vec3b>(i, j)[0] = 0;
                    output_img.at<Vec3b>(i, j)[1] = 255;
                    output_img.at<Vec3b>(i, j)[2] = 123;
                break;  
                case 4 : 
                    output_img.at<Vec3b>(i, j)[0] = 0;
                    output_img.at<Vec3b>(i, j)[1] = 125;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                case 5 : 
                    output_img.at<Vec3b>(i, j)[0] = 255;
                    output_img.at<Vec3b>(i, j)[1] = 0;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                case 6 : 
                    output_img.at<Vec3b>(i, j)[0] = 0;
                    output_img.at<Vec3b>(i, j)[1] = 255;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                case 7 : // human
                    output_img.at<Vec3b>(i, j)[0] = 255;
                    output_img.at<Vec3b>(i, j)[1] = 255;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                case 8 :    // road 
                    output_img.at<Vec3b>(i, j)[0] = 0;
                    output_img.at<Vec3b>(i, j)[1] = 255;
                    output_img.at<Vec3b>(i, j)[2] = 0;
                break;
                case 10 : 
                    output_img.at<Vec3b>(i, j)[0] = 0;
                    output_img.at<Vec3b>(i, j)[1] = 165;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                case 11 : // cart?
                    output_img.at<Vec3b>(i, j)[0] = 193;
                    output_img.at<Vec3b>(i, j)[1] = 182;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                default :
                    output_img.at<Vec3b>(i, j)[0] = 0;
                    output_img.at<Vec3b>(i, j)[1] = 0;
                    output_img.at<Vec3b>(i, j)[2] = 0;
            }
        }
    }
}

void CallbackPhantom_seg_front(const ocamcalib_undistort::PhantomVisionNetMsg::ConstPtr& msg) {
    cv::Mat cv_frame_raw_new = cv_bridge::toCvCopy( msg->image_raw, msg->image_raw.encoding )->image;
    cv::Mat cv_frame_seg = cv_bridge::toCvCopy( msg->image_seg, msg->image_seg.encoding )->image;
    seg2rgb(cv_frame_seg, cv_frame_raw_new);
    
    //resize
    cv::Mat cv_frame_resize;
    cv::resize( cv_frame_raw_new, cv_frame_resize, Size(1920, 1080), 0, 0, INTER_LINEAR );

    cv::Mat cv_frame_resize_pad;
    cv::copyMakeBorder(cv_frame_resize, cv_frame_resize_pad, 0, 128, 0, 0, BORDER_CONSTANT, Scalar(0,0,0) );

    XY_coord xy;
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(960, 604), 0, 0, cv::INTER_LINEAR );
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(640, 402), 0, 0, cv::INTER_LINEAR );
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(480, 302), 0, 0, cv::INTER_LINEAR );
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // imshow("test",cv_frame_resize_pad);                                                              //
    // waitKey(1);                                                                                      //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    int u,v;
        AVM_seg_front = cv::Mat::zeros(400,400, CV_8UC3);
        for(int i=0; i< (cv_frame_resize_pad.size().height)  ;i++)
            for(int j=0 ;j< cv_frame_resize_pad.size().width ;j++){
                // xy = InvProjGRD(4*j,4*i, M_front_param[0], M_front_param[1], M_front_param[2], M_front_param[3], M_front_param[4] ,M_front_param[5], &front_model);
                // xy = InvProjGRD(3*j,3*i, M_front_param[0], M_front_param[1], M_front_param[2], M_front_param[3], M_front_param[4] ,M_front_param[5], &front_model);
                // xy = InvProjGRD(2*j,2*i, M_front_param[0], M_front_param[1], M_front_param[2], M_front_param[3], M_front_param[4] ,M_front_param[5], &front_model);
                xy = InvProjGRD(j,i, M_front_param[0], M_front_param[1], M_front_param[2], M_front_param[3], M_front_param[4] ,M_front_param[5], &front_model);


                if(xy.x < 0){
                    xy.y =0;
                    xy.x =0;
                }
                if( !(xy.x ==0) && (xy.y ==0) || !((abs(xy.x) >=12.5) || (abs(xy.y) >=12.5)) ){
                    v = 200 - 16*xy.x;  // width
                    u = 200 - 16*xy.y;  // height

                    AVM_seg_front.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);   //b
                    AVM_seg_front.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);   //g
                    AVM_seg_front.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);   //r
                }
            }
}
void CallbackPhantom_seg_rear(const ocamcalib_undistort::PhantomVisionNetMsg::ConstPtr& msg){
    cv::Mat cv_frame_raw_new = cv_bridge::toCvCopy( msg->image_raw, msg->image_raw.encoding )->image;
    cv::Mat cv_frame_seg = cv_bridge::toCvCopy( msg->image_seg, msg->image_seg.encoding )->image;
    seg2rgb(cv_frame_seg, cv_frame_raw_new);
    
    //resize
    cv::Mat cv_frame_resize;
    cv::resize( cv_frame_raw_new, cv_frame_resize, Size(1920, 1080), 0, 0, INTER_LINEAR );

    cv::Mat cv_frame_resize_pad;
    cv::copyMakeBorder(cv_frame_resize, cv_frame_resize_pad, 0, 128, 0, 0, BORDER_CONSTANT, Scalar(0,0,0) );

    XY_coord xy;
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(960, 604), 0, 0, cv::INTER_LINEAR );
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(640, 402), 0, 0, cv::INTER_LINEAR );
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(480, 302), 0, 0, cv::INTER_LINEAR );
    int u,v;
    AVM_seg_rear = cv::Mat::zeros(400,400, CV_8UC3);
    for(int i=0; i< (cv_frame_resize_pad.size().height)  ;i++)
        for(int j=0 ;j< cv_frame_resize_pad.size().width ;j++){
            // xy = InvProjGRD(4*j,4*i, M_back_param[0], M_back_param[1], M_back_param[2], M_back_param[3], M_back_param[4] ,M_back_param[5], &rear_model);
            // xy = InvProjGRD(3*j,3*i, M_back_param[0], M_back_param[1], M_back_param[2], M_back_param[3], M_back_param[4] ,M_back_param[5], &rear_model);
            // xy = InvProjGRD(2*j,2*i, M_back_param[0], M_back_param[1], M_back_param[2], M_back_param[3], M_back_param[4] ,M_back_param[5], &rear_model);
            xy = InvProjGRD(j,i, M_back_param[0], M_back_param[1], M_back_param[2], M_back_param[3], M_back_param[4] ,M_back_param[5], &rear_model);


            if(xy.x > 0){
                xy.y =0;
                xy.x =0;
            }
            if( !(xy.x ==0) && (xy.y ==0) || !((abs(xy.x) >=12.5) || (abs(xy.y) >=12.5)) ){
                v = 200 - 16*xy.x;  // width
                u = 200 - 16*xy.y;  // height
                
                AVM_seg_rear.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);   //b
                AVM_seg_rear.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);   //g
                AVM_seg_rear.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);   //r
            }
        }
        // cv::imshow("rear_seg",AVM_seg_rear);
}
void CallbackPhantom_seg_left(const ocamcalib_undistort::PhantomVisionNetMsg::ConstPtr& msg){
    cv::Mat cv_frame_raw_new = cv_bridge::toCvCopy( msg->image_raw, msg->image_raw.encoding )->image;
    cv::Mat cv_frame_seg = cv_bridge::toCvCopy( msg->image_seg, msg->image_seg.encoding )->image;
    seg2rgb(cv_frame_seg, cv_frame_raw_new);
    
    //resize
    cv::Mat cv_frame_resize;
    cv::resize( cv_frame_raw_new, cv_frame_resize, Size(1920, 1080), 0, 0, INTER_LINEAR );

    cv::Mat cv_frame_resize_pad;
    cv::copyMakeBorder(cv_frame_resize, cv_frame_resize_pad, 0, 128, 0, 0, BORDER_CONSTANT, Scalar(0,0,0) );

    XY_coord xy;
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(960, 604), 0, 0, cv::INTER_LINEAR );
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(640, 402), 0, 0, cv::INTER_LINEAR );
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(480, 302), 0, 0, cv::INTER_LINEAR );
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // imshow("test",cv_frame_resize_pad);                                                              //
    // waitKey(1);                                                                                      //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    int u,v;
        AVM_seg_left = cv::Mat::zeros(400,400, CV_8UC3);
        for(int i=0; i< (cv_frame_resize_pad.size().height)  ;i++)
            for(int j=0 ;j< cv_frame_resize_pad.size().width ;j++){
                // xy = InvProjGRD(4*j,4*i, M_left_param[0], M_left_param[1], M_left_param[2], M_left_param[3], M_left_param[4] ,M_left_param[5], &left_model);
                // xy = InvProjGRD(3*j,3*i, M_left_param[0], M_left_param[1], M_left_param[2], M_left_param[3], M_left_param[4] ,M_left_param[5], &left_model);
                // xy = InvProjGRD(2*j,2*i, M_left_param[0], M_left_param[1], M_left_param[2], M_left_param[3], M_left_param[4] ,M_left_param[5], &left_model);
                xy = InvProjGRD(j,i, M_left_param[0], M_left_param[1], M_left_param[2], M_left_param[3], M_left_param[4] ,M_left_param[5], &left_model);


                if(xy.y < 0){
                    xy.y =0;
                    xy.x =0;
                }
                if( !(xy.x ==0) && (xy.y ==0) || !((abs(xy.x) >=12.5) || (abs(xy.y) >=12.5)) ){
                    v = 200 - 16*xy.x;  // width
                    u = 200 - 16*xy.y;  // height
                
                    AVM_seg_left.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);   //b
                    AVM_seg_left.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);   //g
                    AVM_seg_left.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);   //r
                    
                }  
        }
}

void CallbackPhantom_seg_right(const ocamcalib_undistort::PhantomVisionNetMsg::ConstPtr& msg){
    cv::Mat cv_frame_raw_new = cv_bridge::toCvCopy( msg->image_raw, msg->image_raw.encoding )->image;
    cv::Mat cv_frame_seg = cv_bridge::toCvCopy( msg->image_seg, msg->image_seg.encoding )->image;
    seg2rgb(cv_frame_seg, cv_frame_raw_new);
    
    //resize
    cv::Mat cv_frame_resize;
    cv::resize( cv_frame_raw_new, cv_frame_resize, Size(1920, 1080), 0, 0, INTER_LINEAR );

    cv::Mat cv_frame_resize_pad;
    cv::copyMakeBorder(cv_frame_resize, cv_frame_resize_pad, 0, 128, 0, 0, BORDER_CONSTANT, Scalar(0,0,0) );

    XY_coord xy;
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(960, 604), 0, 0, cv::INTER_LINEAR );
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(640, 402), 0, 0, cv::INTER_LINEAR );
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(480, 302), 0, 0, cv::INTER_LINEAR );

    int u,v;
        AVM_seg_right = cv::Mat::zeros(400,400, CV_8UC3);
        for(int i=0; i< (cv_frame_resize_pad.size().height) ;i++)
        for(int j=0 ;j< cv_frame_resize_pad.size().width ;j++){
            // xy = InvProjGRD(4*j,4*i, M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4] ,M_right_param[5], &right_model);
            // xy = InvProjGRD(3*j,3*i, M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4] ,M_right_param[5], &right_model);
            // xy = InvProjGRD(2*j,2*i, M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4] ,M_right_param[5], &right_model);
            xy = InvProjGRD(j,i, M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4] ,M_right_param[5], &right_model);

            if(xy.y > 0){
                xy.y =0;
                xy.x =0;
            }
            
            if( !(xy.x ==0) && (xy.y ==0) || !((abs(xy.x) >=12.5) || (abs(xy.y) >=12.5)) ){     //25 meters
                v = 200 - 16*xy.x; 
                u = 200 - 16*xy.y; 

                AVM_seg_right.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);   //b
                AVM_seg_right.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);   //g
                AVM_seg_right.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);   //r
            }

        }
}

unsigned int data_count=0;
void CallbackPhantom_DR(const std_msgs::Float32MultiArray::ConstPtr& msg){  
    
    double x=0;
    double y=0;
    double m_x=0;
    double m_y=0;
    double DR_x= msg->data.at(0);
    double DR_y= msg->data.at(1);
    double DR_yaw = msg->data.at(2);

    sensor_msgs::ImagePtr msg_img_avm = cv_bridge::CvImage(std_msgs::Header(), "bgr8", result1).toImageMsg();

    // m_avm_data.clear();
    m_avm_data.header.frame_id = "map";
    m_avm_data.points.resize(10000000);
    
    
    // unsigned int data_count=0;

    for(int i=75; i< result1.size().height-75 ;i=i+2){
        for(int j=75 ;j< result1.size().width -75 ;j=j+2){
            x = 12.5 - 0.0625 * i;
            y = 12.5 - 0.0625 * j;

            m_x =  DR_x + cos(DR_yaw) * x - sin(DR_yaw) * y;
            m_y =  DR_y + sin(DR_yaw) * x + cos(DR_yaw) * y;

            if(!( result1.at<cv::Vec3b>(i,j)[1]==0 ) && !(result1.at<cv::Vec3b>(i,j)[0]==0) && !(result1.at<cv::Vec3b>(i,j)[2]==0))
            {   
                m_avm_data.points[data_count].x = m_x;
                m_avm_data.points[data_count].y = m_y;
                m_avm_data.points[data_count].z = 0.0;

                m_avm_data.points[data_count].g = static_cast<uint8_t>(result1.at<cv::Vec3b>(i,j)[1]);      //g
                m_avm_data.points[data_count].b = static_cast<uint8_t>(result1.at<cv::Vec3b>(i,j)[0]);      //b
                m_avm_data.points[data_count++].r = static_cast<uint8_t>(result1.at<cv::Vec3b>(i,j)[2]);    //r
            }
        }
    }
    m_avm_data.points.resize(data_count);
    // std::cout<< data_count << std::endl;
    Pub_avm.publish(m_avm_data);
}


void CallbackPhantom_center(const ocamcalib_undistort::PhantomVisionNetMsg::ConstPtr& msg) 
{   
    cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg->image_raw, msg->image_raw.encoding )->image;

    //resize
    // cv::Mat cv_frame_resize;
    cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(1920, 1080), 0, 0, cv::INTER_LINEAR );
    cv::resize( cv_frame_resize_pad, ORG_image_front, cv::Size(320, 200), 0, 0, cv::INTER_LINEAR );
    
    // padding
    // cv::Mat cv_frame_resize_pad;
    cv::Mat temp;
    cv::copyMakeBorder(cv_frame_resize_pad, cv_frame_resize_pad, 0, 128, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
    // sensor_msgs::ImagePtr msg_img_center = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_frame_resize_pad).toImageMsg();
    XY_coord xy; 
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(960, 604), 0, 0, cv::INTER_LINEAR );
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(640, 402), 0, 0, cv::INTER_LINEAR );
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(480, 302), 0, 0, cv::INTER_LINEAR );
    int u,v;

    string myText = "Front";
    cv::Point myPoint;
    myPoint.x = 10;
    myPoint.y = 20;
    cv::putText( ORG_image_front, myText, myPoint, 2, 0.7, Scalar(0, 255, 255) );
    
    // m_cam_data.clear();
    // m_cam_data.header.frame_id = "map";   //"camera_init";
    // m_cam_data.points.resize(msg_img_center->data.size());
    // cv::Mat temp = cv::Mat::zeros(400,400, CV_8UC3);


    // start = endd;
    // endd = ros::Time::now().toSec();
    // std::cout<<endd-start<<std::endl;


    AVM_front = cv::Mat::zeros(400,400, CV_8UC3);
    // AVM_front_half = cv::Mat::zeros(400,400, CV_8UC3);
    // AVM_front_quarter = cv::Mat::zeros(400,400, CV_8UC3);

    // std::cout<<"asdasdsad"<<std::endl;
    for(int i=0; i< (cv_frame_resize_pad.size().height)  ;i++)
        for(int j=0 ;j< cv_frame_resize_pad.size().width ;j++){
        
            // std::cout<<cv_frame_resize_pad.size().height<<cv_frame_resize_pad.size().width<<std::endl;
            // xy = InvProjGRD(4*j,4*i, M_front_param[0], M_front_param[1], M_front_param[2], M_front_param[3], M_front_param[4] ,M_front_param[5], &front_model);
            // xy = InvProjGRD(3*j+1,3*i+1, M_front_param[0], M_front_param[1], M_front_param[2], M_front_param[3], M_front_param[4] ,M_front_param[5], &front_model);
            // xy = InvProjGRD(2*j,2*i, M_front_param[0], M_front_param[1], M_front_param[2], M_front_param[3], M_front_param[4] ,M_front_param[5], &front_model);
            xy = InvProjGRD(j,i, M_front_param[0], M_front_param[1], M_front_param[2], M_front_param[3], M_front_param[4] ,M_front_param[5], &front_model);

            // if(xy.x < 0){
            //     xy.y = 0;
            //     xy.x = 0;
            // }
            if( !(xy.x ==0) && (xy.y ==0) || !((abs(xy.x) >=12.5) || (abs(xy.y) >=12.5)) ){
                v = 200 - 16*xy.x;  // width
                u = 200 - 16*xy.y;  // height

                // if ((AVM.at<cv::Vec3b>(int(v), int(u))[0] == 0) && (AVM.at<cv::Vec3b>(int(v), int(u))[1] == 0) && (AVM.at<cv::Vec3b>(int(v), int(u))[2] == 0)){
                AVM_front.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);   //b
                AVM_front.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);   //g
                AVM_front.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);   //r

                // if((i%2==0) && (j%2==0)){
                //     AVM_front_half.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);
                //     AVM_front_half.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);
                //     AVM_front_half.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);
                // }
                // if((i%4==0) && (j%4==0)){
                //     AVM_front_quarter.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);
                //     AVM_front_quarter.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);
                //     AVM_front_quarter.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);
                // }
                
                // if(((xy.x <= 1.905) && (xy.x > 0) ) && ((xy.y <= 1.034) && (xy.y > 0))   ||   ((xy.x <= 1.905) && (xy.x > 0) ) && ((xy.y >= -1.034) && (xy.y < 0)))
                // {
                //     AVM_front.at<cv::Vec3b>(int(v), int(u))[0] = 255;
                //     AVM_front.at<cv::Vec3b>(int(v), int(u))[1] = 0;
                //     AVM_front.at<cv::Vec3b>(int(v), int(u))[2] = 255;
                // }
                // }
            }
            
        }
        


    // if(!(AVM_temp.empty()))
    //     cv::addWeighted(AVM_front, 0.5, AVM_temp, 0.5, 0.0, temp);
    // cv::imshow("AVM_front", AVM_front);
    // cv::waitKey(1);

    // size_t data_count_cam = 0;
    // int x,y;
    // int a = 0;
    // for(int i=0; i< cv_frame_resize_pad.size().height ;i++)
    //     for(int j=0 ;j< cv_frame_resize_pad.size().width ;j++){
    //         xy = InvProjGRD(j,i, M_front_param[0], M_front_param[1], M_front_param[2], M_front_param[3], M_front_param[4] ,M_front_param[5], &front_model);
    //         // std::cout << '(' << xy.x << ',' << xy.y << ')'<<std::endl;
    //         m_cam_data.points[data_count_cam].x = xy.x;
    //         m_cam_data.points[data_count_cam].y = xy.y;
    //         m_cam_data.points[data_count_cam].z = 0.0;
            
    //         // for(; a< msg_img_center->data.size(); a+=3)
    //         // m_cam_data.points[data_count_cam].g = msg_img_center->data[a];
    //         // m_cam_data.points[data_count_cam].b = msg_img_center-> data[a + 1]; 
    //         // m_cam_data.points[data_count_cam++].r = msg_img_center-> data[a + 2]; 
    //         m_cam_data.points[data_count_cam].g = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(j,i)[1]);
    //         m_cam_data.points[data_count_cam].b = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(j,i)[0]);
    //         m_cam_data.points[data_count_cam++].r = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(j,i)[2]);
            
    //         // Vec3b : CV_8UC3
    //         // if(( abs( p.x ) < 500 )&&( abs( p.y ) < 200 )&&( abs( p.z ) < 500 ))
    //         // cloud->points.push_back(p);
            
    //     }

    // // std::cout << "asdsadsadsadsadsad"<<std::endl;
    // std::cout << data_count_cam << std::endl;
    
    // m_cam_data.points.resize(data_count_cam);
    // Pub_cam.publish(m_cam_data);

//m_cam_data.points.resize(data_count_cam);

    // imageCallback_front(msg_img_center, mapx_persp_front, mapy_persp_front);
    // Pub_image_center.publish(msg_img_center);

    // cv::waitKey(1);
}
void CallbackPhantom_rear(const ocamcalib_undistort::PhantomVisionNetMsg::ConstPtr& msg) 
{
    cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg->image_raw, msg->image_raw.encoding )->image;
    //resize
    // cv::Mat cv_frame_resize;
    cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(1920, 1080), 0, 0, cv::INTER_LINEAR );
    cv::resize( cv_frame_resize_pad, ORG_image_rear, cv::Size(320, 200), 0, 0, cv::INTER_LINEAR );

    //padding
    // cv::Mat cv_frame_resize_pad;
    cv::Mat temp;
    cv::copyMakeBorder(cv_frame_resize_pad, cv_frame_resize_pad, 0, 128, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
    // sensor_msgs::ImagePtr msg_img_rear = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_frame_resize_pad).toImageMsg();

    XY_coord xy; 
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(960, 604), 0, 0, cv::INTER_LINEAR );
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(640, 402), 0, 0, cv::INTER_LINEAR );
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(480, 302), 0, 0, cv::INTER_LINEAR );

    int u,v;

    string myText = "Rear";
    cv::Point myPoint;
    myPoint.x = 10;
    myPoint.y = 20;
    cv::putText( ORG_image_rear, myText, myPoint, 2, 0.7,Scalar(0, 255, 255) );
    
    // double end = ros::Time::now().toSec();
    // std::cout<<end-start<<std::endl;
    AVM_rear = cv::Mat::zeros(400,400, CV_8UC3);
    AVM_rear_half = cv::Mat::zeros(400,400, CV_8UC3);
    AVM_rear_quarter = cv::Mat::zeros(400,400, CV_8UC3);

    for(int i=0; i< (cv_frame_resize_pad.size().height)  ;i++)
        for(int j=0 ;j< cv_frame_resize_pad.size().width ;j++){
            // xy = InvProjGRD(4*j,4*i, M_back_param[0], M_back_param[1], M_back_param[2], M_back_param[3], M_back_param[4] ,M_back_param[5], &rear_model);
            // xy = InvProjGRD(3*j+1,3*i+1, M_back_param[0], M_back_param[1], M_back_param[2], M_back_param[3], M_back_param[4] ,M_back_param[5], &rear_model);            
            // xy = InvProjGRD(2*j,2*i, M_back_param[0], M_back_param[1], M_back_param[2], M_back_param[3], M_back_param[4] ,M_back_param[5], &rear_model);
            xy = InvProjGRD(j,i, M_back_param[0], M_back_param[1], M_back_param[2], M_back_param[3], M_back_param[4] ,M_back_param[5], &rear_model);
          
            // if(xy.x > 0){
            //     xy.y =0;
            //     xy.x =0;
            // }
            if( !(xy.x ==0) && (xy.y ==0) || !((abs(xy.x) >=12.5) || (abs(xy.y) >=12.5)) ){
                v = 200 - 16*xy.x;  // width
                u = 200 - 16*xy.y;  // height

                // if ((AVM.at<cv::Vec3b>(int(v), int(u))[0] == 0) && (AVM.at<cv::Vec3b>(int(v), int(u))[1] == 0) && (AVM.at<cv::Vec3b>(int(v), int(u))[2] == 0)){
                AVM_rear.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);   //b
                AVM_rear.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);   //g
                AVM_rear.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);   //r
                // }
                // if((i%2==0) && (j%2==0)){
                //     AVM_rear_half.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);
                //     AVM_rear_half.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);
                //     AVM_rear_half.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);
                // }
                // if((i%4==0) && (j%4==0)){
                //     AVM_rear_quarter.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);
                //     AVM_rear_quarter.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);
                //     AVM_rear_quarter.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);
                // }
                // if(((xy.x >= -2.973) && (xy.x < 0) ) && ((xy.y >= -1.034) && (xy.y < 0)) || ((xy.x >= -2.973) && (xy.x < 0) ) && ((xy.y <= 1.034) && (xy.y > 0)))     
                // {
                //     AVM_front.at<cv::Vec3b>(int(v), int(u))[0] = 255;
                //     AVM_front.at<cv::Vec3b>(int(v), int(u))[1] = 0;
                //     AVM_front.at<cv::Vec3b>(int(v), int(u))[2] = 255;
                // }
            }
        }


    // cv::imshow("AVM_rear", AVM_rear);
}
void CallbackPhantom_left(const ocamcalib_undistort::PhantomVisionNetMsg::ConstPtr& msg) 
{
    // double start =ros::Time::now().toSec();
    // // clock_t start = clock();
    cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg->image_raw, msg->image_raw.encoding )->image;
    
    // //resize
    // cv::Mat cv_frame_resize;
    cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(1920, 1080), 0, 0, cv::INTER_LINEAR );
    cv::resize( cv_frame_resize_pad, ORG_image_left, cv::Size(320, 200), 0, 0, cv::INTER_LINEAR );
    
    // //padding
    // cv::Mat cv_frame_resize_pad;
    cv::Mat temp;
    cv::Rect bounds(0,0,cv_frame_resize_pad.cols,cv_frame_resize_pad.rows);
    cv::Rect r(0,12,1920,1080 - 12);
    cv::Mat roi = cv_frame_resize_pad( r & bounds );
    cv::copyMakeBorder(roi, cv_frame_resize_pad, 0, 140, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );   

    // // sensor_msgs::ImagePtr msg_img_left = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_frame_resize_pad).toImageMsg();

    XY_coord xy; 

    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(960, 604), 0, 0, cv::INTER_LINEAR );
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(640, 402), 0, 0, cv::INTER_LINEAR );
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(480, 302), 0, 0, cv::INTER_LINEAR );
    
    // // m_cam_data_left.clear();
    // // m_cam_data_left.header.frame_id = "map";   //"camera_init";
    // // m_cam_data_left.points.resize(msg_img_left->data.size());
    // // cv::namedWindow("My Image");
    // // size_t data_count_cam = 0;
    // // int x,y;
    int u,v;


    // ORG_image_left = temp;

    string myText = "Left";
    cv::Point myPoint;
    myPoint.x = 10;
    myPoint.y = 20;
    cv::putText( ORG_image_left, myText, myPoint, 2, 0.7, Scalar(0, 255, 255));
    
    // // int a = 0;

    // double end = ros::Time::now().toSec();
    // std::cout<<end-start<<std::endl;
    AVM_left = cv::Mat::zeros(400,400, CV_8UC3);
    AVM_left_half = cv::Mat::zeros(400,400, CV_8UC3);
    AVM_left_quarter = cv::Mat::zeros(400,400, CV_8UC3);

    for(int i=0; i< (cv_frame_resize_pad.size().height)  ;i++)
        for(int j=0 ;j< cv_frame_resize_pad.size().width ;j++){
            
    //         // std::cout<<cv_frame_resize_pad.size().height<<cv_frame_resize_pad.size().width<<std::endl;
            // xy = InvProjGRD(4*j,4*i, M_left_param[0], M_left_param[1], M_left_param[2], M_left_param[3], M_left_param[4] ,M_left_param[5], &left_model);
            // xy = InvProjGRD(3*j,3*i, M_left_param[0], M_left_param[1], M_left_param[2], M_left_param[3], M_left_param[4] ,M_left_param[5], &left_model);
            // xy = InvProjGRD(2*j,2*i, M_left_param[0], M_left_param[1], M_left_param[2], M_left_param[3], M_left_param[4] ,M_left_param[5], &left_model);
            xy = InvProjGRD(j,i, M_left_param[0], M_left_param[1], M_left_param[2], M_left_param[3], M_left_param[4] ,M_left_param[5], &left_model);

            // xy.x =0 ;
            // xy.y=0;
            // if((xy.x ==0) && (xy.y ==0)){

            // }
            // else if((abs(xy.x) >=13) || (abs(xy.y) >=13)){

            // }

            // 200 - 10*xy.x
            // AVM.at<char>(j,i) = 
            if(xy.y < 0){
                xy.y =0;
                xy.x =0;
            }

            if( !(xy.x ==0) && (xy.y ==0) || !((abs(xy.x) >=12.5) || (abs(xy.y) >=12.5)) ){
                v = 200 - 16*xy.x;  // width
                u = 200 - 16*xy.y;  // height
                // if ((AVM.at<cv::Vec3b>(int(v), int(u))[0] == 0) && (AVM.at<cv::Vec3b>(int(v), int(u))[1] == 0) && (AVM.at<cv::Vec3b>(int(v), int(u))[2] == 0)){
                AVM_left.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);   //b
                AVM_left.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);   //g
                AVM_left.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);   //r
                // if((i%2==0) && (j%2==0)){
                //     AVM_left_half.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);
                //     AVM_left_half.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);
                //     AVM_left_half.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);
                // }
                // if((i%4==0) && (j%4==0)){
                //     AVM_left_quarter.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);
                //     AVM_left_quarter.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);
                //     AVM_left_quarter.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);
                // }
                // if(((xy.x >= -2.973) && (xy.x < 0) ) && ((xy.y <= 1.034) && (xy.y > 0))  ||   ((xy.x <= 1.905) && (xy.x > 0) ) && ((xy.y <= 1.034)) && (xy.y > 0))     
                // {
                //     AVM_front.at<cv::Vec3b>(int(v), int(u))[0] = 255;
                //     AVM_front.at<cv::Vec3b>(int(v), int(u))[1] = 255;
                //     AVM_front.at<cv::Vec3b>(int(v), int(u))[2] = 0;
                // }
                    // std::cout<<"asdasdsad"<<std::endl;
                // }
            }

            // std::cout << '(' << xy.x << ',' << xy.y << ')'<<std::endl;
            // if( (xy.x ==0) && (xy.y ==0) ){
            //     //0,0 means it is not GRD

            // }
            // else if ((abs(xy.x) >=20) || (abs(xy.y) >=20))
            // {

            // }
            // else{
            //     m_cam_data_left.points[data_count_cam].x = xy.x;
            //     m_cam_data_left.points[data_count_cam].y = xy.y;
            //     m_cam_data_left.points[data_count_cam].z = 0.0;
                
            //     // for(; a< msg_img_center->data.size(); a+=3)
            //     // m_cam_data.points[data_count_cam].g = msg_img_center->data[a];
            //     // m_cam_data.points[data_count_cam].b = msg_img_center-> data[a + 1]; 
            //     // m_cam_data.points[data_count_cam++].r = msg_img_center-> data[a + 2]; 

            //     // at -> (y,x)
            //     m_cam_data_left.points[data_count_cam].g = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);
            //     m_cam_data_left.points[data_count_cam].b = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);
            //     m_cam_data_left.points[data_count_cam].r = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);
                
            //     // Vec3b : CV_8UC3
            //     // if(( abs( m_cam_data_left.points[data_count_cam].x ) > 15 ) || ( abs( m_cam_data_left.points[data_count_cam].y ) > 15 ))
            //     //     m_cam_data_left.points.push_back(m_cam_data_left.points[data_count_cam]);
                
            //     data_count_cam++;

            // }
        }

    // if(!(AVM_right.empty()))
    //     cv::hconcat(AVM_left, AVM_right, AVM_temp);
    // std::cout<<"asdasdsad"<<std::endl;
    // cv::imshow("AVM_left", AVM_left);
    // cv::waitKey(1);

    // sensor_msgs::ImagePtr msg_img_left = cv_bridge::CvImage(std_msgs::Header(), "bgr8", AVM).toImageMsg();
    // Pub_image_left.publish(msg_img_left);

    // double end = ros::Time::now().toSec();
    // std::cout<< "Time : " << end - start << std::endl;
    
    // clock_t end = clock();
    // std::cout<<"Time: "<< (double)(end - start)/CLOCKS_PER_SEC<<std::endl;
    // m_cam_data_left.points.resize(data_count_cam);
    // Pub_cam_left.publish(m_cam_data_left);

    //m_cam_data.points.resize(data_count_cam);

    // imageCallback_front(msg_img_center, mapx_persp_front, mapy_persp_front);
    // Pub_image_center.publish(msg_img_center);

    // cv::waitKey(1);

}
void CallbackPhantom_right(const ocamcalib_undistort::PhantomVisionNetMsg::ConstPtr& msg) 
{
    cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg->image_raw, msg->image_raw.encoding )->image;

    //resize
    // cv::Mat cv_frame_resize;
    cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(1920, 1080), 0, 0, cv::INTER_LINEAR );
    cv::resize( cv_frame_resize_pad, ORG_image_right, cv::Size(320, 200), 0, 0, cv::INTER_LINEAR );
    
    //padding
    // cv::Mat cv_frame_resize_pad;
    cv::Mat temp;

    cv::Rect bounds(0,0,cv_frame_resize_pad.cols,cv_frame_resize_pad.rows);
    cv::Rect r(0,12,1920,1080 - 12);
    cv::Mat roi = cv_frame_resize_pad( r & bounds );
    cv::copyMakeBorder(roi, cv_frame_resize_pad, 0, 140, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );

    // sensor_msgs::ImagePtr msg_img_right = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_frame_resize_pad).toImageMsg();
    
    XY_coord xy; 
    
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(960, 604), 0, 0, cv::INTER_LINEAR );
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(640, 402), 0, 0, cv::INTER_LINEAR );
    // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(480, 302), 0, 0, cv::INTER_LINEAR );
    // m_cam_data_right.clear();
    // m_cam_data_right.header.frame_id = "map";   //"camera_init";
    // m_cam_data_right.points.resize(msg_img_right->data.size());

    // size_t data_count_cam = 0;
    // int x,y;
    int u,v;

    string myText = "Right";
    cv::Point myPoint;
    myPoint.x = 10;
    myPoint.y = 20;
    cv::putText( ORG_image_right, myText, myPoint, 2, 0.7, Scalar(0, 255, 255) );
    int a = 0;


    // double end = ros::Time::now().toSec();
    // std::cout<<end-start<<std::endl;
    AVM_right = cv::Mat::zeros(400,400, CV_8UC3);
    AVM_right_half = cv::Mat::zeros(400,400, CV_8UC3);
    AVM_right_quarter = cv::Mat::zeros(400,400, CV_8UC3);

    for(int i=0; i< (cv_frame_resize_pad.size().height) ;i++)
        for(int j=0 ;j< cv_frame_resize_pad.size().width ;j++){
            
            // std::cout<<cv_frame_resize_pad.size().height<<cv_frame_resize_pad.size().width<<std::endl;

            // xy = InvProjGRD(4*j,4*i, M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4] ,M_right_param[5], &right_model);
            // xy = InvProjGRD(3*j,3*i, M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4] ,M_right_param[5], &right_model);
            // xy = InvProjGRD(2*j,2*i, M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4] ,M_right_param[5], &right_model);
            xy = InvProjGRD(j,i, M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4] ,M_right_param[5], &right_model);

            if(xy.y > 0){
                xy.y =0;
                xy.x =0;
            }
            
            if( !(xy.x ==0) && (xy.y ==0) || !((abs(xy.x) >=12.5) || (abs(xy.y) >=12.5)) ){
                v = 200 - 16*xy.x; 
                u = 200 - 16*xy.y; 

                // if ((AVM.at<cv::Vec3b>(int(v), int(u))[0] == 0) && (AVM.at<cv::Vec3b>(int(v), int(u))[1] == 0) && (AVM.at<cv::Vec3b>(int(v), int(u))[2] == 0)){

                AVM_right.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);   //b
                AVM_right.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);   //g
                AVM_right.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);   //r

                // if((i%2==0) && (j%2==0)){
                //     AVM_right_half.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);
                //     AVM_right_half.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);
                //     AVM_right_half.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);
                // }
                // if((i%4==0) && (j%4==0)){
                //     AVM_right_quarter.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);
                //     AVM_right_quarter.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);
                //     AVM_right_quarter.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);
                // }
                    // std::cout<<"asdasdsad"<<std::endl;
                // }
            }

            // std::cout << '(' << xy.x << ',' << xy.y << ')'<<std::endl;
            // if( (xy.x ==0) && (xy.y ==0) ){
            //     //0,0 means it is not GRD

            // }
            // else if ((abs(xy.x) >=20) || (abs(xy.y) >=20))
            // {

            // }
            // else{
            //     m_cam_data_left.points[data_count_cam].x = xy.x;
            //     m_cam_data_left.points[data_count_cam].y = xy.y;
            //     m_cam_data_left.points[data_count_cam].z = 0.0;
                
            //     // for(; a< msg_img_center->data.size(); a+=3)
            //     // m_cam_data.points[data_count_cam].g = msg_img_center->data[a];
            //     // m_cam_data.points[data_count_cam].b = msg_img_center-> data[a + 1]; 
            //     // m_cam_data.points[data_count_cam++].r = msg_img_center-> data[a + 2]; 

            //     // at -> (y,x)
            //     m_cam_data_left.points[data_count_cam].g = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);
            //     m_cam_data_left.points[data_count_cam].b = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);
            //     m_cam_data_left.points[data_count_cam].r = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);
                
            //     // Vec3b : CV_8UC3
            //     // if(( abs( m_cam_data_left.points[data_count_cam].x ) > 15 ) || ( abs( m_cam_data_left.points[data_count_cam].y ) > 15 ))
            //     //     m_cam_data_left.points.push_back(m_cam_data_left.points[data_count_cam]);
                
            //     data_count_cam++;

            // }
        }
    // m_cam_data_right.points.resize(data_count_cam);
    // Pub_cam_right.publish(m_cam_data_right);
    // std::cout<< "Point : "<< data_count_cam <<std::endl;
    // sensor_msgs::ImagePtr msg_img_right = cv_bridge::CvImage(std_msgs::Header(), "bgr8", AVM_right).toImageMsg();
    // Pub_image_right.publish(msg_img_right);
    // cv::imshow("AVM_right", AVM_right);
    // cv::waitKey(1);
}

int main(int argc, char **argv)
{   
    ros::init(argc, argv, "undistort_node");
    ros::NodeHandle nodeHandle("~");
    auto params = getParameters(nodeHandle); 

    // cv::VideoWriter videoWriter;            //for saving video
    // start =ros::Time::now().toSec(); 
    // if (params.cameraType != "fisheye")
    // {
    //     std::cerr << "Don't support camera type '" << params.cameraType
    //               << "', currently only support 'fisheye'" << std::endl;
    //     return 1;
    // }

    // If starts with a / consider it an absolute path
    // std::string calibration_front = params.calibrationFile[0] == '/' ?
    //                             params.calibrationFile : std::string(ros::package::getPath("ocamcalib_undistort")
    //                                                                  + "/" + std::string(params.calibrationFile));

    std::string calibration_front ="/home/dyros-vehicle/catkin_ws/src/ocamcalib_undistort/include/calib_results_phantom_190_028_front.txt";
    std::string calibration_left = "/home/dyros-vehicle/catkin_ws/src/ocamcalib_undistort/include/calib_results_phantom_190_022_left.txt";
    std::string calibration_right ="/home/dyros-vehicle/catkin_ws/src/ocamcalib_undistort/include/calib_results_phantom_190_023_right.txt";
    std::string calibration_rear = "/home/dyros-vehicle/catkin_ws/src/ocamcalib_undistort/include/calib_results_phantom_190_029_rear.txt" ;        

    // std::cout<<calibration <<std::endl;
    // ocam_model front_model; // our ocam_models for the fisheye and catadioptric cameras
    // ocam_model  left_model;
    // ocam_model right_model;
    // ocam_model  rear_model;

    if(flag == 0)
    {
        if(!get_ocam_model(&front_model, calibration_front.c_str()))
        {
            return 2;
        }
        if(!get_ocam_model(&left_model, calibration_left.c_str()))
        {
            return 2;
        }
        if(!get_ocam_model(&right_model, calibration_right.c_str()))
        {
            return 2;
        }
        if(!get_ocam_model(&rear_model, calibration_rear.c_str()))
        {
            return 2;
        }
        flag =1;
    }

    // double point3D[3]={13,13,0}, point2D[2];
    // world2cam(point2D, point3D ,&front_model);
    // std::cout << point2D[0] << point2D[1] <<std::endl;
    // double point2D[2];
    // double point3D[3]= { -1.26, -4.67, -5.24};
    // world2cam(point2D ,point3D , &front_model);
    // std::cout<<point2D[0] <<"," << point2D[1]<<std::endl ;


    // printOcamModel(front_model);
    // printOcamModel(left_model);
    // printOcamModel(right_model);
    // printOcamModel(rear_model);

    // params.rightBound = params.rightBound == 0 ? front_model.width : params.rightBound;
    // params.bottomBound = params.bottomBound == 0 ? front_model.height : params.bottomBound;

    // CvMat* cmapx_persp_front = cvCreateMat(front_model.height, front_model.width, CV_32FC1);
    // CvMat* cmapy_persp_front = cvCreateMat(front_model.height, front_model.width, CV_32FC1);

    // CvMat* cmapx_persp_left = cvCreateMat(left_model.height, left_model.width, CV_32FC1);
    // CvMat* cmapy_persp_left = cvCreateMat(left_model.height, left_model.width, CV_32FC1);

    // CvMat* cmapx_persp_right = cvCreateMat(right_model.height, right_model.width, CV_32FC1);
    // CvMat* cmapy_persp_right = cvCreateMat(right_model.height, right_model.width, CV_32FC1);

    // CvMat* cmapx_persp_rear = cvCreateMat(rear_model.height, rear_model.width, CV_32FC1);
    // CvMat* cmapy_persp_rear = cvCreateMat(rear_model.height, rear_model.width, CV_32FC1);

    
    // //undistort 
    // create_perspecive_undistortion_LUT(cmapx_persp_front, cmapy_persp_front, &front_model, params.scaleFactor);
    // create_perspecive_undistortion_LUT(cmapx_persp_left, cmapy_persp_left, &left_model, params.scaleFactor);
    // create_perspecive_undistortion_LUT(cmapx_persp_right, cmapy_persp_right, &right_model, params.scaleFactor);
    // create_perspecive_undistortion_LUT(cmapx_persp_rear, cmapy_persp_rear, &rear_model, params.scaleFactor);

    
    // Need to convert to C++ style to play nice with ROS
    // mapx_persp_front = cv::cvarrToMat(cmapx_persp_front);
    // mapy_persp_front = cv::cvarrToMat(cmapy_persp_front);

    // mapx_persp_left = cv::cvarrToMat(cmapx_persp_left);
    // mapy_persp_left = cv::cvarrToMat(cmapy_persp_left);

    // mapx_persp_right = cv::cvarrToMat(cmapx_persp_right);
    // mapy_persp_right = cv::cvarrToMat(cmapy_persp_right);

    // mapx_persp_rear = cv::cvarrToMat(cmapx_persp_rear);
    // mapy_persp_rear = cv::cvarrToMat(cmapy_persp_rear);

    // image_transport::ImageTransport img_transport(nodeHandle);
    
    // cv::Rect roi(params.leftBound,
    //              params.topBound,
    //              params.rightBound-params.leftBound,
    //              params.bottomBound-params.topBound);

    //std::cout << "asdf " <<std::endl;
                                
    // auto publisher = img_transport.advertise(params.outTopic, 1);
    // ros::Subscriber subscriber = img_transport.subscribe("camera/image_center", 1,
    //                         [&](const sensor_msgs::ImageConstPtr& msg)
    //                         {
    //                             std::cout << "asdf " <<std::endl;
    //                             imageCallback(msg, mapx_persp, mapy_persp, roi, publisher);
    //                         },
    //                         ros::VoidPtr(),
    //                         image_transport::TransportHints(params.transportHint));
    
    //////////////////////////
    
    // ros::Subscriber subscriber = nodeHandle.subscribe("/camera/image_left", 1 , callback_image);
    // ros::Subscriber subscriber = nodeHandle.subscribe("/camera/image_left", 1 , callback_image);
    // ros::Subscriber subscriber = nodeHandle.subscribe("/camera/image_left", 1 , callback_image);
    // ros::Subscriber subscriber = nodeHandle.subscribe("/camera/image_left", 1 , callback_image);

    // Pub_image_center = nodeHandle.advertise<sensor_msgs::Image>("camera/image_center", 1);
    // Pub_image_rear = nodeHandle.advertise<sensor_msgs::Image>("camera/image_rear", 1);
    // Pub_image_left = nodeHandle.advertise<sensor_msgs::Image>("camera/image_left", 1);
    // Pub_image_right = nodeHandle.advertise<sensor_msgs::Image>("camera/image_right", 1);


    ros::Subscriber Sub_phantom_front_center_svm = nodeHandle.subscribe("/phantomnet/output/front_center_svm", 1 , CallbackPhantom_center);
    ros::Subscriber Sub_phantom_rear_center_svm = nodeHandle.subscribe("/phantomnet/output/rear_center_svm", 1, CallbackPhantom_rear);
    ros::Subscriber Sub_phantom_side_left = nodeHandle.subscribe("/phantomnet/output/side_left", 1, CallbackPhantom_left);
    ros::Subscriber Sub_phantom_side_right = nodeHandle.subscribe("/phantomnet/output/side_right", 1, CallbackPhantom_right);

    ros::Subscriber Sub_phantom_front_seg = nodeHandle.subscribe("/phantomnet/output/front_center_svm", 1 , CallbackPhantom_seg_front);
    ros::Subscriber Sub_phantom_rear_seg = nodeHandle.subscribe("/phantomnet/output/rear_center_svm", 1 , CallbackPhantom_seg_rear);
    ros::Subscriber Sub_phantom_left_seg = nodeHandle.subscribe("/phantomnet/output/side_left", 1 , CallbackPhantom_seg_left);
    ros::Subscriber Sub_phantom_right_seg = nodeHandle.subscribe("/phantomnet/output/side_right", 1 , CallbackPhantom_seg_right);

    ros::Subscriber Sub_phantom_DR_Path = nodeHandle.subscribe("/LocalizationData", 1 , CallbackPhantom_DR);

    // Pub_AVM_full_img = nodeHandle.advertise<sensor_msgs::Image>("/AVM_image_full", 1);
    // Pub_AVM_side_img = nodeHandle.advertise<sensor_msgs::Image>("/AVM_image_side", 1);
    // Pub_AVM_center_img = nodeHandle.advertise<sensor_msgs::Image>("/AVM_image_center", 1);
    
    // Pub_image = nodeHandle.advertise<sensor_msgs::Image>("/ocamcalib_undistorted_front", 1);
    // Pub_image_center = nodeHandle.advertise<sensor_msgs::Image>("/ocamcalib_undistorted_front", 1);
    // Pub_image_rear = nodeHandle.advertise<sensor_msgs::Image>("/ocamcalib_undistorted_rear", 1);
    // Pub_image_left = nodeHandle.advertise<sensor_msgs::Image>("/ocamcalib_undistorted_left", 1);
    // Pub_image_right = nodeHandle.advertise<sensor_msgs::Image>("/ocamcalib_undistorted_right", 1);

    Pub_avm = nodeHandle.advertise<VPointImage>("grid_data_cam_left", 1);

    // m_cam_data.clear();

    // Pub_cam_left = nodeHandle.advertise<VPointImage>("grid_data_cam_left", 1);
    // Pub_cam_right = nodeHandle.advertise<VPointImage>("grid_data_cam_right", 1);

    // ros::spin();

    // float videoFPS = 10;
    // int videoWidth = 400;
	// int videoHeight = 400;
    
    // ros::Rate loop_rate(10); 
    // videoWriter.open("/home/Desktop/out.mp4", 0X21, 
	// 	videoFPS , cv::Size(videoWidth, videoHeight), true);

    // cv::namedWindow("video");
    idx = 0;
    while(ros::ok())        //half -> 0.03
    { 
        cv::waitKey(50);
        ros::AsyncSpinner spinner(8); // Use 4 threads
        spinner.start();
    
        // double start =ros::Time::now().toSec();
        // if( !(AVM_front.empty()) &&  !(AVM_right.empty()) &&  !(AVM_left.empty())  &&  !(AVM_rear.empty()))

        // std::cout<<"11111111111111111111111111111"<<std::endl;
        
        // cv::addWeighted(AVM_front, 1, AVM_rear, 1, 0.0, result);


        result1 = AVM_front + AVM_right + AVM_left + AVM_rear;
        full = result1;
        half = AVM_front_half + AVM_right_half + AVM_left_half + AVM_rear_half; 
        quarter = AVM_front_quarter + AVM_right_quarter + AVM_left_quarter + AVM_rear_quarter; 

        result2 = AVM_front + AVM_rear;
        full_cen = result2;
        half_cen = AVM_front_half + AVM_rear_half;
        quarter_cen = AVM_front_quarter + AVM_rear_quarter;

        result3 = AVM_right + AVM_left;
        full_side = result3;
        half_side = AVM_right_half + AVM_left_half;
        quarter_side = AVM_right_quarter + AVM_left_quarter;

        
        // rectangle(result1, Point(179, 161.9), Point(220.68, 259.46), Scalar(0, 0, 255), 2);
        // rectangle(result2, Point(179, 161.9), Point(220.68, 259.46), Scalar(0, 0, 255), 2);
        // rectangle(result3, Point(179, 161.9), Point(220.68, 259.46), Scalar(0, 0, 255), 2);
        
        // cv::hconcat(result1, result2, result1);
        // cv::hconcat(result1, result3, result1);

        // videoWriter << result1;
        // cv::imshow("result1", full_side);


        // int keycode = waitKey(); 
        // if(keycode == 'c')    // space bar button
        // {   
        //     std::cout <<"press center" <<std::endl;

        //     sprintf(buf2, "/home/dyros-vehicle/Desktop/AVM_IMG/full_resolution/Center/AVM_Center%05d.jpg",idx);
        //     sprintf(buf5, "/home/dyros-vehicle/Desktop/AVM_IMG/half_resolution/Center/AVM_Center%05d.jpg",idx);
        //     sprintf(buf8, "/home/dyros-vehicle/Desktop/AVM_IMG/quarter_resolution/Center/AVM_Center%05d.jpg",idx);
        //     imwrite(buf2, result2);
        //     imwrite(buf5, half_cen);
        //     imwrite(buf8, quarter_cen);

        //     idx++;
        // }
        // else if(keycode == 's'){

        //     std::cout <<"press side" <<std::endl;

        //     sprintf(buf3, "/home/dyros-vehicle/Desktop/AVM_IMG/full_resolution/Side/AVM_Side%05d.jpg",idx);
        //     sprintf(buf6, "/home/dyros-vehicle/Desktop/AVM_IMG/half_resolution/Side/AVM_Side%05d.jpg",idx);
        //     sprintf(buf9, "/home/dyros-vehicle/Desktop/AVM_IMG/quarter_resolution/Side/AVM_Side%05d.jpg",idx);
        //     imwrite(buf3, result3);
        //     imwrite(buf6, half_side);
        //     imwrite(buf9, quarter_side);

        //     idx++;
        // }
        // else if(keycode == 'f'){

        //     std::cout <<"press full" <<std::endl;
            
        //     sprintf(buf,  "/home/dyros-vehicle/Desktop/AVM_IMG/full_resolution/Sum/AVM_Full%05d.jpg",idx); 
        //     sprintf(buf4, "/home/dyros-vehicle/Desktop/AVM_IMG/half_resolution/Sum/AVM_Full%05d.jpg",idx);
        //     sprintf(buf7, "/home/dyros-vehicle/Desktop/AVM_IMG/quarter_resolution/Sum/AVM_Full%05d.jpg",idx); 
        //     imwrite(buf,  full);
        //     imwrite(buf4, half);
        //     imwrite(buf7, quarter);

        //     idx++;
        // }



        //publish AVM image
        // if(!(result1.empty())){
        //     sensor_msgs::ImagePtr msg_avm_full_img = cv_bridge::CvImage(std_msgs::Header(), "bgr8", full).toImageMsg();
        //     sensor_msgs::ImagePtr msg_avm_left_right_img = cv_bridge::CvImage(std_msgs::Header(), "bgr8", result3).toImageMsg();
        //     sensor_msgs::ImagePtr msg_avm_front_rear_img = cv_bridge::CvImage(std_msgs::Header(), "bgr8", result2).toImageMsg();

        //     Pub_AVM_full_img.publish(msg_avm_full_img);
        //     Pub_AVM_side_img.publish(msg_avm_left_right_img);
        //     Pub_AVM_center_img.publish(msg_avm_front_rear_img);
        // }

        // rectangle(AVM_front, Point(179, 161.9), Point(220.68, 259.46), Scalar(0, 0, 255), 2);
        // rectangle(AVM_rear, Point(179, 161.9), Point(220.68, 259.46), Scalar(0, 0, 255), 2);
        // rectangle(AVM_left, Point(179, 161.9), Point(220.68, 259.46), Scalar(0, 0, 255), 2);
        // rectangle(AVM_right, Point(179, 161.9), Point(220.68, 259.46), Scalar(0, 0, 255), 2);

        // cv::imshow("front", AVM_front);
        // cv::imshow("rear", AVM_rear);
        // cv::imshow("left", AVM_left);
        // cv::imshow("right", AVM_right);      
        // cv::imshow("full_avm", result1);
        // cv::imshow("front_rear", result2);
        // cv::imshow("left_right", result3);       
        // rectangle(result1, Point(179, 161.9), Point(220.68, 259.46), Scalar(0, 0, 255), 2); 
        // circle(result, Point(200, 200), 3, Scalar(255, 0, 255), 2, 1, 0);        
        // cv::imshow("concat_img",result);
        // setMouseCallback("front", onMouse, 0);
        // setMouseCallback("rear", onMouse, 0);
        // setMouseCallback("left", onMouse, 0);
        // setMouseCallback("right", onMouse, 0);
        // waitKey(1);

        result4 = AVM_seg_front + AVM_seg_right + AVM_seg_left + AVM_seg_rear;
        // result5 = AVM_seg_front + AVM_seg_rear;
        // result6 = AVM_seg_right + AVM_seg_left;
        // rectangle(result4, Point(179, 161.9), Point(220.68, 259.46), Scalar(0, 0, 255), 2); 
        // rectangle(result5, Point(179, 161.9), Point(220.68, 259.46), Scalar(0, 0, 255), 2); 
        // rectangle(result6, Point(179, 161.9), Point(220.68, 259.46), Scalar(0, 0, 255), 2); 
        // cv::hconcat(result4, result5, result4);
        // cv::hconcat(result4, result6, result4);
        // cv::imshow("concat_img2",result_seg);
        // cv::vconcat(result,result_seg,temp);
        // rectangle(result_seg, Point(179, 161.9), Point(220.68, 259.46), Scalar(0, 0, 255), 2); 
        
        // cv::imshow("result",result_seg);

        // std::cout<<result1.size()<<std::endl;
        // std::cout<<result4.size()<<std::endl;


        // cv::vconcat(result1,result4,result1);

        //org
        // cv::vconcat(ORG_image_front, ORG_image_rear,  ORG);
        // cv::vconcat(ORG, ORG_image_left,  ORG);
        // cv::vconcat(ORG, ORG_image_right, ORG);
        
        // cv::vconcat(ORG_image_front, ORG_image_left, ORG_image_front);
        // cv::imshow("ORG_image", ORG);

        // cv::imshow("result",temp);
        // cv::waitKey(1); 

        // cv::hconcat(ORG2, temp, temp);     //#############problem

        // cv::imshow("ORG2",ORG2);
        // cv::imshow("temp",result1);


        // if (ORG.size().height == result1.size().height) {
        //     cv::hconcat(ORG, result1,  temp2);
        // // std::cout<<ORG.size()<<std::endl;
        // // std::cout<<result1.size()<<std::endl;
        //     cv::imshow("temp",temp2);
        //     cv::waitKey(1);
        // }
        
        // cv::hconcat(ORG2, temp, temp2);
         
        // ros::spinOnce();
// ros::waitForShutdown();

        // loop_rate.sleep();


        // double end = ros::Time::now().toSec();
        // std::cout<< "Time : " << end - start << std::endl;
    }
    // ros::waitForShutdown();
    // while (ros::ok())
    // {
    //     if( !(AVM_front.empty()) && !(AVM_right.empty()) && !(AVM_left.empty()) && !(AVM_rear.empty()) )
    //     {
    //         // cv::addWeighted(AVM_front, 1, AVM_rear, 1, 0.0, result);
    //         result1 = AVM_front + AVM_right + AVM_left + AVM_rear;
    //         result2 = AVM_front + AVM_rear;
    //         result3 = AVM_right + AVM_left;
    //         cv::imshow("full_avm", result1);
    //         cv::imshow("front_rear", result2);
    //         cv::imshow("left_right", result3);
    //         cv::waitKey(1);
    //     }
    //     ros::spinOnce();
    // }
    

    // cvReleaseMat(&cmapx_persp_front);
    // cvReleaseMat(&cmapy_persp_front);
    // cvReleaseMat(&cmapx_persp_left);
    // cvReleaseMat(&cmapy_persp_left);
    // cvReleaseMat(&cmapx_persp_right);
    // cvReleaseMat(&cmapy_persp_right);
    // cvReleaseMat(&cmapx_persp_rear);
    // cvReleaseMat(&cmapy_persp_rear);

    // cv::destroyWindow("video");

    return 0;
}
