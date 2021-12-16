#include "ocamcalib_undistort/ocam_functions.h"
#include "ocamcalib_undistort/Parameters.h"
 
#include <iostream>
#include <string>
#include <exception>
#include <ros/ros.h>
#include <ros/package.h>
#include <nav_msgs/OccupancyGrid.h>

#include <cv_bridge/cv_bridge.h>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"

// #include <ocamcalib_undistort/PhantomVisionNetMsg.h> 
// #include <ocamcalib_undistort/VisionPhantomnetData.h>
// #include <ocamcalib_undistort/VisionPhantomnetDataList.h>

#include <std_msgs/Float32MultiArray.h>
#include <pcl_ros/point_cloud.h>
#include <algorithm>

using namespace cv;
using namespace std;

//선택지 6개
#define REOLUTION_MODE 1    //full : 0, half = 1, quarter = 2
#define DIRECTION 0         //side : 0, center = 1, both : 2
#define SEG_IMG_PUB 0       // 1: true, 2: false

#define AVM_IMG_WIDTH 300
#define AVM_IMG_HEIGHT 300
// #define CROP_TOP 12
#define PADDING_CENTER_BOTTOM 128
#define PADDING_REAR_BOTTOM 128
#define PADDING_LEFT_BOTTOM 100
#define PADDING_RIGHT_BOTTOM 100

#define FULL_IMG_RESOL_WIDTH 1920
#define FULL_IMG_RESOL_HEIGHT 1208
#define CROP_ROI_WIDTH 1920
#define CROP_ROI_HEIGHT 1080

#define REAL_OCCUPANCY_SIZE_X 21    // AVM_IMG_WIDTH 400PIX == 25Meter
#define REAL_OCCUPANCY_SIZE_Y 21    // AVM_IMG_HEIGHT 400PIX == 25Meter

#define PIXEL_PER_METER AVM_IMG_WIDTH/REAL_OCCUPANCY_SIZE_X           //400PIX / 25M

ros::Subscriber sub_tmp;
ros::Subscriber Sub_phantom_side_left, Sub_phantom_side_right, Sub_phantom_front_center_svm, Sub_phantom_rear_center_svm;
ros::Subscriber Sub_phantom_left_seg, Sub_phantom_right_seg, Sub_phantom_front_seg, Sub_phantom_rear_seg;
ros::Subscriber Sub_phantom_DR_Path;

ros::Publisher Pub_AVM_img, Pub_AVM_seg_img, Pub_AVM_seg_img_gray, Pub_AVM_DR;

cv::Mat AVM_left = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);// = cv::Mat::zeros(450,450, CV_8UC3);
cv::Mat AVM_right = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);// = cv::Mat::zeros(450,450, CV_8UC3);
cv::Mat AVM_front = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);
cv::Mat AVM_rear = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);

cv::Mat AVM_seg_front = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);
cv::Mat AVM_seg_rear = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);
cv::Mat AVM_seg_left = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);
cv::Mat AVM_seg_right = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);
cv::Mat AVM_seg_left_gray = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC1);
cv::Mat AVM_seg_right_gray = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC1);

cv::Mat aggregated_img    = cv::Mat::zeros(AVM_IMG_WIDTH, AVM_IMG_HEIGHT , CV_8UC3);
cv::Mat aggregated_seg_img = cv::Mat::zeros(AVM_IMG_WIDTH, AVM_IMG_HEIGHT , CV_8UC3);
cv::Mat aggregated_seg_img_gray = cv::Mat::zeros(AVM_IMG_WIDTH, AVM_IMG_HEIGHT , CV_8UC1);

// Calibration Results
ocam_model front_model;
ocam_model  left_model;
ocam_model right_model;
ocam_model  rear_model;

#define M_DEG2RAD  3.1415926 / 180.0

// // Extrinsic Parameters
// double M_front_param[6] = {0.688 * M_DEG2RAD,  21.631 * M_DEG2RAD,   3.103* M_DEG2RAD   ,1.905,   0.033, 0.707 };
// double M_left_param[6] =  {1.133 * M_DEG2RAD,  19.535 * M_DEG2RAD,   92.160* M_DEG2RAD  ,0.0,     1.034, 0.974 };
// double M_right_param[6] = {3.440 * M_DEG2RAD,  18.273 * M_DEG2RAD,  -86.127* M_DEG2RAD  ,0.0,    -1.034, 0.988 };
// double M_back_param[6] =  {0.752 * M_DEG2RAD,  31.238 * M_DEG2RAD,  -178.189* M_DEG2RAD ,-2.973, -0.065, 0.883 };

// // New Extrinsic Parameters 15 mm --> more corect than 9 mm
double M_front_param[6] = {0.672 * M_DEG2RAD,  21.378 * M_DEG2RAD,   1.462* M_DEG2RAD   ,   1.885,   0.038, 0.686 };
double M_left_param[6] =  {0.963 * M_DEG2RAD,  19.283 * M_DEG2RAD,   91.702* M_DEG2RAD  ,   0.0,    1.059, 0.978 };
double M_right_param[6] = {1.714 * M_DEG2RAD,  19.713 * M_DEG2RAD,  -87.631* M_DEG2RAD  ,   0.0,    -1.059, 0.972 };
double M_back_param[6] =  {-0.257 * M_DEG2RAD, 32.645 * M_DEG2RAD,  179.773* M_DEG2RAD ,   -3.002, -0.033, 0.922 };

// // New Extrinsic Parameters 9 mm
// double M_front_param[6] = {0.617 * M_DEG2RAD,  21.397 * M_DEG2RAD,   1.381* M_DEG2RAD   ,   1.880,   0.038, 0.689 };
// double M_left_param[6] =  {0.970 * M_DEG2RAD,  19.231 * M_DEG2RAD,   91.699* M_DEG2RAD  ,   0.0,    1.053, 0.979 };
// double M_right_param[6] = {1.659 * M_DEG2RAD,  19.690 * M_DEG2RAD,  -87.631* M_DEG2RAD  ,   0.0,    -1.053, 0.979 };
// double M_back_param[6] =  {-0.150 * M_DEG2RAD, 32.634 * M_DEG2RAD,  179.708* M_DEG2RAD ,   -2.997, -0.033, 0.924 };

int flag = 0; 
int resolution = 1;

// occupancy grid map for path planning
nav_msgs::OccupancyGrid occupancyGridMap;
ros::Publisher Pub_occupancyGridMap;
int m_dimension = 35;
double m_gridResol = 0.25;
const int m_gridDim = (int)(m_dimension*(int)(1/m_gridResol));
int num_obsL[140][140] = {{0,}, {0,}};
int num_obsR[140][140] = {{0,}, {0,}};

// for checking DR error
struct CARPOSE {
    double x,y,th,vel;
};CARPOSE m_car;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_avm_data(new pcl::PointCloud<pcl::PointXYZRGB>);
bool m_flagDR = false;
unsigned int m_avmDRsize = 0, m_DRcnt = 0;
unsigned int CNTforIMAGE_save = 172, CNTforIMAGE = 0;

void seg2rgb(cv::Mat input_img, cv::Mat& output_img, cv::Mat& output_img_gray) {
    for (int i = 0 ; i < input_img.rows ; i++) {
        for (int j = 0 ; j < input_img.cols ; j++) {
            // if ((int)input_img.at<uchar>(i, j) != 0 && 
            //     (int)input_img.at<uchar>(i, j) != 1 && 
            //     (int)input_img.at<uchar>(i, j) != 2 && 
            //     (int)input_img.at<uchar>(i, j) != 3 &&
            //     (int)input_img.at<uchar>(i, j) != 4 && 
            //     (int)input_img.at<uchar>(i, j) != 5 &&
            //     (int)input_img.at<uchar>(i, j) != 6 &&
            //     (int)input_img.at<uchar>(i, j) != 9 &&
            //     (int)input_img.at<uchar>(i, j) != 10 &&
            //     (int)input_img.at<uchar>(i, j) != 13 &&
            //     (int)input_img.at<uchar>(i, j) != 14 &&
            //     (int)input_img.at<uchar>(i, j) != 15)
            //     cout << (int)input_img.at<uchar>(i, j) << " " ;

            if ((int)input_img.at<uchar>(i, j) == 1 || (int)input_img.at<uchar>(i, j) == 2)
                output_img_gray.at<uchar>(i, j) = 255;
            else
                output_img_gray.at<uchar>(i, j) = 0;

            switch((int)input_img.at<uchar>(i, j)){
                case 0 : // 
                    output_img.at<Vec3b>(i, j)[0] = 78;
                    output_img.at<Vec3b>(i, j)[1] = 56;
                    output_img.at<Vec3b>(i, j)[2] = 24;
                break; 
                case 1 : // vehicle
                    output_img.at<Vec3b>(i, j)[0] = 0;
                    output_img.at<Vec3b>(i, j)[1] = 0;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                case 2 : // wheel
                    output_img.at<Vec3b>(i, j)[0] = 0;
                    output_img.at<Vec3b>(i, j)[1] = 255;
                    output_img.at<Vec3b>(i, j)[2] = 123;
                break;  
                case 3 : 
                    output_img.at<Vec3b>(i, j)[0] = 255;
                    output_img.at<Vec3b>(i, j)[1] = 0;
                    output_img.at<Vec3b>(i, j)[2] = 0;
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
                case 9 : // human
                    output_img.at<Vec3b>(i, j)[0] = 255;
                    output_img.at<Vec3b>(i, j)[1] = 255;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                case 10 : 
                    output_img.at<Vec3b>(i, j)[0] = 35;
                    output_img.at<Vec3b>(i, j)[1] = 111;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                case 13 :    // road 
                    output_img.at<Vec3b>(i, j)[0] = 0;
                    output_img.at<Vec3b>(i, j)[1] = 255;
                    output_img.at<Vec3b>(i, j)[2] = 0;
                break;
                case 14 : 
                    output_img.at<Vec3b>(i, j)[0] = 0;
                    output_img.at<Vec3b>(i, j)[1] = 165;
                    output_img.at<Vec3b>(i, j)[2] = 255;
                break;  
                case 15 : // cart?
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
    // cout << endl;
}


void real2arr(double recvX, double recvY, int& outX, int& outY) {
    outX = (m_gridDim/2.0) + recvX / m_gridResol;
    outY = (m_gridDim/2.0) + recvY / m_gridResol;
}

// void CallbackPhantom_center(const ocamcalib_undistort::PhantomVisionNetMsg::ConstPtr& msg) 
// {   
//     if( DIRECTION != 0){
//         cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg->image_raw, msg->image_raw.encoding )->image;

//         //resize, bagfile_size: (512, 288) * 3.75 => (1920, 1080)
//         cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(CROP_ROI_WIDTH, CROP_ROI_HEIGHT), 0, 0, cv::INTER_LINEAR );
//         // padding
//         cv::Mat temp;
        
//         cv::copyMakeBorder(cv_frame_resize_pad, cv_frame_resize_pad, 0, PADDING_CENTER_BOTTOM, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );

//         XY_coord xy; 
//         if(REOLUTION_MODE == 1){       //half_resolution
//             cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(FULL_IMG_RESOL_WIDTH / 2.0, FULL_IMG_RESOL_HEIGHT /2.0), 0, 0, cv::INTER_LINEAR );
//             resolution = 2;
//         }
//         else if(REOLUTION_MODE == 2){  //quarter_resolution
//             cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(FULL_IMG_RESOL_WIDTH / 4.0, FULL_IMG_RESOL_HEIGHT / 4.0), 0, 0, cv::INTER_LINEAR );
//             resolution = 4;
//         }
//         int u,v;

//         AVM_front = cv::Mat::zeros(AVM_IMG_WIDTH, AVM_IMG_HEIGHT, CV_8UC3);

//         for(int i=0; i< cv_frame_resize_pad.size().height  ;i++)
//             for(int j=0 ;j< cv_frame_resize_pad.size().width ;j++){
            
//                 // Image_pixel to World_Coordinate (x (meter),y (meter))
//                 xy = InvProjGRD(resolution * j, resolution * i, M_front_param[0], M_front_param[1], M_front_param[2], M_front_param[3], M_front_param[4] ,M_front_param[5], &front_model);

//                 if( !(xy.x ==0) && (xy.y ==0) || !((abs(xy.x) >=(REAL_OCCUPANCY_SIZE_X / 2.0)) || (abs(xy.y) >= (REAL_OCCUPANCY_SIZE_Y / 2.0))) ){
//                     v = (AVM_IMG_WIDTH / 2.0) - PIXEL_PER_METER*xy.x;  // AVM image width  (MAX : 400)
//                     u = (AVM_IMG_HEIGHT / 2.0) - PIXEL_PER_METER*xy.y;  // AVM image height (Max : 400)

//                     AVM_front.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);   //b
//                     AVM_front.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);   //g
//                     AVM_front.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);   //r
//                 }   
//             }
//     }
// }
// void CallbackPhantom_rear(const ocamcalib_undistort::PhantomVisionNetMsg::ConstPtr& msg) 
// {
//     if( DIRECTION != 0){
//         cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg->image_raw, msg->image_raw.encoding )->image;
//         //resize
//         cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(CROP_ROI_WIDTH, CROP_ROI_HEIGHT), 0, 0, cv::INTER_LINEAR );

//         //padding
//         cv::Mat temp;
//         cv::copyMakeBorder(cv_frame_resize_pad, cv_frame_resize_pad, 0, PADDING_REAR_BOTTOM, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );

//         XY_coord xy;
//         if(REOLUTION_MODE == 1){       //half_resolution
//             cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(FULL_IMG_RESOL_WIDTH / 2.0, FULL_IMG_RESOL_HEIGHT /2.0), 0, 0, cv::INTER_LINEAR );
//             resolution = 2;
//         }
//         else if(REOLUTION_MODE == 2){  //quarter_resolution
//             cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(FULL_IMG_RESOL_WIDTH / 4.0, FULL_IMG_RESOL_HEIGHT /4.0), 0, 0, cv::INTER_LINEAR );
//             resolution = 4;
//         }
//         int u,v;
        
//         AVM_rear = cv::Mat::zeros(AVM_IMG_WIDTH, AVM_IMG_HEIGHT, CV_8UC3);

//         for(int i=0; i< cv_frame_resize_pad.size().height  ;i++)
//             for(int j=0 ;j< cv_frame_resize_pad.size().width ;j++){
//                 xy = InvProjGRD(resolution * j,resolution * i, M_back_param[0], M_back_param[1], M_back_param[2], M_back_param[3], M_back_param[4] ,M_back_param[5], &rear_model);
            
//                 if( !(xy.x ==0) && (xy.y ==0) || !((abs(xy.x) >= (REAL_OCCUPANCY_SIZE_X / 2.0)) || (abs(xy.y) >=(REAL_OCCUPANCY_SIZE_Y / 2.0))) ){
//                     v = (AVM_IMG_WIDTH / 2.0) - PIXEL_PER_METER*xy.x;   // AVM image width  (MAX : 400) 
//                     u = (AVM_IMG_HEIGHT / 2.0) - PIXEL_PER_METER*xy.y;  // AVM image height (Max : 400)

//                     AVM_rear.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);   //b
//                     AVM_rear.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);   //g
//                     AVM_rear.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);   //r
//                 }
//             }
//     }
// }

// void CallbackPhantom_left(const ocamcalib_undistort::PhantomVisionNetMsg::ConstPtr& msg) 
// void CallbackPhantom_left(const ocamcalib_undistort::VisionPhantomnetData::ConstPtr& msg) 
void CallbackPhantom_left(const sensor_msgs::ImageConstPtr& msg) 
{
   if( DIRECTION != 1){
        // cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg->image_raw, msg->image_raw.encoding )->image;
        // cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg->image_viz, msg->image_viz.encoding )->image;
        cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy(msg, "bgr8" )->image;
        
        // cv::imshow("left", cv_frame_resize_pad);
        // cv::waitKey(1);
        
        // //resize
        cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(CROP_ROI_WIDTH, CROP_ROI_HEIGHT), 0, 0, cv::INTER_LINEAR );
    
        // //padding
        cv::Mat temp;
        // cv::Rect bounds(0,0,cv_frame_resize_pad.cols,cv_frame_resize_pad.rows);
        // cv::Rect r(0,12,1920,1080 - 12);    //
        // cv::Rect r(0,100,1920,1080 - 100);    //
        // cv::Mat roi = cv_frame_resize_pad( r & bounds );
        // cv::copyMakeBorder(roi, cv_frame_resize_pad, 0, 100, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        // cv::copyMakeBorder(cv_frame_resize_pad, cv_frame_resize_pad, 0, PADDING_LEFT_BOTTOM, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        cv::copyMakeBorder(cv_frame_resize_pad, cv_frame_resize_pad, PADDING_LEFT_BOTTOM, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        // cv::copyMakeBorder(cv_frame_resize_pad, cv_frame_resize_pad, 0, 0, 0, 100, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
// bottom

        XY_coord xy;

        if(REOLUTION_MODE == 1){       //half_resolution
            cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(FULL_IMG_RESOL_WIDTH / 2.0, FULL_IMG_RESOL_HEIGHT /2.0), 0, 0, cv::INTER_LINEAR );
            resolution = 2;
        }
        else if(REOLUTION_MODE == 2){  //quarter_resolution
            cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(FULL_IMG_RESOL_WIDTH / 4.0, FULL_IMG_RESOL_HEIGHT /4.0), 0, 0, cv::INTER_LINEAR );
            resolution = 4;
        }
        int u,v;

        AVM_left = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);


        for(int i=1; i< cv_frame_resize_pad.size().height  ;i++)
            for(int j=0 ;j< cv_frame_resize_pad.size().width ;j++){
                xy = InvProjGRD(resolution * j,resolution * i, M_left_param[0], M_left_param[1], M_left_param[2], M_left_param[3], M_left_param[4] ,M_left_param[5], &left_model);


                if( !(xy.x ==0) && (xy.y ==0) || !((abs(xy.x) >= (REAL_OCCUPANCY_SIZE_X / 2.0)) || (abs(xy.y) >= (REAL_OCCUPANCY_SIZE_Y / 2.0))) ){
                    v = (AVM_IMG_WIDTH / 2.0) - PIXEL_PER_METER*xy.x;  // AVM image width  (MAX : 400)
                    u = (AVM_IMG_HEIGHT / 2.0) - PIXEL_PER_METER*xy.y; // AVM image height (Max : 400)
                
                    AVM_left.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);   //b
                    AVM_left.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);   //g
                    AVM_left.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);   //r
                }
            }
    }
}

void CallbackPhantom_seg_left(const ocamcalib_undistort::PhantomVisionNetMsg::ConstPtr& msg){
    if( DIRECTION != 1 && SEG_IMG_PUB){
        cv::Mat cv_frame_seg = cv_bridge::toCvCopy( msg->image_seg, msg->image_seg.encoding )->image;
        cv::Mat cv_frame_raw_new = cv_bridge::toCvCopy( msg->image_raw, msg->image_raw.encoding )->image;
        cv::Mat cv_frame_raw_new_gray = cv_bridge::toCvCopy( msg->image_raw, msg->image_seg.encoding )->image;
        
        seg2rgb(cv_frame_seg, cv_frame_raw_new, cv_frame_raw_new_gray);
        
        //resize
        cv::Mat cv_frame_resize, cv_frame_resize_gray;
        cv::resize( cv_frame_raw_new, cv_frame_resize, Size(1920, 1080), 0, 0, INTER_LINEAR );
        cv::resize( cv_frame_raw_new_gray, cv_frame_resize_gray, Size(1920, 1080), 0, 0, INTER_LINEAR );

        cv::Mat cv_frame_resize_pad, cv_frame_resize_pad_gray;
        cv::copyMakeBorder(cv_frame_resize, cv_frame_resize_pad, 0, 128, 0, 0, BORDER_CONSTANT, Scalar(0,0,0) );
        cv::copyMakeBorder(cv_frame_resize_gray, cv_frame_resize_pad_gray, 0, 128, 0, 0, BORDER_CONSTANT, Scalar(0,0,0) );

        XY_coord xy;
        
        if(REOLUTION_MODE == 1){       //half_resolution
            cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(960, 604), 0, 0, cv::INTER_LINEAR );
            cv::resize( cv_frame_resize_pad_gray, cv_frame_resize_pad_gray, cv::Size(960, 604), 0, 0, cv::INTER_LINEAR );
            resolution = 2;
        }
        else if(REOLUTION_MODE == 2){  //quarter_resolution
            cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(480, 302), 0, 0, cv::INTER_LINEAR );
            cv::resize( cv_frame_resize_pad_gray, cv_frame_resize_pad_gray, cv::Size(480, 302), 0, 0, cv::INTER_LINEAR );
            resolution = 4;
        }
        int u,v;

        AVM_seg_left = cv::Mat::zeros(400,400, CV_8UC3);
        AVM_seg_left_gray = cv::Mat::zeros(400,400, CV_8UC1);
        
        for(int i=0; i < m_gridDim ; i++) for(int j=0 ;j < m_gridDim ;j++) num_obsL[j][i] = 0;

        for(int i=0; i< cv_frame_resize_pad.size().height ;i++)
            for(int j=0 ;j< cv_frame_resize_pad.size().width ;j++){
                xy = InvProjGRD(resolution * j, resolution * i, M_left_param[0], M_left_param[1], M_left_param[2], M_left_param[3], M_left_param[4] ,M_left_param[5], &left_model);

                if( !(xy.x ==0) && (xy.y ==0) || !((abs(xy.x) >=(REAL_OCCUPANCY_SIZE_X / 2.0)) || (abs(xy.y) >=(REAL_OCCUPANCY_SIZE_Y / 2.0))) ){     //25 meters
                    v = (AVM_IMG_WIDTH / 2.0) - PIXEL_PER_METER*xy.x; 
                    u = (AVM_IMG_HEIGHT / 2.0) - PIXEL_PER_METER*xy.y;

                    AVM_seg_left.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);   //b
                    AVM_seg_left.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);   //g
                    AVM_seg_left.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);   //r

                    AVM_seg_left_gray.at<uchar>(int(v), int(u)) = static_cast<uint8_t>(cv_frame_resize_pad_gray.at<uchar>(i,j));   //r

                    int arrX, arrY; 
                    real2arr(xy.x, xy.y, arrX, arrY);
                    if (AVM_seg_left_gray.at<uchar>(int(v), int(u)) == 255) 
                        num_obsL[arrX][arrY]++;
                }
            }
    }
}

// void CallbackPhantom_right(const ocamcalib_undistort::PhantomVisionNetMsg::ConstPtr& msg) 
// void CallbackPhantom_right(const ocamcalib_undistort::VisionPhantomnetData::ConstPtr& msg) 
void CallbackPhantom_right(const sensor_msgs::ImageConstPtr& msg) 
{
    if( DIRECTION != 1){
        // cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg->image_raw, msg->image_raw.encoding )->image;
        // cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg->image_viz, msg->image_viz.encoding )->image;
        cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg, "bgr8" )->image;
        
        // cv::imshow("right", cv_frame_resize_pad);
        // cv::waitKey(1);

        //resize
        cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(CROP_ROI_WIDTH, CROP_ROI_HEIGHT), 0, 0, cv::INTER_LINEAR );
    
        //padding
        cv::Mat temp;

        // cv::Rect bounds(0,0,cv_frame_resize_pad.cols,cv_frame_resize_pad.rows);
        // cv::Rect r(0,12,1920,1080 - 12);
        // cv:R:Mat roi = cv_frame_resize_pad( r & bounds );
        // cv::copyMakeBorder(roi, cv_frame_resize_pad, 0, 140, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        //new1
        // cv::copyMakeBorder(cv_frame_resize_pad, cv_frame_resize_pad, 0, PADDING_RIGHT_BOTTOM, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        //new2
        cv::copyMakeBorder(cv_frame_resize_pad, cv_frame_resize_pad, PADDING_RIGHT_BOTTOM, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        // cv::copyMakeBorder(cv_frame_resize_pad, cv_frame_resize_pad, 0, 0, 0, 100, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        
        XY_coord xy; 
        
        if(REOLUTION_MODE == 1){       //half_resolution
            cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(FULL_IMG_RESOL_WIDTH / 2.0, FULL_IMG_RESOL_HEIGHT /2.0), 0, 0, cv::INTER_LINEAR );
            resolution = 2;
        }
        else if(REOLUTION_MODE == 2){  //quarter_resolution
            cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(FULL_IMG_RESOL_WIDTH / 4.0, FULL_IMG_RESOL_HEIGHT /4.0), 0, 0, cv::INTER_LINEAR );
            resolution = 4;
        }
        int u,v;

        AVM_right = cv::Mat::zeros(AVM_IMG_WIDTH, AVM_IMG_HEIGHT, CV_8UC3);

        for(int i=1; i< cv_frame_resize_pad.size().height ;i++)
            for(int j=0 ;j< cv_frame_resize_pad.size().width ;j++){
                xy = InvProjGRD(resolution * j,resolution * i, M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4] ,M_right_param[5], &right_model);
                
                if( !(xy.x ==0) && (xy.y ==0) || !((abs(xy.x) >=(REAL_OCCUPANCY_SIZE_X / 2.0)) || (abs(xy.y) >=(REAL_OCCUPANCY_SIZE_Y / 2.0))) ){
                    v = (AVM_IMG_WIDTH / 2.0) - PIXEL_PER_METER*xy.x;   // AVM image width  (MAX : 400)
                    u = (AVM_IMG_HEIGHT / 2.0) - PIXEL_PER_METER*xy.y;  // AVM image height (Max : 400)

                    AVM_right.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);   //b
                    AVM_right.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);   //g
                    AVM_right.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);   //r
                }
            }
    }
}

// void CallbackPhantom_seg_right(const ocamcalib_undistort::PhantomVisionNetMsg::ConstPtr& msg){
//     if( DIRECTION != 1 && SEG_IMG_PUB){
//         cv::Mat cv_frame_seg = cv_bridge::toCvCopy( msg->image_seg, msg->image_seg.encoding )->image;
//         cv::Mat cv_frame_raw_new = cv_bridge::toCvCopy( msg->image_raw, msg->image_raw.encoding )->image;
//         cv::Mat cv_frame_raw_new_gray = cv_bridge::toCvCopy( msg->image_raw, msg->image_seg.encoding )->image;
        
//         seg2rgb(cv_frame_seg, cv_frame_raw_new, cv_frame_raw_new_gray);
        
//         //resize
//         cv::Mat cv_frame_resize, cv_frame_resize_gray;
//         cv::resize( cv_frame_raw_new, cv_frame_resize, Size(1920, 1080), 0, 0, INTER_LINEAR );
//         cv::resize( cv_frame_raw_new_gray, cv_frame_resize_gray, Size(1920, 1080), 0, 0, INTER_LINEAR );

//         cv::Mat cv_frame_resize_pad, cv_frame_resize_pad_gray;
//         cv::copyMakeBorder(cv_frame_resize, cv_frame_resize_pad, 0, 128, 0, 0, BORDER_CONSTANT, Scalar(0,0,0) );
//         cv::copyMakeBorder(cv_frame_resize_gray, cv_frame_resize_pad_gray, 0, 128, 0, 0, BORDER_CONSTANT, Scalar(0,0,0) );

        
//         if(REOLUTION_MODE == 1){       //half_resolution
//             cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(960, 604), 0, 0, cv::INTER_LINEAR );
//             cv::resize( cv_frame_resize_pad_gray, cv_frame_resize_pad_gray, cv::Size(960, 604), 0, 0, cv::INTER_LINEAR );
//             resolution = 2;
//         }
//         else if(REOLUTION_MODE == 2){  //quarter_resolution
//             cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(480, 302), 0, 0, cv::INTER_LINEAR );
//             cv::resize( cv_frame_resize_pad_gray, cv_frame_resize_pad_gray, cv::Size(480, 302), 0, 0, cv::INTER_LINEAR );
//             resolution = 4;
//         }

//         XY_coord xy, arr_xy;
//         int u,v;

//         AVM_seg_right = cv::Mat::zeros(400,400, CV_8UC3);
//         AVM_seg_right_gray = cv::Mat::zeros(400,400, CV_8UC1);
        
//         for(int i=0; i < m_gridDim ; i++) for(int j=0 ;j < m_gridDim ;j++) num_obsR[j][i] = 0;

//         for(int i=0; i< cv_frame_resize_pad.size().height ;i++)
//             for(int j=0 ;j< cv_frame_resize_pad.size().width ;j++){
//                 xy = InvProjGRD(resolution * j, resolution * i, M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4] ,M_right_param[5], &right_model);

//                 if( !(xy.x ==0) && (xy.y ==0) || !((abs(xy.x) >=12.5) || (abs(xy.y) >=12.5)) ){     //25 meters
//                     v = 200 - 16*xy.x; 
//                     u = 200 - 16*xy.y; 

//                     AVM_seg_right.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);   //b
//                     AVM_seg_right.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);   //g
//                     AVM_seg_right.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);   //r

//                     AVM_seg_right_gray.at<uchar>(int(v), int(u)) = static_cast<uint8_t>(cv_frame_resize_pad_gray.at<uchar>(i,j));   //gray

//                     int arrX, arrY; 
//                     real2arr(xy.x, xy.y, arrX, arrY);
//                     if (AVM_seg_right_gray.at<uchar>(int(v), int(u)) == 255) 
//                         num_obsR[arrX][arrY]++;
//                 }
//             }
//     }
// }

void Local2Global(double Lx, double Ly, double &gX, double &gY) {
    gX = m_car.x + (Lx * cos(m_car.th) - Ly * sin(m_car.th));
    gY = m_car.y + (Lx * sin(m_car.th) + Ly * cos(m_car.th));
}

void AVMpointCloud(cv::Mat img) {
    int avmCutRange = 0, idxSparse = 1;
    if (m_flagDR) {idxSparse = 3; avmCutRange = 75;}

    if (m_DRcnt%3 == 0) {
        m_DRcnt = 0;
        for(int i = avmCutRange ; i < img.size().height - avmCutRange ; i = i+idxSparse){
            for(int j = avmCutRange ; j < img.size().width -avmCutRange ; j = j+idxSparse){
                if(!( img.at<cv::Vec3b>(i,j)[1] == 0) && !(img.at<cv::Vec3b>(i,j)[0] == 0) && !(img.at<cv::Vec3b>(i,j)[2] == 0)) {   
                    double x = 12.5 - 0.0625 * i, y = 12.5 - 0.0625 * j, gX, gY;
                    Local2Global(x, y, gX, gY);

                    pcl::PointXYZRGB pt;
                    pt.x = gX;  pt.y = gY;  pt.z = 0.0;

                    uint8_t r = static_cast<uint8_t>(img.at<cv::Vec3b>(i,j)[2]), g = static_cast<uint8_t>(img.at<cv::Vec3b>(i,j)[1]), b = static_cast<uint8_t>(img.at<cv::Vec3b>(i,j)[0]);
                    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
                    pt.rgb = *reinterpret_cast<float*>(&rgb);

                    m_avm_data->push_back(pt);
                }
            }
        }
    }

    if (m_flagDR) {
        if (m_avm_data->size() > 700000)
            m_avm_data->erase(m_avm_data->begin(), m_avm_data->begin() + 10000);
    } else {
        m_avm_data->erase(m_avm_data->begin(), m_avm_data->begin() + m_avmDRsize);
        m_avmDRsize = m_avm_data->size();
    }
    
    // cout << m_avm_data->size() << endl;
    Pub_AVM_DR.publish(m_avm_data);
}

void CallbackPhantom_DR(const std_msgs::Float32MultiArray::ConstPtr& msg){
    m_car.x = msg->data.at(0);      // x
    m_car.y = msg->data.at(1);      // y
    m_car.th = msg->data.at(2);     // theta
    m_car.vel = msg->data.at(3);    // [m/s]
    m_flagDR = true;
    m_DRcnt++;
    AVMpointCloud(aggregated_img);
}

// void CallbackPhantom_temp(const ocamcalib_undistort::VisionPhantomnetDataList::ConstPtr &msg) {

//     // for (int i = 0 ; i < 22 ; i++) 
//     //     cout << "[" << i << "] encoding: " << msg->nets_data[i].image_viz.encoding << ", width: " << msg->nets_data[i].image_viz.width << ", height: " << msg->nets_data[i].image_viz.height << endl;
//     // cout << endl;

//     // if (msg->nets_data[0].image_viz.width > 0)
//     // cv::imshow("[0] front_center",              cv_bridge::toCvCopy(msg->nets_data[0].image_viz, msg->nets_data[0].image_viz.encoding)->image);
//     // cv::waitKey(1);
//     // if (msg->nets_data[1].image_viz.width > 0)
//     // cv::imshow("[1] front_center_crop",         cv_bridge::toCvCopy(msg->nets_data[1].image_viz, msg->nets_data[1].image_viz.encoding)->image);
//     // cv::waitKey(1);
//     // if (msg->nets_data[2].image_viz.width > 0)
//     // cv::imshow("[2] front_center_narrow",       cv_bridge::toCvCopy(msg->nets_data[2].image_viz, msg->nets_data[2].image_viz.encoding)->image);
//     // cv::waitKey(1);
//     // if (msg->nets_data[3].image_viz.width > 0)
//     // cv::imshow("[3] front_center_narrow_crop",  cv_bridge::toCvCopy(msg->nets_data[3].image_viz, msg->nets_data[3].image_viz.encoding)->image);
//     // cv::waitKey(1);
//     // if (msg->nets_data[4].image_viz.width > 0)
//     // cv::imshow("[4] front_center_svm",          cv_bridge::toCvCopy(msg->nets_data[4].image_viz, msg->nets_data[4].image_viz.encoding)->image);
//     // cv::waitKey(1);
//     // if (msg->nets_data[5].image_viz.width > 0)
//     // cv::imshow("[5] front_center_svm_crop",     cv_bridge::toCvCopy(msg->nets_data[5].image_viz, msg->nets_data[5].image_viz.encoding)->image);
//     // cv::waitKey(1);

//     cout << "[" << 0 << "] encoding: " << msg->nets_data[0].image_viz.encoding << ", width: " << msg->nets_data[0].image_viz.width << ", id: " << (msg->nets_data[0].frame) << endl;
//     cout << "[" << 0 << "] encoding: " << msg->nets_data[0].image_seg.encoding << ", width: " << msg->nets_data[0].image_seg.width << ", name: " << (msg->nets_data[0].frame_next) << endl;
    
//     if (msg->nets_data[0].image_viz.width > 0) {
//         cv::imshow("[0] front_center_viz", cv_bridge::toCvCopy(msg->nets_data[0].image_viz, msg->nets_data[0].image_viz.encoding)->image);
//         // cv::waitKey(1);
        
//         cv::Mat cv_frame_seg = cv_bridge::toCvCopy( msg->nets_data[0].image_seg, msg->nets_data[0].image_seg.encoding )->image;
//         cv::Mat cv_frame_raw_new = cv_bridge::toCvCopy( msg->nets_data[0].image_viz, msg->nets_data[0].image_viz.encoding )->image;
//         cv::Mat cv_frame_raw_new_gray = cv_bridge::toCvCopy( msg->nets_data[0].image_viz, msg->nets_data[0].image_seg.encoding )->image;
        
//         seg2rgb(cv_frame_seg, cv_frame_raw_new, cv_frame_raw_new_gray);
//         cv::imshow("[0] front_center_seg", cv_frame_raw_new);
//         cv::waitKey(1);
//     }
// }

void occupancyGridmapPub() {
    int cnt = 0;
    // occupancyGridMap.data.clear();
    occupancyGridMap.data.resize(occupancyGridMap.info.width*occupancyGridMap.info.width);
    for(int i=0; i < m_gridDim ; i++) 
        for(int j=0 ;j < m_gridDim ;j++) 
            if (num_obsL[j][i] > 0 || num_obsR[j][i] > 0)   occupancyGridMap.data[cnt++] = 100;
            else                                            occupancyGridMap.data[cnt++] = 0;

    if (Pub_occupancyGridMap.getNumSubscribers() > 0)
        Pub_occupancyGridMap.publish(occupancyGridMap);
}

int main(int argc, char **argv)
{   
    ros::init(argc, argv, "undistort_node");
    ros::NodeHandle nodeHandle("~");

    std::string calibration_front ="/home/dyros-phantom/catkin_ws/src/ocamcalib_undistort/include/calib_results_phantom_190_028_front.txt";
    std::string calibration_left = "/home/dyros-phantom/catkin_ws/src/ocamcalib_undistort/include/calib_results_phantom_190_022_left.txt";
    std::string calibration_right ="/home/dyros-phantom/catkin_ws/src/ocamcalib_undistort/include/calib_results_phantom_190_023_right.txt";
    std::string calibration_rear = "/home/dyros-phantom/catkin_ws/src/ocamcalib_undistort/include/calib_results_phantom_190_029_rear.txt" ;        

    if(flag == 0) {
        if(!get_ocam_model(&front_model, calibration_front.c_str()) ||
        !get_ocam_model(&left_model, calibration_left.c_str()) ||
        !get_ocam_model(&right_model, calibration_right.c_str()) ||
        !get_ocam_model(&rear_model, calibration_rear.c_str()))
            return 2;
        flag =1;
    }
/// old
    // Sub_phantom_side_left           = nodeHandle.subscribe("/phantomnet/output/side_left", 1, CallbackPhantom_left);
    // Sub_phantom_side_right          = nodeHandle.subscribe("/phantomnet/output/side_right", 1, CallbackPhantom_right);
    // Sub_phantom_front_center_svm    = nodeHandle.subscribe("/phantomnet/output/front_center_svm", 1 , CallbackPhantom_center);
    // Sub_phantom_rear_center_svm     = nodeHandle.subscribe("/phantomnet/output/rear_center_svm", 1, CallbackPhantom_rear);

    // Sub_phantom_right_seg   = nodeHandle.subscribe("/phantomnet/output/side_right", 1 , CallbackPhantom_seg_right);
    // Sub_phantom_left_seg    = nodeHandle.subscribe("/phantomnet/output/side_left", 1 , CallbackPhantom_seg_left);
//
    Sub_phantom_side_left           = nodeHandle.subscribe("/csi_cam/side_left/image_raw", 1, CallbackPhantom_left);
    Sub_phantom_side_right          = nodeHandle.subscribe("/csi_cam/side_right/image_raw", 1, CallbackPhantom_right);

    // sub_tmp = nodeHandle.subscribe("/phantomvision/phantomnets", 100, CallbackPhantom_temp);

    // Sub_phantom_side_left           = nodeHandle.subscribe("/phantomnet/output/side_left", 1, CallbackPhantom_left);
    // Sub_phantom_side_right          = nodeHandle.subscribe("/phantomnet/output/side_right", 1, CallbackPhantom_right);
    // Sub_phantom_front_center_svm    = nodeHandle.subscribe("/phantomnet/output/front_center_svm", 1 , CallbackPhantom_center);
    // Sub_phantom_rear_center_svm     = nodeHandle.subscribe("/phantomnet/output/rear_center_svm", 1, CallbackPhantom_rear);

    // Sub_phantom_right_seg   = nodeHandle.subscribe("/phantomnet/output/side_right", 1 , CallbackPhantom_seg_right);
    // Sub_phantom_left_seg    = nodeHandle.subscribe("/phantomnet/output/side_left", 1 , CallbackPhantom_seg_left);

    Sub_phantom_DR_Path = nodeHandle.subscribe("/LocalizationData", 1 , CallbackPhantom_DR);

    Pub_AVM_img          = nodeHandle.advertise<sensor_msgs::Image>("/AVM_image", 1);
    Pub_AVM_seg_img      = nodeHandle.advertise<sensor_msgs::Image>("/AVM_seg_image", 1);
    Pub_AVM_seg_img_gray = nodeHandle.advertise<sensor_msgs::Image>("/AVM_seg_image_gray", 1);
    Pub_AVM_DR           = nodeHandle.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/AVM_image_DR", 10);

    occupancyGridMap.header.frame_id = "map";
    occupancyGridMap.info.resolution = m_gridResol;
    occupancyGridMap.info.width = occupancyGridMap.info.height = m_gridDim;
    occupancyGridMap.info.origin.position.x = occupancyGridMap.info.origin.position.y = -m_dimension/2 - m_gridResol*2;
    occupancyGridMap.info.origin.position.z = 0.1;
    occupancyGridMap.data.resize(occupancyGridMap.info.width*occupancyGridMap.info.width);
    Pub_occupancyGridMap = nodeHandle.advertise<nav_msgs::OccupancyGrid>("/occ_map", 1);

    m_avm_data->clear();
    m_avm_data->header.frame_id = "map";
    
    ros::Rate loop_rate(20);
    // ros::spin();

    while(ros::ok()) { 
        if( DIRECTION == 0){              //side
            if (SEG_IMG_PUB) {
                if (m_flagDR) {
                    ros::AsyncSpinner spinner(4+1);
                    spinner.start();
                } else {
                    ros::AsyncSpinner spinner(4);
                    spinner.start();
                }
            }
            else {
                if (m_flagDR) {
                    ros::AsyncSpinner spinner(2+1);
                    spinner.start();
                } else {
                    ros::AsyncSpinner spinner(2);
                    spinner.start();
                }
            }
            aggregated_img = AVM_right + AVM_left;

            // if (CNTforIMAGE++ % 3 == 0)
            //     imwrite("/home/dyros-phantom/catkin_ws/src/ocamcalib_undistort/image/PhantomAVM_"+to_string(CNTforIMAGE_save++)+".jpg", aggregated_img);

            
            //add
            if (SEG_IMG_PUB)
                aggregated_seg_img = AVM_seg_right + AVM_seg_left;
            aggregated_seg_img_gray = AVM_seg_right_gray + AVM_seg_left_gray;
            //add
        }
        else if( DIRECTION == 1 ){        //center
            ros::AsyncSpinner spinner(2);
            spinner.start();

            aggregated_img = AVM_front + AVM_rear;
        }
        else if( DIRECTION == 2){         //both
            ros::AsyncSpinner spinner(4);
            spinner.start();

            aggregated_img = AVM_front + AVM_right + AVM_left + AVM_rear;
        }
        Pub_AVM_img.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", aggregated_img).toImageMsg());
        if (SEG_IMG_PUB) 
            Pub_AVM_seg_img.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", aggregated_seg_img).toImageMsg());
        Pub_AVM_seg_img_gray.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", aggregated_seg_img_gray).toImageMsg());

        occupancyGridmapPub();

        if (!m_flagDR)
            AVMpointCloud(aggregated_img);

        ros::spinOnce();
         loop_rate.sleep();
    }
    return 0;
}
