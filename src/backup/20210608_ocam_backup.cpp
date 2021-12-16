#include "ocamcalib_undistort/ocam_functions.h"
#include "ocamcalib_undistort/Parameters.h"
 
#include <iostream>
#include <string>
#include <exception>

#include <ros/ros.h>
#include <ros/package.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseArray.h>

#include <cv_bridge/cv_bridge.h>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"

#include <std_msgs/Float32MultiArray.h>
#include <pcl_ros/point_cloud.h>
#include <algorithm>

// #include <ocamcalib_undistort/PhantomVisionNetMsg.h> 
// #include <ocamcalib_undistort/VisionPhantomnetData.h>
// #include <ocamcalib_undistort/VisionPhantomnetDataList.h>
#include <ocamcalib_undistort/ParkingPhantomnetData.h>
#include <ocamcalib_undistort/ParkingPhantomnetDetection.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/io/io.hpp>
#include <boost/program_options.hpp>

using namespace cv;
using namespace std;

//선택지 6개
#define REOLUTION_MODE 1    //full: 0, half: 1, quarter: 2
#define DIRECTION 0         //side: 0, center: 1, both: 2
#define SEG_IMG_PUB 1       // 1: true, 2: false

#define AVM_IMG_WIDTH  300
#define AVM_IMG_HEIGHT 300
// #define CROP_TOP 12
#define PADDING_CENTER_BOTTOM 128
#define PADDING_REAR_BOTTOM 128

#define PADDING_VALUE 120
bool Padding_UP = false;  //1 up, 0 down

#define FULL_IMG_RESOL_WIDTH  1920
#define FULL_IMG_RESOL_HEIGHT 1208
#define CROP_ROI_WIDTH 1920
#define CROP_ROI_HEIGHT 1080

#define REAL_OCCUPANCY_SIZE_X 25    // AVM_IMG_WIDTH 400PIX == 25Meter
#define REAL_OCCUPANCY_SIZE_Y 25    // AVM_IMG_HEIGHT 400PIX == 25Meter

#define PIXEL_PER_METER AVM_IMG_WIDTH/REAL_OCCUPANCY_SIZE_X           //400PIX / 25M

//////////////////////////////////////////////////////
#define M_SPACE_WIDTH 2.65  // 2.6
#define M_SPACE_LENGTH 7.0  // 5.2
#define FREE_ERASE_GAIN 1.2

#define FRONT 0
#define REAR 1
#define LEFT 2
#define RIGHT 3

ros::Subscriber sub_tmp;
ros::Subscriber Sub_phantom_side_left, Sub_phantom_side_right, Sub_phantom_front_center, Sub_phantom_rear_center;
ros::Subscriber Sub_phantom_left_seg, Sub_phantom_right_seg, Sub_phantom_front_seg, Sub_phantom_rear_seg;
ros::Subscriber Sub_phantom_DR_Path, Sub_parkingGoal;

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
double m_gridResol = 0.2;//0.25;
const int m_gridDim = (int)(m_dimension*(int)(1/m_gridResol));
// int num_obsL[140][140] = {{0,}, {0,}};
// int num_obsR[140][140] = {{0,}, {0,}};
int num_obsL[175][175] = {{0,}, {0,}};
int num_obsR[175][175] = {{0,}, {0,}};

int num_freeL[175][175] = {{0,}, {0,}};
int num_freeR[175][175] = {{0,}, {0,}};

// for checking DR error
struct CARPOSE {
    double x,y,th,vel;
};CARPOSE m_car;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_avm_data(new pcl::PointCloud<pcl::PointXYZRGB>);
bool m_flagDR = false;
unsigned int m_avmDRsize = 0, m_DRcnt = 0;
unsigned int CNTforIMAGE_save = 172, CNTforIMAGE = 0;

void arr2real(int recvX, int recvY, double& outX, double& outY) {
    outX = recvX * m_gridResol - (m_gridResol*m_gridDim - m_gridResol) / 2.0;
    outY = recvY * m_gridResol - (m_gridResol*m_gridDim - m_gridResol) / 2.0;
}

void real2arr(double recvX, double recvY, int& outX, int& outY) {
    outX = (m_gridDim/2.0) + recvX / m_gridResol;
    outY = (m_gridDim/2.0) + recvY / m_gridResol;
}

//Yangwoo===============================================================
double GOAL_G[3] = {0.0, 0.0, (0)};
int modx1 = 0, mody1 =0, modx2 = 0, mody2 =0, modx3 = 0, mody3 =0, modx4 = 0, mody4 =0;
int modx1free = 0, mody1free =0, modx2free = 0, mody2free =0, modx3free = 0, mody3free =0, modx4free = 0, mody4free =0;

void coord(double dx, double dy, double xx, double yy, double thh, int& x_, int& y_) {
    double modx = cos(M_PI/2+thh)*dx - sin(M_PI/2+thh)*dy + xx;
    double mody = sin(M_PI/2+thh)*dx + cos(M_PI/2+thh)*dy + yy;
    real2arr(modx, mody, x_, y_);
}

int withinpoint(int x, int y) {
    typedef boost::geometry::model::d2::point_xy<int> point_type;
    typedef boost::geometry::model::polygon<point_type> polygon_type;

    polygon_type poly;
    poly.outer().assign({
        point_type {modx1, mody1}, point_type {modx2, mody2},
        point_type {modx3, mody3}, point_type {modx4, mody4},
        point_type {modx1, mody1}
    });

    point_type p(x, y);

    return boost::geometry::within(p, poly);
}

int withinpoint_free(int x, int y) {
    typedef boost::geometry::model::d2::point_xy<int> point_type;
    typedef boost::geometry::model::polygon<point_type> polygon_type;

    polygon_type poly;
    poly.outer().assign({
        point_type {modx1free, mody1free}, point_type {modx2free, mody2free},
        point_type {modx3free, mody3free}, point_type {modx4free, mody4free},
        point_type {modx1free, mody1free}
    });

    point_type p(x, y);

    return boost::geometry::within(p, poly);
}

void CallbackParkingGoal(const geometry_msgs::PoseArray::ConstPtr& end) {    //[end] which is the coordinates of the goal
    GOAL_G[0] = end->poses[0].position.x;
    GOAL_G[1] = end->poses[0].position.y;
    GOAL_G[2] = tf::getYaw(end->poses[0].orientation);

    coord(M_SPACE_WIDTH/2, -M_SPACE_LENGTH/2,  GOAL_G[0],GOAL_G[1], GOAL_G[2], modx1, mody1);
    coord(M_SPACE_WIDTH/2, M_SPACE_LENGTH/2,   GOAL_G[0],GOAL_G[1], GOAL_G[2], modx2, mody2);
    coord(-M_SPACE_WIDTH/2, M_SPACE_LENGTH/2,  GOAL_G[0],GOAL_G[1], GOAL_G[2], modx3, mody3);
    coord(-M_SPACE_WIDTH/2, -M_SPACE_LENGTH/2, GOAL_G[0],GOAL_G[1], GOAL_G[2], modx4, mody4);

    coord( M_SPACE_WIDTH/2*FREE_ERASE_GAIN, -M_SPACE_LENGTH/2*FREE_ERASE_GAIN, GOAL_G[0],GOAL_G[1], GOAL_G[2], modx1free, mody1free);
    coord( M_SPACE_WIDTH/2*FREE_ERASE_GAIN,  M_SPACE_LENGTH/2*FREE_ERASE_GAIN, GOAL_G[0],GOAL_G[1], GOAL_G[2], modx2free, mody2free);
    coord(-M_SPACE_WIDTH/2*FREE_ERASE_GAIN,  M_SPACE_LENGTH/2*FREE_ERASE_GAIN, GOAL_G[0],GOAL_G[1], GOAL_G[2], modx3free, mody3free);
    coord(-M_SPACE_WIDTH/2*FREE_ERASE_GAIN, -M_SPACE_LENGTH/2*FREE_ERASE_GAIN, GOAL_G[0],GOAL_G[1], GOAL_G[2], modx4free, mody4free);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

void Local2Global(double Lx, double Ly, double &gX, double &gY) {
    gX = m_car.x + (Lx * cos(m_car.th) - Ly * sin(m_car.th));
    gY = m_car.y + (Lx * sin(m_car.th) + Ly * cos(m_car.th));
}

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
// Segmentation: 14 classes
// (ID and description)
// 0 : background
// 1 : lane - solid, dashed, dotted
// 2 : lane - parking, stop, arrow, etc
// 4 : vehicle - all types
// 6 : wheel
// 9 : general - cone, curbstone, parking block, etc
// 10: cycle, bicyclist, motorcyclist
// 14: pedestrian
// 15: freespace
// 17: parking space
// 18: crosswalk
// 19: speed bump
// 20: foot
// 21: head
            if ((int)input_img.at<uchar>(i, j) == 4 || (int)input_img.at<uchar>(i, j) == 6 || (int)input_img.at<uchar>(i, j) == 17) {
                output_img_gray.at<uchar>(i, j) = 255;
                if ((int)input_img.at<uchar>(i, j) == 17) {
                    output_img_gray.at<uchar>(i, j) = 17;
                }
            }
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
                    output_img.at<Vec3b>(i, j)[1] = 0;
                    output_img.at<Vec3b>(i, j)[2] = 255;
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
                // case 13 :    // road 
                //     output_img.at<Vec3b>(i, j)[0] = 0;
                //     output_img.at<Vec3b>(i, j)[1] = 255;
                //     output_img.at<Vec3b>(i, j)[2] = 0;
                // break;
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
                case 17 :    // road 
                    output_img.at<Vec3b>(i, j)[0] = 0;
                    output_img.at<Vec3b>(i, j)[1] = 255;
                    output_img.at<Vec3b>(i, j)[2] = 0;
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
void Forward_Warping(cv::Mat input_img, cv::Mat& output_img, int Direction_Mode)
{
    //resize, bagfile_size: (512, 288) * 3.75 => (1920, 1080)
    cv::resize( input_img, input_img, cv::Size(CROP_ROI_WIDTH, CROP_ROI_HEIGHT), 0, 0, cv::INTER_LINEAR );
    // padding
    cv::Mat temp;

    // //padding 1920 1280
    if (Padding_UP)
        cv::copyMakeBorder(input_img, input_img, PADDING_VALUE, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
    else
        cv::copyMakeBorder(input_img, input_img, 0, PADDING_VALUE, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
    
    XY_coord xy; 
    if(REOLUTION_MODE == 1){       //half_resolution
        cv::resize( input_img, input_img, cv::Size(FULL_IMG_RESOL_WIDTH / 2.0, FULL_IMG_RESOL_HEIGHT /2.0), 0, 0, cv::INTER_LINEAR );
        resolution = 2;
    }
    else if(REOLUTION_MODE == 2){  //quarter_resolution
        cv::resize( input_img, input_img, cv::Size(FULL_IMG_RESOL_WIDTH / 4.0, FULL_IMG_RESOL_HEIGHT / 4.0), 0, 0, cv::INTER_LINEAR );
        resolution = 4;
    }
    int u,v;

    AVM_front = cv::Mat::zeros(AVM_IMG_WIDTH, AVM_IMG_HEIGHT, CV_8UC3);
    
    double *Dir_param;
    ocam_model *model;

    if(Direction_Mode == FRONT){
        Dir_param = M_front_param;
        model = &front_model;
    }
    else if(Direction_Mode == REAR){
        Dir_param = M_back_param; 
        model = &rear_model;
    }
    else if(Direction_Mode == LEFT){
        Dir_param = M_left_param;
        model = &left_model;
    }
    else if(Direction_Mode == RIGHT){
        Dir_param = M_right_param;
        model = &right_model;
    }
    for(int i=0; i< input_img.size().height  ;i++)
        for(int j=0 ;j< input_img.size().width ;j++){
        
            // Image_pixel to World_Coordinate (x (meter),y (meter))
            xy = InvProjGRD(resolution * j, resolution * i, Dir_param[0], Dir_param[1], Dir_param[2], Dir_param[3], Dir_param[4] ,Dir_param[5], model);

            if( !(xy.x ==0) && (xy.y ==0) || !((abs(xy.x) >=(REAL_OCCUPANCY_SIZE_X / 2.0)) || (abs(xy.y) >= (REAL_OCCUPANCY_SIZE_Y / 2.0))) ){
                v = (AVM_IMG_WIDTH / 2.0) - PIXEL_PER_METER*xy.x;  // AVM image width  (MAX : 400)
                u = (AVM_IMG_HEIGHT / 2.0) - PIXEL_PER_METER*xy.y;  // AVM image height (Max : 400)

                output_img.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(input_img.at<cv::Vec3b>(i,j)[0]);   //b
                output_img.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(input_img.at<cv::Vec3b>(i,j)[1]);   //g
                output_img.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(input_img.at<cv::Vec3b>(i,j)[2]);   //r
            }   
        }
}

void CallbackPhantom_center(const sensor_msgs::ImageConstPtr& msg) 
// void CallbackPhantom_center(const ocamcalib_undistort::PhantomVisionNetMsg::ConstPtr& msg) 
{   
    if( DIRECTION != 0){
        // cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg->image_raw, msg->image_raw.encoding )->image;
        cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy(msg, "bgr8" )->image;

        //resize, bagfile_size: (512, 288) * 3.75 => (1920, 1080)
        cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(CROP_ROI_WIDTH, CROP_ROI_HEIGHT), 0, 0, cv::INTER_LINEAR );
        // padding
        cv::Mat temp;
        
        cv::copyMakeBorder(cv_frame_resize_pad, cv_frame_resize_pad, 0, PADDING_CENTER_BOTTOM, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );

        XY_coord xy; 
        if(REOLUTION_MODE == 1){       //half_resolution
            cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(FULL_IMG_RESOL_WIDTH / 2.0, FULL_IMG_RESOL_HEIGHT /2.0), 0, 0, cv::INTER_LINEAR );
            resolution = 2;
        }
        else if(REOLUTION_MODE == 2){  //quarter_resolution
            cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(FULL_IMG_RESOL_WIDTH / 4.0, FULL_IMG_RESOL_HEIGHT / 4.0), 0, 0, cv::INTER_LINEAR );
            resolution = 4;
        }
        int u,v;

        AVM_front = cv::Mat::zeros(AVM_IMG_WIDTH, AVM_IMG_HEIGHT, CV_8UC3);

        for(int i=0; i< cv_frame_resize_pad.size().height  ;i++)
            for(int j=0 ;j< cv_frame_resize_pad.size().width ;j++){
            
                // Image_pixel to World_Coordinate (x (meter),y (meter))
                xy = InvProjGRD(resolution * j, resolution * i, M_front_param[0], M_front_param[1], M_front_param[2], M_front_param[3], M_front_param[4] ,M_front_param[5], &front_model);

                if( !(xy.x ==0) && (xy.y ==0) || !((abs(xy.x) >=(REAL_OCCUPANCY_SIZE_X / 2.0)) || (abs(xy.y) >= (REAL_OCCUPANCY_SIZE_Y / 2.0))) ){
                    v = (AVM_IMG_WIDTH / 2.0) - PIXEL_PER_METER*xy.x;  // AVM image width  (MAX : 400)
                    u = (AVM_IMG_HEIGHT / 2.0) - PIXEL_PER_METER*xy.y;  // AVM image height (Max : 400)

                    AVM_front.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);   //b
                    AVM_front.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);   //g
                    AVM_front.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);   //r
                }   
            }
    }
}


void CallbackPhantom_rear(const sensor_msgs::ImageConstPtr& msg) 
// void CallbackPhantom_rear(const ocamcalib_undistort::PhantomVisionNetMsg::ConstPtr& msg) 
{
    if( DIRECTION != 0){
        // cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg->image_raw, msg->image_raw.encoding )->image;
        cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy(msg, "bgr8" )->image;
        //resize
        cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(CROP_ROI_WIDTH, CROP_ROI_HEIGHT), 0, 0, cv::INTER_LINEAR );

        //padding
        cv::Mat temp;
        cv::copyMakeBorder(cv_frame_resize_pad, cv_frame_resize_pad, 0, PADDING_REAR_BOTTOM, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );

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
        
        AVM_rear = cv::Mat::zeros(AVM_IMG_WIDTH, AVM_IMG_HEIGHT, CV_8UC3);

        for(int i=0; i< cv_frame_resize_pad.size().height  ;i++)
            for(int j=0 ;j< cv_frame_resize_pad.size().width ;j++){
                xy = InvProjGRD(resolution * j,resolution * i, M_back_param[0], M_back_param[1], M_back_param[2], M_back_param[3], M_back_param[4] ,M_back_param[5], &rear_model);
            
                if( !(xy.x ==0) && (xy.y ==0) || !((abs(xy.x) >= (REAL_OCCUPANCY_SIZE_X / 2.0)) || (abs(xy.y) >=(REAL_OCCUPANCY_SIZE_Y / 2.0))) ){
                    v = (AVM_IMG_WIDTH / 2.0) - PIXEL_PER_METER*xy.x;   // AVM image width  (MAX : 400) 
                    u = (AVM_IMG_HEIGHT / 2.0) - PIXEL_PER_METER*xy.y;  // AVM image height (Max : 400)

                    AVM_rear.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);   //b
                    AVM_rear.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);   //g
                    AVM_rear.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);   //r
                }
            }
    }
}

void CallbackPhantom_left(const sensor_msgs::ImageConstPtr& msg) 

{
   if( DIRECTION != 1){
        // cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg->image_raw, msg->image_raw.encoding )->image;
        // cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg->image_viz, msg->image_viz.encoding )->image;
        cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy(msg, "bgr8" )->image;
        // cout << cv_frame_resize_pad.size().width << " " << cv_frame_resize_pad.size().height  << endl;
        // cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg->viz, msg->viz.encoding )->image;
        
        // cv::imshow("left", cv_frame_resize_pad);
        // cv::waitKey(1);
        
        // //resize

        cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(CROP_ROI_WIDTH, CROP_ROI_HEIGHT), 0, 0, cv::INTER_LINEAR );
    
        // //padding 1920 1280
        if (Padding_UP)
            cv::copyMakeBorder(cv_frame_resize_pad, cv_frame_resize_pad, PADDING_VALUE, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        else
            cv::copyMakeBorder(cv_frame_resize_pad, cv_frame_resize_pad, 0, PADDING_VALUE, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        
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

        AVM_left = cv::Mat::zeros(AVM_IMG_WIDTH, AVM_IMG_HEIGHT, CV_8UC3);

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

        AVM_left.at<cv::Vec3b>(200, 200)[0] = 0;   //b
        AVM_left.at<cv::Vec3b>(200, 200)[1] = 0;   //g
        AVM_left.at<cv::Vec3b>(200, 200)[2] = 0;   //r

    }
}

// void CallbackPhantom_seg_left(const ocamcalib_undistort::PhantomVisionNetMsg::ConstPtr& msg){
void CallbackPhantom_seg_left(const ocamcalib_undistort::ParkingPhantomnetData::ConstPtr& msg) { 
    if( DIRECTION != 1 && SEG_IMG_PUB){
        cv::Mat cv_frame_seg = cv_bridge::toCvCopy( msg->segmentation, msg->segmentation.encoding )->image;
        cv::Mat cv_frame_raw_new = cv_bridge::toCvCopy( msg->viz, msg->viz.encoding )->image;
        cv::Mat cv_frame_raw_new_gray = cv_bridge::toCvCopy( msg->viz, msg->segmentation.encoding )->image;
        
        seg2rgb(cv_frame_seg, cv_frame_raw_new, cv_frame_raw_new_gray);
        
        //resize
        cv::Mat cv_frame_resize, cv_frame_resize_gray;
        cv::resize( cv_frame_raw_new, cv_frame_resize, Size(1920, 1080), 0, 0, INTER_LINEAR );
        cv::resize( cv_frame_raw_new_gray, cv_frame_resize_gray, Size(1920, 1080), 0, 0, INTER_LINEAR );

        cv::Mat cv_frame_resize_pad, cv_frame_resize_pad_gray;
        if (Padding_UP) {
            cv::copyMakeBorder(cv_frame_resize, cv_frame_resize_pad, PADDING_VALUE, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
            cv::copyMakeBorder(cv_frame_resize_gray, cv_frame_resize_pad_gray, PADDING_VALUE, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        }
        else {
            cv::copyMakeBorder(cv_frame_resize, cv_frame_resize_pad, 0, PADDING_VALUE, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
            cv::copyMakeBorder(cv_frame_resize_gray, cv_frame_resize_pad_gray, 0, PADDING_VALUE, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        }

        XY_coord xy;
        
        if(REOLUTION_MODE == 1){       //half_resolution
            cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(FULL_IMG_RESOL_WIDTH / 2.0, FULL_IMG_RESOL_HEIGHT /2.0), 0, 0, cv::INTER_LINEAR );
            cv::resize( cv_frame_resize_pad_gray, cv_frame_resize_pad_gray, cv::Size(FULL_IMG_RESOL_WIDTH / 2.0, FULL_IMG_RESOL_HEIGHT /2.0), 0, 0, cv::INTER_LINEAR );
            resolution = 2;
        }
        else if(REOLUTION_MODE == 2){  //quarter_resolution
            cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(FULL_IMG_RESOL_WIDTH / 4.0, FULL_IMG_RESOL_HEIGHT /4.0), 0, 0, cv::INTER_LINEAR );
            cv::resize( cv_frame_resize_pad_gray, cv_frame_resize_pad_gray, cv::Size(FULL_IMG_RESOL_WIDTH / 4.0, FULL_IMG_RESOL_HEIGHT /4.0), 0, 0, cv::INTER_LINEAR );
            resolution = 4;
        }
        int u,v;

        AVM_seg_left = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);
        AVM_seg_left_gray = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC1);
        
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
                    if (v != 200 && u != 200) {
                        if (AVM_seg_left_gray.at<uchar>(int(v), int(u)) == 255 || AVM_seg_left_gray.at<uchar>(int(v), int(u)) == 17 ) 
                            num_obsL[arrX][arrY]++;
                        if (AVM_seg_left_gray.at<uchar>(int(v), int(u)) == 17 ) 
                            num_freeL[arrX][arrY]++;
                    }
                }
            }
        AVM_seg_left.at<cv::Vec3b>(200, 200)[0] = 0;   //b
        AVM_seg_left.at<cv::Vec3b>(200, 200)[1] = 0;   //g
        AVM_seg_left.at<cv::Vec3b>(200, 200)[2] = 0;   //r
    }
}

// void CallbackPhantom_right(const ocamcalib_undistort::PhantomVisionNetMsg::ConstPtr& msg) 
// void CallbackPhantom_right(const ocamcalib_undistort::VisionPhantomnetData::ConstPtr& msg) 
void CallbackPhantom_right(const sensor_msgs::ImageConstPtr& msg) 
// void CallbackPhantom_right(const ocamcalib_undistort::ParkingPhantomnetData::ConstPtr& msg) 
{
    if( DIRECTION != 1){
        // cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg->image_raw, msg->image_raw.encoding )->image;
        // cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg->image_viz, msg->image_viz.encoding )->image;
        cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg, "bgr8" )->image;
        // cv::Mat cv_frame_resize_pad = cv_bridge::toCvCopy( msg->viz, msg->viz.encoding )->image;
        
        // cv::imshow("right", cv_frame_resize_pad);
        // cv::waitKey(1);

        //resize
        // cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(CROP_ROI_WIDTH, CROP_ROI_HEIGHT), 0, 0, cv::INTER_LINEAR );
    
        // //padding
        // if (Padding_UP)
        //     cv::copyMakeBorder(cv_frame_resize_pad, cv_frame_resize_pad, PADDING_VALUE, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        // else
        //     cv::copyMakeBorder(cv_frame_resize_pad, cv_frame_resize_pad, 0, PADDING_VALUE, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        
        // XY_coord xy; 
        
        // if(REOLUTION_MODE == 1){       //half_resolution
        //     cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(FULL_IMG_RESOL_WIDTH / 2.0, FULL_IMG_RESOL_HEIGHT /2.0), 0, 0, cv::INTER_LINEAR );
        //     resolution = 2;
        // }
        // else if(REOLUTION_MODE == 2){  //quarter_resolution
        //     cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(FULL_IMG_RESOL_WIDTH / 4.0, FULL_IMG_RESOL_HEIGHT /4.0), 0, 0, cv::INTER_LINEAR );
        //     resolution = 4;
        // }
        // int u,v;

        AVM_right = cv::Mat::zeros(AVM_IMG_WIDTH, AVM_IMG_HEIGHT, CV_8UC3);
        Forward_Warping(cv_frame_resize_pad, AVM_left, LEFT);

        // for(int i=1; i< cv_frame_resize_pad.size().height ;i++)
        //     for(int j=0 ;j< cv_frame_resize_pad.size().width ;j++){
        //         xy = InvProjGRD(resolution * j,resolution * i, M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4] ,M_right_param[5], &right_model);
                
        //         if( !(xy.x ==0) && (xy.y ==0) || !((abs(xy.x) >=(REAL_OCCUPANCY_SIZE_X / 2.0)) || (abs(xy.y) >=(REAL_OCCUPANCY_SIZE_Y / 2.0))) ){
        //             v = (AVM_IMG_WIDTH / 2.0) - PIXEL_PER_METER*xy.x;   // AVM image width  (MAX : 400)
        //             u = (AVM_IMG_HEIGHT / 2.0) - PIXEL_PER_METER*xy.y;  // AVM image height (Max : 400)

        //             AVM_right.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);   //b
        //             AVM_right.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);   //g
        //             AVM_right.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);   //r
        //         }
        //     }

        AVM_right.at<cv::Vec3b>(200, 200)[0] = 0;   //b
        AVM_right.at<cv::Vec3b>(200, 200)[1] = 0;   //g
        AVM_right.at<cv::Vec3b>(200, 200)[2] = 0;   //r
    }
}

// void CallbackPhantom_seg_right(const ocamcalib_undistort::PhantomVisionNetMsg::ConstPtr& msg){
void CallbackPhantom_seg_right(const ocamcalib_undistort::ParkingPhantomnetData::ConstPtr& msg) {
    if( DIRECTION != 1 && SEG_IMG_PUB){
        cv::Mat cv_frame_seg = cv_bridge::toCvCopy( msg->segmentation, msg->segmentation.encoding )->image;
        cv::Mat cv_frame_raw_new = cv_bridge::toCvCopy( msg->viz, msg->viz.encoding )->image;
        cv::Mat cv_frame_raw_new_gray = cv_bridge::toCvCopy( msg->viz, msg->segmentation.encoding )->image;
        
        seg2rgb(cv_frame_seg, cv_frame_raw_new, cv_frame_raw_new_gray);
        
        //resize
        cv::Mat cv_frame_resize, cv_frame_resize_gray;
        cv::resize( cv_frame_raw_new, cv_frame_resize, Size(1920, 1080), 0, 0, INTER_LINEAR );
        cv::resize( cv_frame_raw_new_gray, cv_frame_resize_gray, Size(1920, 1080), 0, 0, INTER_LINEAR );

        cv::Mat cv_frame_resize_pad, cv_frame_resize_pad_gray;
        
        if (Padding_UP) {
            cv::copyMakeBorder(cv_frame_resize, cv_frame_resize_pad, PADDING_VALUE, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
            cv::copyMakeBorder(cv_frame_resize_gray, cv_frame_resize_pad_gray, PADDING_VALUE, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        }
        else {
            cv::copyMakeBorder(cv_frame_resize, cv_frame_resize_pad, 0, PADDING_VALUE, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
            cv::copyMakeBorder(cv_frame_resize_gray, cv_frame_resize_pad_gray, 0, PADDING_VALUE, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        }
        XY_coord xy;
        
        if(REOLUTION_MODE == 1){       //half_resolution
            cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(FULL_IMG_RESOL_WIDTH / 2.0, FULL_IMG_RESOL_HEIGHT /2.0), 0, 0, cv::INTER_LINEAR );
            cv::resize( cv_frame_resize_pad_gray, cv_frame_resize_pad_gray, cv::Size(FULL_IMG_RESOL_WIDTH / 2.0, FULL_IMG_RESOL_HEIGHT /2.0), 0, 0, cv::INTER_LINEAR );
            resolution = 2;
        }
        else if(REOLUTION_MODE == 2){  //quarter_resolution
            cv::resize( cv_frame_resize_pad, cv_frame_resize_pad, cv::Size(FULL_IMG_RESOL_WIDTH / 4.0, FULL_IMG_RESOL_HEIGHT /4.0), 0, 0, cv::INTER_LINEAR );
            cv::resize( cv_frame_resize_pad_gray, cv_frame_resize_pad_gray, cv::Size(FULL_IMG_RESOL_WIDTH / 4.0, FULL_IMG_RESOL_HEIGHT /4.0), 0, 0, cv::INTER_LINEAR );
            resolution = 4;
        }
        int u,v;

        AVM_seg_right = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC3);
        AVM_seg_right_gray = cv::Mat::zeros(AVM_IMG_WIDTH,AVM_IMG_HEIGHT, CV_8UC1);
        
        for(int i=0; i < m_gridDim ; i++) for(int j=0 ;j < m_gridDim ;j++) num_obsR[j][i] = 0;

        for(int i=0; i< cv_frame_resize_pad.size().height ;i++)
            for(int j=0 ;j< cv_frame_resize_pad.size().width ;j++){
                xy = InvProjGRD(resolution * j, resolution * i, M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4] ,M_right_param[5], &right_model);

                if( !(xy.x ==0) && (xy.y ==0) || !((abs(xy.x) >=REAL_OCCUPANCY_SIZE_X/ 2.0) || (abs(xy.y) >=REAL_OCCUPANCY_SIZE_Y/ 2.0)) ){     //25 meters
                    v = (AVM_IMG_WIDTH / 2.0) - PIXEL_PER_METER*xy.x;   // AVM image width  (MAX : 400)
                    u = (AVM_IMG_HEIGHT / 2.0) - PIXEL_PER_METER*xy.y;  // AVM image height (Max : 400)

                    AVM_seg_right.at<cv::Vec3b>(int(v), int(u))[0] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[0]);   //b
                    AVM_seg_right.at<cv::Vec3b>(int(v), int(u))[1] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[1]);   //g
                    AVM_seg_right.at<cv::Vec3b>(int(v), int(u))[2] = static_cast<uint8_t>(cv_frame_resize_pad.at<cv::Vec3b>(i,j)[2]);   //r

                    AVM_seg_right_gray.at<uchar>(int(v), int(u)) = static_cast<uint8_t>(cv_frame_resize_pad_gray.at<uchar>(i,j));   //gray

                    int arrX, arrY; 
                    real2arr(xy.x, xy.y, arrX, arrY);

                    if (v != 200 && u != 200) {
                        if (AVM_seg_right_gray.at<uchar>(int(v), int(u)) == 255 || AVM_seg_right_gray.at<uchar>(int(v), int(u)) == 17 && (v != 200 && u != 200))  
                            num_obsR[arrX][arrY]++;
                        if (AVM_seg_right_gray.at<uchar>(int(v), int(u)) == 17 && (v != 200 && u != 200)) 
                            num_freeR[arrX][arrY]++;
                    }
                }
            }

        AVM_seg_right_gray.at<cv::Vec3b>(200, 200)[0] = 0;   //b
        AVM_seg_right_gray.at<cv::Vec3b>(200, 200)[1] = 0;   //g
        AVM_seg_right_gray.at<cv::Vec3b>(200, 200)[2] = 0;   //r
    }
}

void AVMpointCloud(cv::Mat img) {
    int avmCutRange = 0, idxSparse = 1;
    if (m_flagDR) {idxSparse = 3; avmCutRange = 75;}

    // if (m_DRcnt%3 == 0) {
        m_DRcnt = 0;
        for(int i = avmCutRange ; i < img.size().height - avmCutRange ; i = i+idxSparse){
            for(int j = avmCutRange ; j < img.size().width -avmCutRange ; j = j+idxSparse){
                // if(!(img.at<cv::Vec3b>(i,j)[1] == 0) && !(img.at<cv::Vec3b>(i,j)[0] == 0) && !(img.at<cv::Vec3b>(i,j)[2] == 0)) {   
                    double x = 12.5 - 0.0625 * i, y = 12.5 - 0.0625 * j, gX, gY;
                    Local2Global(x, y, gX, gY);

                    pcl::PointXYZRGB pt;
                    pt.x = gX;  pt.y = gY;  pt.z = 0.0;

                    uint8_t r = static_cast<uint8_t>(img.at<cv::Vec3b>(i,j)[2]), g = static_cast<uint8_t>(img.at<cv::Vec3b>(i,j)[1]), b = static_cast<uint8_t>(img.at<cv::Vec3b>(i,j)[0]);
                    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
                    pt.rgb = *reinterpret_cast<float*>(&rgb);

                    m_avm_data->push_back(pt);
                // }
            }
        }
    // }

    if (m_flagDR) {
        if (m_avm_data->size() > 500000)
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
        for(int j=0 ;j < m_gridDim ; j++) {
            if (num_obsL[j][i] > 0 || num_obsR[j][i] > 0)   {
                occupancyGridMap.data[cnt] = 100;

                if (withinpoint(j, i) == 1)
                    occupancyGridMap.data[cnt] = 0;
            }
            else                                            
                occupancyGridMap.data[cnt] = 0;

            if (num_freeL[j][i] > 0 || num_freeR[j][i] > 0)   {
                if (withinpoint_free(j, i) == 1)
                    occupancyGridMap.data[cnt] = 0;
            }
            cnt++;
        }
    // for(int i=0; i < m_gridDim ; i++) 
    //     for(int j=0 ;j < m_gridDim ; j++) 
    //         // if (num_obsL[j][i] > 0 || num_obsR[j][i] > 0)   {
    //         if ((j == modx1 && i == mody1) ||
    //             (j == modx2 && i == mody2) ||
    //             (j == modx3 && i == mody3) ||
    //             (j == modx4 && i == mody4) )   {
    //             occupancyGridMap.data[cnt] = 100;

    //             cnt++;
    //         }
    //         else                                            
    //             occupancyGridMap.data[cnt++] = 0;

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
    Sub_phantom_side_left       = nodeHandle.subscribe("/csi_cam/side_left/image_raw", 1, CallbackPhantom_left);
    Sub_phantom_side_right      = nodeHandle.subscribe("/csi_cam/side_right/image_raw", 1, CallbackPhantom_right);
    Sub_phantom_front_center    = nodeHandle.subscribe("/csi_cam/front_center/image_raw", 1 , CallbackPhantom_center);
    Sub_phantom_rear_center     = nodeHandle.subscribe("/csi_cam/rear_center/image_raw", 1, CallbackPhantom_rear);

    // sub_tmp = nodeHandle.subscribe("/phantomvision/phantomnets", 100, CallbackPhantom_temp);

    // Sub_phantom_side_left           = nodeHandle.subscribe("/parking/phantomnet/side_left", 1, CallbackPhantom_left);
    // Sub_phantom_side_right          = nodeHandle.subscribe("/parking/phantomnet/side_right", 1, CallbackPhantom_right);
    // Sub_phantom_front_center_svm    = nodeHandle.subscribe("/phantomnet/output/front_center_svm", 1 , CallbackPhantom_center);
    // Sub_phantom_rear_center_svm     = nodeHandle.subscribe("/phantomnet/output/rear_center_svm", 1, CallbackPhantom_rear);

    Sub_phantom_left_seg    = nodeHandle.subscribe("/parking/phantomnet/side_left", 1 , CallbackPhantom_seg_left);
    Sub_phantom_right_seg   = nodeHandle.subscribe("/parking/phantomnet/side_right", 1 , CallbackPhantom_seg_right);

    Sub_phantom_DR_Path = nodeHandle.subscribe("/LocalizationData", 1 , CallbackPhantom_DR);

    Sub_parkingGoal = nodeHandle.subscribe("/parking_cands", 1, CallbackParkingGoal);

    Pub_AVM_img          = nodeHandle.advertise<sensor_msgs::Image>("/AVM_image", 1);
    Pub_AVM_seg_img      = nodeHandle.advertise<sensor_msgs::Image>("/AVM_seg_image", 1);
    Pub_AVM_seg_img_gray = nodeHandle.advertise<sensor_msgs::Image>("/AVM_seg_image_gray", 1);
    Pub_AVM_DR           = nodeHandle.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/AVM_image_DR", 1);

    occupancyGridMap.header.frame_id = "map";
    occupancyGridMap.info.resolution = m_gridResol;
    occupancyGridMap.info.width = occupancyGridMap.info.height = m_gridDim;
    occupancyGridMap.info.origin.position.x = occupancyGridMap.info.origin.position.y = -m_dimension/2 - m_gridResol*2;
    occupancyGridMap.info.origin.position.z = 0.1;
    occupancyGridMap.data.resize(occupancyGridMap.info.width*occupancyGridMap.info.width);
    Pub_occupancyGridMap = nodeHandle.advertise<nav_msgs::OccupancyGrid>("/occ_map", 1);

    m_avm_data->clear();
    m_avm_data->header.frame_id = "map";
    
    ros::Rate loop_rate(100);
    // ros::spin();

/////////////////////////////////////////////////////////////////////////////////////////////////////
//@Yangwoo
    
    m_car.x = m_car.y = m_car.th = 0.0;

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

        // if (!m_flagDR)
            AVMpointCloud(aggregated_img);

        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}