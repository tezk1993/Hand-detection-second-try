#include "opencv2/opencv.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>

#define PI 3.1415
using namespace cv;
using namespace std;

int minH = 0;
int maxH = 179;

int minS = 0;
int maxS = 255;

int minV = 0;
int maxV = 255;

int erossion = 5;
int dilation = 5;
Mat cameraFrame, blurFrame, threshold1, threshold2, closedFrame, hsvFrame, colorObjectFrame, thresholdFrame;
VideoCapture stream1;
Mat grayscale;
Mat fgMaskMOG;
Mat foreground;
Mat background;
Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(20, 16, true);



void show_windows(Mat background, Mat cameraFrame, Mat foreground, Mat thresholdFrame,Mat hsvFrame){
	imshow("Background", background); //show the original image
	imshow("cameraFrame", cameraFrame); //show the original image
	imshow("foreGround", foreground); //show the original image
	imshow("threshold", thresholdFrame); //show the original image
	imshow("hsv", hsvFrame);
	
	// the camera will be deinitialized automatically in VideoCapture destructor

}

void morphologicalImgProc(Mat &frame) {
	Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(9, 9), Point(5, 5));
	Mat element1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(7, 7), Point(5, 5));
	dilate(frame, frame, element);
	erode(frame, frame, element);
	morphologyEx(frame, frame, MORPH_OPEN, element);
	morphologyEx(frame, frame, MORPH_CLOSE, element);
}

void hand_detection(Mat src, Mat dest){
	Rect boundRect;
	int largestObj;
	int boundingBoxHeight = 0;
	vector<vector<Point> > contours; //store all the contours
	vector<vector<Point> > contoursSet(contours.size());//store large contours
	vector<Vec4i> hierarchy;
	vector<Point> convexHullPoint;
	vector<Point> fingerPoint;
	Point centerP;
	int numObjects = 0;
	double area = 0;
	double maxArea = 0;
	bool handFound = false;


	findContours(src, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	numObjects = hierarchy.size();
	for (unsigned int i = 0; i < contours.size(); i++) {
		Mat tempContour = Mat(contours[i]);
		int area = contourArea(tempContour);
		if (area > maxArea) {
			maxArea = area;
			largestObj = i;
		}
	}
	if (maxArea > 4000){
		handFound = true;
		boundRect = boundingRect(contours[largestObj]);
		//draw the boundary of the object
		drawContours(dest, contours, largestObj, Scalar(0, 0, 255), 3, 8, hierarchy);
		//find the convex points for the largest object which is hand
		convexHull(contours[largestObj], convexHullPoint, true, true);
		approxPolyDP(Mat(contours[largestObj]), contours[largestObj], 3, true);
		//use moment method to find the center point
		Moments moment = moments(Mat(contours[largestObj]), true);
		int centerX = moment.m10 / moment.m00;
		int centerY = moment.m01 / moment.m00;
		Point centerPoint(centerX, centerY);
		centerP = centerPoint;
		Point printPoint(centerX, centerY + 15);
		Point printPoint1(boundRect.x, boundRect.y);
		circle(dest, centerPoint, 8, Scalar(255, 0, 0), CV_FILLED);
		//put the BoundingBox in the contour region
		rectangle(dest, boundRect, Scalar(0, 0, 255), 2, 8, 0);
		boundingBoxHeight = boundRect.height;
		//if( boundingBoxHeight <= 200)
		//	handFound = false;
		if (handFound) {
			int countHullPoint = convexHullPoint.size();
			int maxdist = 0;
			int pos = 0;
			for (int j = 1; j < countHullPoint; j++) {
				pos = j;
				if (centerP.y >= convexHullPoint[j].y && centerP.y >= convexHullPoint[pos].y) {
					pos = j;
					int dist = (centerP.x - convexHullPoint[j].x) ^ 2 + (centerP.y - convexHullPoint[j].y) ^ 2;
					if (abs(convexHullPoint[j - 1].x - convexHullPoint[j].x) < 12){
						if (dist > maxdist){
							maxdist = dist;
							pos = j;
						}
					}
					else if (j == 0 || abs(convexHullPoint[j - 1].x - convexHullPoint[j].x) >= 12){
						fingerPoint.push_back(convexHullPoint[pos]);
						cv::line(dest, centerP, convexHullPoint[pos], Scalar(0, 255, 0), 3, CV_AA, 0);
						circle(dest, convexHullPoint[pos], 8, Scalar(255, 0, 0), CV_FILLED);
						pos = j;
					}

				}
			}
		}
	}
}

void trackbar(int &minH, int &maxH, int &minS, int &maxS, int &minV, int &maxV)
{
	namedWindow("Control", CV_WINDOW_KEEPRATIO);

	cvCreateTrackbar("LowH", "Control", &minH, 179); //Hue (0 - 179)
	cvCreateTrackbar("HighH", "Control", &maxH, 179);

	cvCreateTrackbar("LowS", "Control", &minS, 255); //Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Control", &maxS, 255);

	cvCreateTrackbar("LowV", "Control", &minV, 255); //Value (0 - 255)
	cvCreateTrackbar("HighV", "Control", &maxV, 255);
}

int main(int, char)
{
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened()) // check if we succeeded
		return -1;
	
	
	cap.read(background);

	while (true)
	{

		bool bSuccess = cap.read(cameraFrame); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}
		
		trackbar(minH, maxH, minS, maxS, minV, maxV);

		pMOG2->apply(cameraFrame, foreground);

		medianBlur(foreground, foreground, 3);
		
		cvtColor(foreground, foreground, CV_GRAY2BGR);

		cvtColor(foreground, hsvFrame, CV_BGR2HSV);

		inRange(hsvFrame, Scalar(0, 0, 255), Scalar(256, 256, 256), thresholdFrame);

		//inRange(hsvFrame, Scalar(minH, minS, minV), Scalar(maxH, maxS, maxV), thresholdFrame);

		medianBlur(thresholdFrame, thresholdFrame, 5);
		
		morphologicalImgProc(thresholdFrame);

		hand_detection(thresholdFrame,cameraFrame);
		
		show_windows(background, cameraFrame, foreground, thresholdFrame,hsvFrame);

		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}
	return 0;
}


