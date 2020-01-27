#include "pch.h"
#include <sstream>
#include <string>
#include <iostream>
#include <opencv\cv.h>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>



using namespace System;
using namespace System::IO::Ports;
using namespace std;
using namespace cv;


//initial min and max HSV filter values.
//these will be changed using trackbars
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;

//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;

//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS = 5;

//minimum and maximum object area
const int MIN_OBJECT_AREA = 10 * 10;
const int MAX_OBJECT_AREA = FRAME_HEIGHT * FRAME_WIDTH / 1.5;

//names that will appear at the top of each window
const string windowName = "Original image";
const string HSVWindowName = "HSV image";
const string binaryWindowName = "Binary image";
const string trackbarWindowName = "Trackbars";


void on_trackbar(int, void*)
{
	//This function gets called whenever a
	// trackbar position is changed
}

string intToString(int number) {
	std::stringstream ss;
	ss << number;
	return ss.str();
}

void createTrackbars() {
	//create window for trackbars

	namedWindow(trackbarWindowName, 0);

	char TrackbarName[50];
	sprintf_s(TrackbarName, "H_MIN");
	sprintf_s(TrackbarName, "H_MAX");
	sprintf_s(TrackbarName, "S_MIN");
	sprintf_s(TrackbarName, "S_MAX");
	sprintf_s(TrackbarName, "V_MIN");
	sprintf_s(TrackbarName, "V_MAX");

	createTrackbar("H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar);
	createTrackbar("H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar);
	createTrackbar("S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar);
	createTrackbar("S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar);
	createTrackbar("V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar);
	createTrackbar("V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar);
}

void drawObject(int x, int y, Mat& frame) {

	//use some of the openCV drawing functions to draw crosshairs
	//on your tracked image!

	circle(frame, Point(x, y), 20, Scalar(0, 255, 0), 2);
	if (y - 25 > 0)
		line(frame, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
	if (y + 25 < FRAME_HEIGHT)
		line(frame, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, FRAME_HEIGHT), Scalar(0, 255, 0), 2);
	if (x - 25 > 0)
		line(frame, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
	if (x + 25 < FRAME_WIDTH)
		line(frame, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(FRAME_WIDTH, y), Scalar(0, 255, 0), 2);

	putText(frame, intToString(x) + "," + intToString(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);

}

void morphOps(Mat& thresh) {

	//create structuring element that will be used to "dilate" and "erode" image.
	//the element chosen here is a 3px by 3px rectangle

	Mat erodeElement = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	//dilate with larger element so make sure object is nicely visible
	Mat dilateElement = getStructuringElement(MORPH_ELLIPSE, Size(8, 8));

	erode(thresh, thresh, erodeElement);
	erode(thresh, thresh, erodeElement);


	dilate(thresh, thresh, dilateElement);
	dilate(thresh, thresh, dilateElement);



}

void trackFilteredObject(int& x, int& y, Mat threshold, Mat& cameraFeed, SerialPort^ port) {


	int xCenter = FRAME_WIDTH / 2;
	int yCenter = FRAME_HEIGHT / 2;
	bool yCentered = false;
	bool xCentered = false;
	bool centered = false;	

	Mat temp;
	threshold.copyTo(temp);
	//these two vectors needed for output of findContours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	//find contours of filtered image using openCV findContours function
	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	//use moments method to find our filtered object
	double refArea = 0;
	bool objectFound = false;
	if (hierarchy.size() > 0) {
		int numObjects = hierarchy.size();

		if (numObjects > 1) {
			putText(cameraFeed, "Please remove other projections", Point(0, 350), 2, 1, Scalar(0, 0, 255), 1);
			return;
		}

		//if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
		if (numObjects < MAX_NUM_OBJECTS) {

				for (int index = 0; index >= 0; index = hierarchy[index][0]) {

					Moments moment = moments((cv::Mat)contours[index]);
					double area = moment.m00;

					//if the area is less than 10 px by 10 px then it is probably just noise
					//if the area is the same as the 3/2 of the image size, probably just a bad filter
					//we only want the object with the largest area so we safe a reference area each
					//iteration and compare it to the area in the next iteration.
					if (area > MIN_OBJECT_AREA&& area<MAX_OBJECT_AREA && area>refArea) {
						x = moment.m10 / area;
						y = moment.m01 / area;						
						objectFound = true;
						refArea = area;

					}
					else objectFound = false;
				}
			
			if (objectFound) {
				drawObject(x, y, cameraFeed);

				string modifiedX = intToString(x);
				if (modifiedX.length() == 1) {
					modifiedX = string(2, '0').append(modifiedX);
				}
				else if (modifiedX.length() == 2) {
					modifiedX = string(1, '0').append(modifiedX);					
				}

				string modifiedY = intToString(y);
				if (modifiedY.length() == 1) {
					modifiedY = string(2, '0').append(modifiedY);
				}
				else if (modifiedY.length() == 2) {
					modifiedY = string(1, '0').append(modifiedY);
				}

				string serialString = "location-" + modifiedX + "-" + modifiedY + "-move";

				System::String^ serialCommand = gcnew System::String(serialString.c_str());
				port->WriteLine(serialCommand);
				delete serialCommand;
				serialCommand = nullptr;

				if (y < 120) {
					putText(cameraFeed, "High speed forward", Point(0, 350), 2, 1, Scalar(0, 255, 0), 1);
					if (x < (FRAME_WIDTH / 2) - 30) {
						putText(cameraFeed, "Turn left", Point(0, 400), 2, 1, Scalar(0, 255, 0), 1);
					}
					if (x > (FRAME_WIDTH / 2) + 30) {
						putText(cameraFeed, "Turn right", Point(0, 400), 2, 1, Scalar(0, 255, 0), 1);
					}
				}
				else if (y > 120 && y < 220) {
					putText(cameraFeed, "Medium speed forward", Point(0, 350), 2, 1, Scalar(0, 255, 0), 1);
					if (x < (FRAME_WIDTH / 2) - 30) {
						putText(cameraFeed, "Turn left", Point(0, 400), 2, 1, Scalar(0, 255, 0), 1);
					}
					if (x > (FRAME_WIDTH / 2) + 30) {
						putText(cameraFeed, "Turn right", Point(0, 400), 2, 1, Scalar(0, 255, 0), 1);
					}
				}
				else if (y > 220 && y < 320) {
					putText(cameraFeed, "Low speed forward", Point(0, 350), 2, 1, Scalar(0, 255, 0), 1);
					if (x < (FRAME_WIDTH / 2) - 30) {
						putText(cameraFeed, "Turn left", Point(0, 400), 2, 1, Scalar(0, 255, 0), 1);
					}
					if (x > (FRAME_WIDTH / 2) + 30) {
						putText(cameraFeed, "Turn right", Point(0, 400), 2, 1, Scalar(0, 255, 0), 1);
					}
				}
				else {
					putText(cameraFeed, "Robot in position", Point(0, 350), 2, 1, Scalar(243, 226, 68), 1);
				}	
			}
		}
		else putText(cameraFeed, "Too much noise", Point(0, 350), 2, 1, Scalar(0, 0, 255), 1);
	}
	else {
		port->WriteLine("location-000-000-stop");
		putText(cameraFeed, "No laser detected", Point(0, 350), 2, 1, Scalar(0, 0, 255), 1);
	}
}

int main(int argc, char* argv[])
{
	
	//open arduino connection 
	SerialPort^ port = gcnew SerialPort("COM10", 9600);
	port->Open();
	cout << "Serial open" << endl;

	bool trackObjects = true;
	bool useMorphOps = true;

	//Matrix to store each frame of the webcam feed
	Mat cameraFeed;

	//matrix storage for HSV image
	Mat HSV;

	//matrix storage for binary threshold image
	Mat threshold;

	//x and y values for the location of the object
	int x = 0, y = 0;

	//create slider bars for HSV filtering
	createTrackbars();

	//video capture object to acquire webcam feed
	VideoCapture capture;

	//open capture object at location zero (default location for webcam)
	capture.open(1);

	// check if the connection succeeded, or terminate the program
	if (!capture.isOpened()) return -1;

	//set height and width of capture frame
	capture.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);	

	//start an infinite loop where webcam feed is copied to cameraFeed matrix
	while (1) {
		//store image to matrix
		capture.read(cameraFeed);

		//draw grid
		line(cameraFeed, Point(FRAME_WIDTH/2, 0), Point(FRAME_WIDTH/2, FRAME_HEIGHT - (FRAME_HEIGHT/3)), Scalar(243, 226, 68), 1);
		line(cameraFeed, Point(0, FRAME_HEIGHT - (FRAME_HEIGHT / 3)), Point(FRAME_WIDTH, FRAME_HEIGHT - (FRAME_HEIGHT / 3)), Scalar(243, 226, 68), 1);
		line(cameraFeed, Point(FRAME_WIDTH / 2 -30, 0), Point(FRAME_WIDTH / 2-30, FRAME_HEIGHT - (FRAME_HEIGHT / 3)), Scalar(255, 255, 255), 1);
		line(cameraFeed, Point(FRAME_WIDTH / 2 + 30, 0), Point(FRAME_WIDTH / 2 + 30, FRAME_HEIGHT - (FRAME_HEIGHT / 3)), Scalar(255, 255, 255), 1);
		

		//convert frame from BGR to HSV colorspace
		cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
		medianBlur(HSV, HSV, 3);

		//filter HSV image between values and store filtered image to threshold matrix		
		//inRange(HSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold);

		//for red circular laser pointer (lower S_MAX)
		inRange(HSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, 10, V_MAX), threshold);

		//perform morphological operations on thresholded image to eliminate noise
		//and emphasize the filtered object(s)
		if (useMorphOps) morphOps(threshold);

		//pass in thresholded frame to our object tracking function
		//this function will return the x and y coordinates of the
		//filtered object
		if (trackObjects) trackFilteredObject(x, y, threshold, cameraFeed, port);



		//show frames 
		imshow(binaryWindowName, threshold);
		imshow(windowName, cameraFeed);
		imshow(HSVWindowName, HSV);


		//delay 10ms so that screen can refresh.
		waitKey(10);
	}

	delete port;
	port = nullptr;

	return 0;
}

