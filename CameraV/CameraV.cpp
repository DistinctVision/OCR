// CameraV.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include <conio.h>
#include <cv.h>
#include <highgui.h>
#include <stdlib.h>
#include <stdio.h>
#include <Windows.h>

#include "ImageProcessor.h"

int iter = 0;

CvCapture* capture;
ImageProcessor* procImage;

void myTrackbarCallback(int pos) 
{
	procImage->setThreshold(pos - 256);
}

void myMouseCallback( int event, int x, int y, int flags, void* param )
{
	char name[50];
    IplImage* img = (IplImage*) param;
    switch( event )
	{
        case CV_EVENT_MOUSEMOVE: 
			break;

        case CV_EVENT_LBUTTONDOWN:
			procImage->addFindedObject(cvPoint(x, y), name);
            break;

        case CV_EVENT_LBUTTONUP:
            break;
    }
}

int _tmain(int argc, _TCHAR* argv[])
{
	setlocale(LC_ALL,"Russian");//устанавливаем русский язык
	// получаем любую подключённую камеру
    capture = cvCreateCameraCapture(CV_CAP_ANY);
    assert( capture );//проверяем захват камеры

	//устанавливаем разрешение изображения с камеры
    /*cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 640); 
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 480); */

    // узнаем ширину и высоту кадра
	double width = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
    double height = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);

	//IplImage* im = cvLoadImage("1.jpg");
	IplImage* frame = NULL;//это будет наш кадр

    cvNamedWindow("capture", CV_WINDOW_AUTOSIZE);//окно для вывода
	int tPosition = 256 - 10;
	cvCreateTrackbar("Threshold", "capture", &tPosition, 512, myTrackbarCallback);
	cvSetMouseCallback("capture", myMouseCallback, (void*) frame);

    printf("[info] press Enter for capture image and Esc for quit!\n\n");
	
    int counter=0;

	bool debugFindedMap = false, debugFoundedMap = false, debugProcessImage = false;

	procImage = new ImageProcessor();//экзепляр класса для обработки
	procImage->setSize((int)width, (int)height);//нужно указать параметры кадра
	procImage->openData();//открываем информацию уже об обученных символах 

    for(;;)
	{
        // получаем кадр
		//frame = cvCloneImage(im);
        frame = cvQueryFrame(capture);
		//cvSmooth(frame, frame, CV_GAUSSIAN, 3, 3);
		// ищем символы
		procImage->process(frame);
		// рисуем результат на нашем изображении
		procImage->drawInfo();
        // показываем кадр
        cvShowImage("capture", frame);
        char c = cvWaitKey(3);
		if (c == 27)  // нажата ESC
			   break;
		else if (c == 32)
			cvWaitKey();
#if ((Debug_OpenCV > 0) && (Debug_OpenCV <= 4))
		else if (c == '1')
		{
			if (debugProcessImage) debugProcessImage = false;
			else debugProcessImage = true;
			procImage->setDebugProcessImage(debugProcessImage);
		}
#endif
#if (Debug_OpenCV_Map == 1)
		else if (c == '2')
		{
			if (debugFoundedMap) debugFoundedMap = false;
			else debugFoundedMap = true;
			procImage->setDebugFoundedMap(debugFoundedMap);
		}
#endif
#if (Debug_OpenCV_FindedObject == 1)
		else if (c == '3')
		{
			if (debugFindedMap) debugFindedMap = false;
			else debugFindedMap = true;
			procImage->setDebugFindedMap(debugFindedMap);

		}
#endif
		else if (c == '0')
			procImage->clearFindedObject();
		iter++;
    }
    // освобождаем ресурсы
	procImage->saveData();//сохраняем информацию об обученных символах
	delete procImage;//удаляем обработчик изображения
    cvReleaseCapture( &capture );//удаляем захват камеры
    cvDestroyWindow("capture");//удаляем окно
    return 0;
}

