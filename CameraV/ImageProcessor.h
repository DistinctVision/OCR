#pragma once

#include <stdio.h>
#include <cv.h>
#include <math.h>
#include <string>
#include <Windows.h>

#define Debug_OpenCV 2//4, 3, 2, 1
#define Debug_OpenCV_FindedObject 1//1
#define Debug_OpenCV_Map 1//1

#if ((Debug_OpenCV_Map == 1) || (Debug_OpenCV_FindedObject == 1))
#include <highgui.h>
#endif

#if (Debug_OpenCV_FindedObject == 1)
#define maxCountFindedObject 10
#endif

#define MAXNUMBER 999999.0f
#define MINNUMBER -MAXNUMBER

#define maxCountSegment 800
#define maxCountPrevResult 200
#define deltaMapSize 11
#define mapSize (deltaMapSize+deltaMapSize+1)
#define maxCountSizeMap 16

inline bool operator == (CvPoint& p1, CvPoint& p2);

inline int fabs(int a)
{
	if (a >= 0) return a;
	return (-a);
}

class ImageProcessor
{
public:
	typedef struct
	{
		bool active;
		float cx, cy;
		int count;
		float ux, uy, uxy;
		int fObject;
		float probability;
		bool inverse;
		int minX, minY;//AABB
		int maxX, maxY;//
		float localMinX, localMaxX;//localAABB
		float localMinY, localMaxY;//
		float scale;
		float angle;
		float e;
		bool map[mapSize][mapSize];
	} Segment;
	typedef struct
	{
		float map[mapSize][mapSize];
		int iter;
	} Traning;
	typedef struct
	{
		float e;
		bool map[mapSize][mapSize];
		int countFillPoint;
		char name[50];
		Traning* traning;
	} FindedObject;
	Segment* mSegment;
	int countSegment;

private:
	typedef struct
	{
		bool value;
		CvPoint parent;
		float cx, cy;
		int count;
		int minX, minY;
		int maxX, maxY;
	} Element;
	typedef struct
	{
		bool active;
		int fObject;
		int minX, minY;
		int maxX, maxY;
	} PrevResult;
	PrevResult* mPrevResult;
	int countPrevResult;
	uchar** bwImage;
	int** integralImage;
	Element** image;
	CvSize size;
	int minSize, maxSize;
	IplImage* scrImage;
	int threshold_C;
	uchar threshold;
	int regionArea;
	int regionSize;
	float epsEccentricity;
	FindedObject* findedObject;
	int countFindedObject;
	int traningTime;
	float speedTraning;
	float limitTraning;
	float probabilityTraning;
	float probabilityA;
	float probabilityB;
	float probabilityC;
	float halfScalePoint;
	int epsPrevResult;
#if (Debug_OpenCV_Map == 1)
	IplImage* map;
	bool debugMap;
#endif
#if (Debug_OpenCV_FindedObject == 1)
	IplImage* findedMap;
	bool debugFindedMap;
	CvFont font;
#endif
#if ((Debug_OpenCV > 0) && (Debug_OpenCV <= 4))
	bool debugImage;
#endif
#if ((Debug_OpenCV > 0) && (Debug_OpenCV <= 2))
	uchar* r;
	uchar* g;
	uchar* b;
#endif
	CvFont fontText;
	inline CvPoint getLastParent(CvPoint& parent)
	{
		CvPoint beforeParent = image[parent.x][parent.y].parent;
		if (beforeParent.x < 0)
			return parent;
		return getLastParent(beforeParent);
	};
	inline uchar solveColor(uchar& R, uchar& G, uchar& B)
	{
		return (uchar)((0.11f * R) + (0.39f * G) + (0.5f * B));
		//return (uchar)((0.5f * R) + (0.39f * G) + (0.11f * B));
		//return (R + G + B) / 3;
	}
	inline bool valid(uchar& color)
	{
		if (color <= threshold) return true;
		return false;
	}
	inline void clearMapSegment(int i);
	inline void solveRegion(CvPoint& p);
	inline void step0();
	inline void step1();
	inline void step2();
	inline void step3();
	inline void step4();
	inline void initPrevResult();
	inline float compareSegment(int nSegment, int nFindedObject);
	inline float compareSegmentInverse(int nSegment, int nFindedObject);
	inline void compareAllObjects();
	inline void updateTraning();
	inline void startTraningFindedObject(int nFindedObject);
	inline void addFindedObject(int nSegment, char name[50]);
	//inline void deleteFindedObject(int i);
	inline void deleteFindedObject(int i);
public:
	ImageProcessor();
	~ImageProcessor();
	void openData();
	void saveData();
	void setSize(int w, int h);
	void addFindedObject(CvPoint p, char name[50]);
	void deleteFindedObject(char* s);
	void setThreshold(int a)
	{
		threshold_C = a;
	}
	void process(IplImage* scr);
	void drawInfo();
	void clearFindedObject()
	{
		countFindedObject = 0;
		printf("Все образы удалены.\n");
	}
#if (Debug_OpenCV_Map == 1)
	void setDebugFoundedMap(bool d)
	{
		debugMap = d;
	}
#endif
#if (Debug_OpenCV_FindedObject == 1)
	void setDebugFindedMap(bool d)
	{
		debugFindedMap = d;
	}
#endif
#if ((Debug_OpenCV > 0) && (Debug_OpenCV <= 4))
	void setDebugProcessImage(bool d)
	{
		debugImage = d;
	}
#endif
};

