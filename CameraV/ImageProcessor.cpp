#include "StdAfx.h"
#include "ImageProcessor.h"

inline bool operator == (CvPoint& p1, CvPoint& p2)
{
	if (p1.x != p2.x) return false;
	if (p1.y != p2.y) return false;
	return true;
}

ImageProcessor::ImageProcessor()
{
#if (Debug_OpenCV_Map == 1)
	map = cvCreateImage(cvSize(mapSize * maxCountSizeMap, mapSize * maxCountSizeMap), 8, 1);
	debugMap = false;
	cvNamedWindow("Founded Object", CV_WINDOW_AUTOSIZE);
#endif
#if (Debug_OpenCV_FindedObject == 1)
	findedMap = cvCreateImage(cvSize(mapSize * maxCountFindedObject, mapSize + 20), 8, 1);
	cvNamedWindow("Finded Object", CV_WINDOW_AUTOSIZE);
	cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 0.5, 0.5, 0, 1, CV_AA);
	debugFindedMap = false;
#endif
#if ((Debug_OpenCV > 0) && (Debug_OpenCV <= 4))
	debugImage = false;
#endif
#if ((Debug_OpenCV > 0) && (Debug_OpenCV <= 2))
	r = (uchar* )malloc(sizeof(uchar) * maxCountSegment);
	g = (uchar* )malloc(sizeof(uchar) * maxCountSegment);
	b = (uchar* )malloc(sizeof(uchar) * maxCountSegment);
	for (int i=0; i<maxCountSegment; i++)
	{
		r[i] = rand()%255;
		g[i] = rand()%255;
		b[i] = rand()%255;
	}
#endif
	cvInitFont(&fontText, CV_FONT_HERSHEY_COMPLEX, 0.5f, 0.5f, 0, 1, CV_AA);
	mSegment = (Segment* )malloc(sizeof(Segment) * maxCountSegment);
	mPrevResult = (PrevResult* )malloc(sizeof(PrevResult) * maxCountPrevResult);
	size.width = size.height = 0;
	image = NULL;
	scrImage = NULL;
	findedObject = NULL;
	countSegment = 0;
	countFindedObject = 0;
	countPrevResult = 0;

	minSize = 10;
	maxSize = 250;
	threshold_C = -25;
	regionSize = 12; regionArea = regionSize * regionSize * 4;
	epsEccentricity = 5.0f;
	traningTime = 1;
	speedTraning = 0.0;//15f;
	limitTraning = 0.5f;
	probabilityTraning = 0.91f;
	probabilityA = 0.93f;
	probabilityB = 0.84f;
	probabilityC = 0.78f;
	epsPrevResult = 20;
	halfScalePoint = 1.4142f * 0.5f;
}

ImageProcessor::~ImageProcessor()
{
	free(findedObject);
	free(mSegment);
	free(mPrevResult);
	if (image)
	{
		for (int i=0; i<size.width; i++)
		{
			free(bwImage[i]);
			free(integralImage[i]);
			free(image[i]);
		}
		free(bwImage);
		free(integralImage);
		free(image);
#if ((Debug_OpenCV > 0) && (Debug_OpenCV <= 2))
		free(r);
		free(g);
		free(b);
#endif		
	}
#if (Debug_OpenCV_Map == 1)
	cvDestroyWindow("Founded Object");
	free(map);
#endif
#if (Debug_OpenCV_FindedObject == 1)
	cvDestroyWindow("Finded Object");
	free(findedMap);
#endif
}

void ImageProcessor::setSize(int w, int h)
{
	int i;
	if (image)
	{
		for (i=0; i<size.width; i++)
		{
			free(image[i]);
			free(bwImage[i]);
			free(integralImage[i]);
		}
		free(image);
		free(bwImage);
		free(integralImage);
	}
	size.width = w;
	size.height = h;
	image = (Element** )malloc(sizeof(Element*) * size.width);
	bwImage = (uchar** )malloc(sizeof(uchar*) * size.width);
	integralImage = (int** )malloc(sizeof(int*) * size.width);
	for (i=0; i<size.width; i++)
	{
		image[i] = (Element* )malloc(sizeof(Element) * size.height);
		bwImage[i] = (uchar* )malloc(sizeof(uchar) * size.height);
		integralImage[i] = (int*)malloc(sizeof(int) * size.height);
	}
}

inline void ImageProcessor::clearMapSegment(int i)
{
	CvPoint p;
	for (p.x=0; p.x<mapSize; p.x++)
		for (p.y=0; p.y<mapSize; p.y++)
			mSegment[i].map[p.x][p.y] = false;
}

inline void ImageProcessor::solveRegion(CvPoint& p)
{
#if (Debug_OpenCV != 4)
	CvPoint begin, end;
	begin.x = p.x - regionSize;
	if (begin.x < 0)
	{
		end.x = p.x + regionSize - begin.x;
		begin.x = 0;
	}
	else
	{
		end.x = p.x + regionSize;
		if (end.x > (size.width-1))
		{
			begin.x -= end.x - size.width + 1;
			end.x = size.width - 1;
		}
	}
	begin.y = p.y - regionSize;
	if (begin.y < 0)
	{
		end.y = p.y + regionSize - begin.y;
		begin.y = 0;
	}
	else
	{
		end.y = p.y + regionSize;
		if (end.y > (size.height-1))
		{
			begin.y -= end.y - size.height + 1;
			end.y = size.height - 1;
		}
	}
	int A = (integralImage[begin.x][begin.y]+integralImage[end.x][end.y]) - 
		(integralImage[begin.x][end.y] + integralImage[end.x][begin.y]);
	A = (A / regionArea) + threshold_C;
	if (A > 255) A = 255;
	if (A < 0) A = 0;
	threshold = A;
#else
	threshold = threshold_C;
#endif
}

inline void ImageProcessor::initPrevResult()
{
	countPrevResult = 0;
	for (int i=0; i<countSegment; i++)
		if (mSegment[i].active)
		{
			mPrevResult[countPrevResult].active = false;
			mPrevResult[countPrevResult].fObject = mSegment[i].fObject;
			mPrevResult[countPrevResult].maxX = mSegment[i].maxX;
			mPrevResult[countPrevResult].maxY = mSegment[i].maxY;
			mPrevResult[countPrevResult].minX = mSegment[i].minX;
			mPrevResult[countPrevResult].minY = mSegment[i].minY;
			countPrevResult++;
		}
}

inline void ImageProcessor::step0()
{
	uchar* ptr;
	uchar* elementPtr;
	CvPoint p;
	ptr = (uchar* )(scrImage->imageData);
	bwImage[0][0] = solveColor(ptr[2], ptr[1], ptr[0]);
	integralImage[0][0] = bwImage[0][0];
	for (p.x=1; p.x<size.width; p.x++)
	{
		elementPtr = &ptr[p.x * 3];
		bwImage[p.x][0] = solveColor(elementPtr[2], elementPtr[1], elementPtr[0]);
		integralImage[p.x][0] = integralImage[p.x-1][0] + bwImage[p.x][0];
	}
	for (p.y=1; p.y<size.height; p.y++)
	{
		elementPtr = (uchar* )(scrImage->imageData + scrImage->widthStep * p.y);
		bwImage[0][p.y] = solveColor(elementPtr[2], elementPtr[1], elementPtr[0]);
		integralImage[0][p.y] = integralImage[0][p.y-1] + bwImage[0][p.y];
	}
	for (p.y=1; p.y<size.height; p.y++)
	{
		ptr = (uchar* )(scrImage->imageData + p.y * scrImage->widthStep);
		for (p.x=1; p.x<size.width; p.x++)
		{
			elementPtr = &ptr[p.x * 3];
			bwImage[p.x][p.y] = solveColor(elementPtr[2], elementPtr[1], elementPtr[0]);
			integralImage[p.x][p.y] = integralImage[p.x][p.y-1] + integralImage[p.x-1][p.y] - integralImage[p.x-1][p.y-1] + bwImage[p.x][p.y];
		}
	}
}

inline void ImageProcessor::step1()
{
	CvPoint p, parent, parent2;
#if ((Debug_OpenCV == 3) || (Debug_OpenCV == 4))
	uchar* ptr;
	ptr = (uchar* ) scrImage->imageData;
	uchar* elementPtr = ptr;
#endif
	p.x = p.y = 0;
#if (Debug_OpenCV == 4)
	threshold = 80;
#endif
	solveRegion(p);
	image[0][0].count = -1;
	if (valid(bwImage[0][0]))
	{
#if ((Debug_OpenCV == 3) || (Debug_OpenCV == 4))
		if (debugImage)
		{
			elementPtr[0] = 0;
			elementPtr[1] = 0;
			elementPtr[2] = 0;
		}
#endif
		image[0][0].value = true;
		image[0][0].parent.x = -1;//у него нет родительской метки, т.е. он же и является родителем
		image[0][0].parent.y = 1;// >= 0 - значит этот родитель еще не обрабатывался на 2ом шаге
		image[0][0].cx = 0.0f;
		image[0][0].cy = 0.0f;
		image[0][0].minX = image[0][0].minY = image[0][0].maxX = image[0][0].maxY = 0; 
	}
	else
	{
#if ((Debug_OpenCV == 3) || (Debug_OpenCV == 4))
		if (debugImage)
		{
			elementPtr[0] = 255;
			elementPtr[1] = 255;
			elementPtr[2] = 255;
		}
#endif
		image[0][0].value = false;
	}
	p.y = 0;
	//Верхняя полоса пикселов
	for (p.x=1; p.x<size.width; p.x++)
	{
#if ((Debug_OpenCV == 3) || (Debug_OpenCV == 4))
		if (debugImage)
			elementPtr = &ptr[p.x*3];
#endif
		solveRegion(p);
		image[p.x][p.y].count = -1;
		if (valid(bwImage[p.x][p.y]))
		{
#if ((Debug_OpenCV == 3) || (Debug_OpenCV == 4))
			if (debugImage)
			{
				elementPtr[0] = 0;
				elementPtr[1] = 0;
				elementPtr[2] = 0;
			}
#endif
			image[p.x][p.y].value = true;
			if (image[p.x-1][p.y].value)
			{
				parent.x = p.x-1;
				parent.y = p.y;
				parent = getLastParent(parent);
				image[p.x][p.y].parent = parent;
				image[parent.x][parent.y].count--;
				image[parent.x][parent.y].cx += (float)p.x;
				image[parent.x][parent.y].cy += (float)p.y;
				if (image[parent.x][parent.y].minX > p.x) image[parent.x][parent.y].minX = p.x;
				if (image[parent.x][parent.y].maxX < p.x) image[parent.x][parent.y].maxX = p.x;
				if (image[parent.x][parent.y].minY > p.y) image[parent.x][parent.y].minY = p.y;
				if (image[parent.x][parent.y].maxY < p.y) image[parent.x][parent.y].maxY = p.y;
			}
			else
			{
				image[p.x][p.y].parent.x = -1;//у него нет родительской метки, т.е. он же и является родителем
				image[p.x][p.y].parent.y = 1;// >= 0 - значит этот родитель еще не обрабатывался на 2ом шаге
				image[p.x][p.y].cx = (float)p.x;
				image[p.x][p.y].cy = (float)p.y;
				image[p.x][p.y].minX = image[p.x][p.y].maxX = p.x;
				image[p.x][p.y].minY = image[p.x][p.y].maxY = p.y;
			}
		}
		else
		{
#if ((Debug_OpenCV == 3) || (Debug_OpenCV == 4))
			if (debugImage)
			{
				elementPtr[0] = 255;
				elementPtr[1] = 255;
				elementPtr[2] = 255;
			}
#endif
			image[p.x][p.y].value = false;
		}
	}
	//Идем дальше (вниз)
	for (p.y=1; p.y<size.height; p.y++)
	{
		p.x = 0;
		//Первый пиксель в полосе
#if ((Debug_OpenCV == 3) || (Debug_OpenCV == 4))
		if (debugImage)
		{
			ptr = (uchar* ) (scrImage->imageData + p.y * scrImage->widthStep);
			elementPtr = &ptr[p.x*3];
		}
#endif
		solveRegion(p);
		image[p.x][p.y].count = -1;
		if (valid(bwImage[p.x][p.y]))
		{
#if ((Debug_OpenCV == 3) || (Debug_OpenCV == 4))
			if (debugImage)
			{
				elementPtr[0] = 0;
				elementPtr[1] = 0;
				elementPtr[2] = 0;
			}
#endif
			image[p.x][p.y].value = true;
			if (image[p.x][p.y-1].value)
			{
				parent.x = p.x;
				parent.y = p.y-1;
				parent = getLastParent(parent);
				image[p.x][p.y].parent = parent;
				image[parent.x][parent.y].count--;
				image[parent.x][parent.y].cx += (float)p.x;
				image[parent.x][parent.y].cy += (float)p.y;
				if (image[parent.x][parent.y].minX > p.x) image[parent.x][parent.y].minX = p.x;
				if (image[parent.x][parent.y].maxX < p.x) image[parent.x][parent.y].maxX = p.x;
				if (image[parent.x][parent.y].minY > p.y) image[parent.x][parent.y].minY = p.y;
				if (image[parent.x][parent.y].maxY < p.y) image[parent.x][parent.y].maxY = p.y;
			}
			else
			{
				image[p.x][p.y].parent.x = -1;//у него нет родительской метки, т.е. он же и является родителем
				image[p.x][p.y].parent.y = 1;// >= 0 - значит этот родитель еще не обрабатывался на 2ом шаге
				image[p.x][p.y].cx = (float)p.x;
				image[p.x][p.y].cy = (float)p.y;
				image[p.x][p.y].minX = image[p.x][p.y].maxX = p.x;
				image[p.x][p.y].minY = image[p.x][p.y].maxY = p.y;
			}
		}
		else
		{
#if ((Debug_OpenCV == 3) || (Debug_OpenCV == 4))
			if (debugImage)
			{
				elementPtr[0] = 255;
				elementPtr[1] = 255;
				elementPtr[2] = 255;
			}
#endif
			image[p.x][p.y].value = false;
		}
		//Остальная полоса
		for (p.x=1; p.x<size.width; p.x++)
		{
#if ((Debug_OpenCV == 3) || (Debug_OpenCV == 4))
			if (debugImage)
				elementPtr = &ptr[p.x*3];
#endif
			solveRegion(p);
			image[p.x][p.y].count = -1;
			if (valid(bwImage[p.x][p.y]))
			{
#if ((Debug_OpenCV == 3) || (Debug_OpenCV == 4))
				if (debugImage)
				{
					elementPtr[0] = 0;
					elementPtr[1] = 0;
					elementPtr[2] = 0;
				}
#endif
				image[p.x][p.y].value = true;
				if (image[p.x][p.y-1].value)
				{
					if (image[p.x-1][p.y].value)
					{
						parent.x = p.x-1;
						parent.y = p.y;
						parent = getLastParent(parent);
						image[p.x][p.y].parent = parent;
						parent2.x = p.x;
						parent2.y = p.y-1;
						parent2 = getLastParent(parent2);
						if (parent == parent2)
						{
							image[parent.x][parent.y].count--;
							image[parent.x][parent.y].cx += (float)p.x;
							image[parent.x][parent.y].cy += (float)p.y;
						}
						else
						{
							image[parent.x][parent.y].count += image[parent2.x][parent2.y].count-1;
							image[parent.x][parent.y].cx += image[parent2.x][parent2.y].cx + (float)p.x;
							image[parent.x][parent.y].cy += image[parent2.x][parent2.y].cy + (float)p.y;
							image[parent2.x][parent2.y].parent = parent;
							image[parent2.x][parent2.y].count = -1;
							if (image[parent.x][parent.y].minX > image[parent2.x][parent2.y].minX) 
								image[parent.x][parent.y].minX = image[parent2.x][parent2.y].minX;
							if (image[parent.x][parent.y].maxX < image[parent2.x][parent2.y].maxX) 
								image[parent.x][parent.y].maxX = image[parent2.x][parent2.y].maxX;
							if (image[parent.x][parent.y].minY > image[parent2.x][parent2.y].minY) 
								image[parent.x][parent.y].minY = image[parent2.x][parent2.y].minY;
							if (image[parent.x][parent.y].maxY < image[parent2.x][parent2.y].maxY) 
								image[parent.x][parent.y].maxY = image[parent2.x][parent2.y].maxY;

						}
						if (image[parent.x][parent.y].minX > p.x) image[parent.x][parent.y].minX = p.x;
						if (image[parent.x][parent.y].maxX < p.x) image[parent.x][parent.y].maxX = p.x;
						if (image[parent.x][parent.y].minY > p.y) image[parent.x][parent.y].minY = p.y;
						if (image[parent.x][parent.y].maxY < p.y) image[parent.x][parent.y].maxY = p.y;
					}
					else
					{
						parent.x = p.x;
						parent.y = p.y-1;
						parent = getLastParent(parent);
						image[p.x][p.y].parent = parent;
						image[parent.x][parent.y].count--;
						image[parent.x][parent.y].cx += (float)p.x;
						image[parent.x][parent.y].cy += (float)p.y;
						if (image[parent.x][parent.y].minX > p.x) image[parent.x][parent.y].minX = p.x;
						if (image[parent.x][parent.y].maxX < p.x) image[parent.x][parent.y].maxX = p.x;
						if (image[parent.x][parent.y].minY > p.y) image[parent.x][parent.y].minY = p.y;
						if (image[parent.x][parent.y].maxY < p.y) image[parent.x][parent.y].maxY = p.y;
					}
				}
				else if (image[p.x-1][p.y].value)
				{
					parent.x = p.x-1;
					parent.y = p.y;
					parent = getLastParent(parent);
					image[p.x][p.y].parent = parent;
					image[parent.x][parent.y].count--;
					image[parent.x][parent.y].cx += (float)p.x;
					image[parent.x][parent.y].cy += (float)p.y;
					if (image[parent.x][parent.y].minX > p.x) image[parent.x][parent.y].minX = p.x;
					if (image[parent.x][parent.y].maxX < p.x) image[parent.x][parent.y].maxX = p.x;
					if (image[parent.x][parent.y].minY > p.y) image[parent.x][parent.y].minY = p.y;
					if (image[parent.x][parent.y].maxY < p.y) image[parent.x][parent.y].maxY = p.y;
				}
				else
				{
					image[p.x][p.y].parent.x = -1;//у него нет родительской метки, т.е. он же и является родителем
					image[p.x][p.y].parent.y = 1;// >= 0 - значит этот родитель еще не обрабатывался на 2ом шаге
					image[p.x][p.y].cx = (float)p.x;
					image[p.x][p.y].cy = (float)p.y;
					image[p.x][p.y].minX = image[p.x][p.y].maxX = p.x;
					image[p.x][p.y].minY = image[p.x][p.y].maxY = p.y;
				}
			}
			else
			{
#if ((Debug_OpenCV == 3) || (Debug_OpenCV == 4))
				if (debugImage)
				{
					elementPtr[0] = 255;
					elementPtr[1] = 255;
					elementPtr[2] = 255;
				}
#endif
				image[p.x][p.y].value = false;
			}
		}
	}
}

inline void ImageProcessor::step2()
{
#if (Debug_OpenCV == 1)
	uchar* elementPtr;
#endif
	int t;
	float dx, dy;
	countSegment = 0;
	CvSize s;
	CvPoint p, parent;
	//uchar* elementPtr;
	for (p.x=0; p.x<size.width; p.x++)
		for (p.y=0; p.y<size.height; p.y++)
		{
			if (image[p.x][p.y].value)
			{
				if (image[p.x][p.y].parent.x < 0)//пиксель - родитель
				{
					if (image[p.x][p.y].parent.y >= 0)//обрабатывался ли он уже
					{//еще нет
						s.width = (image[p.x][p.y].maxX - image[p.x][p.y].minX);
						s.height = (image[p.x][p.y].maxY - image[p.x][p.y].minY);
						if (((s.width > minSize) || (s.height > minSize)) && ((s.width < maxSize) && (s.height < maxSize)))
						{
							mSegment[countSegment].active = true;
							mSegment[countSegment].minX = image[p.x][p.y].minX;
							mSegment[countSegment].maxX = image[p.x][p.y].maxX;
							mSegment[countSegment].minY = image[p.x][p.y].minY;
							mSegment[countSegment].maxY = image[p.x][p.y].maxY;
							mSegment[countSegment].count = -image[p.x][p.y].count;
							mSegment[countSegment].cx = image[p.x][p.y].cx / (float) mSegment[countSegment].count;
							mSegment[countSegment].cy = image[p.x][p.y].cy / (float) mSegment[countSegment].count;
							dx = p.x - mSegment[countSegment].cx;
							dy = p.y - mSegment[countSegment].cy;
							mSegment[countSegment].ux = dx * dx;
							mSegment[countSegment].uy = dy * dy;
							mSegment[countSegment].uxy = - dx * dy;
							image[p.x][p.y].count = countSegment;
							countSegment++;
#if (Debug_OpenCV == 1)
							if (debugImage)
							{
								elementPtr = (uchar* ) (scrImage->imageData + p.y * scrImage->widthStep + p.x * 3);
								elementPtr[2] = r[image[p.x][p.y].count];
								elementPtr[1] = g[image[p.x][p.y].count];
								elementPtr[0] = b[image[p.x][p.y].count];
							}
#endif
							if (countSegment > maxCountSegment) return;
						}
						else
						{
							image[p.x][p.y].count = -1;
						}
						image[p.x][p.y].parent.y = -1;//пиксель - родитель обработан
					}
					//parent = p;
				}
				else
				{
					parent = getLastParent(image[p.x][p.y].parent);//пиксель - родитель
					if (image[parent.x][parent.y].parent.y >= 0)//обрабатывался ли он уже
					{//еще нет
						s.width = (image[parent.x][parent.y].maxX - image[parent.x][parent.y].minX);
						s.height = (image[parent.x][parent.y].maxY - image[parent.x][parent.y].minY);
						if (((s.width > minSize) || (s.height > minSize)) && ((s.width < maxSize) && (s.height < maxSize)))
						{
							mSegment[countSegment].active = true;
							mSegment[countSegment].minX = image[parent.x][parent.y].minX;
							mSegment[countSegment].maxX = image[parent.x][parent.y].maxX;
							mSegment[countSegment].minY = image[parent.x][parent.y].minY;
							mSegment[countSegment].maxY = image[parent.x][parent.y].maxY;
							mSegment[countSegment].count = -image[parent.x][parent.y].count;
							mSegment[countSegment].cx = image[parent.x][parent.y].cx / (float) mSegment[countSegment].count;
							mSegment[countSegment].cy = image[parent.x][parent.y].cy / (float) mSegment[countSegment].count;
							dx = parent.x - mSegment[countSegment].cx;
							dy = parent.y - mSegment[countSegment].cy;
							mSegment[countSegment].ux = dx * dx;
							mSegment[countSegment].uy = dy * dy;
							mSegment[countSegment].uxy = - dx * dy;
							dx = p.x - mSegment[countSegment].cx;
							dy = p.y - mSegment[countSegment].cy;
							mSegment[countSegment].ux += dx * dx;
							mSegment[countSegment].uy += dy * dy;
							mSegment[countSegment].uxy -= dx * dy;
							image[parent.x][parent.y].count = countSegment;
							image[p.x][p.y].count = countSegment;
							countSegment++;
#if (Debug_OpenCV == 1)
							if (debugImage)
							{
								elementPtr = (uchar* ) (scrImage->imageData + parent.y * scrImage->widthStep + parent.x * 3);
								elementPtr[2] = r[image[parent.x][parent.y].count];
								elementPtr[1] = g[image[parent.x][parent.y].count];
								elementPtr[0] = b[image[parent.x][parent.y].count];
								elementPtr = (uchar* ) (scrImage->imageData + p.y * scrImage->widthStep + p.x * 3);
								elementPtr[2] = r[image[p.x][p.y].count];
								elementPtr[1] = g[image[p.x][p.y].count];
								elementPtr[0] = b[image[p.x][p.y].count];
							}
#endif
							if (countSegment > maxCountSegment) return;
						}
						else
						{
							image[parent.x][parent.y].count = -1;
							image[p.x][p.y].count = -1;
						}
						image[parent.x][parent.y].parent.y = -1;//пиксель - родитель обработан
					}
					else
					{//уже да
						image[p.x][p.y].count = image[parent.x][parent.y].count;
						t = image[p.x][p.y].count;
						if (t >= 0)
						{
#if (Debug_OpenCV == 1)
							if (debugImage)
							{
								elementPtr = (uchar* ) (scrImage->imageData + p.y * scrImage->widthStep + p.x * 3);
								elementPtr[2] = r[image[p.x][p.y].count];
								elementPtr[1] = g[image[p.x][p.y].count];
								elementPtr[0] = b[image[p.x][p.y].count];
							}
#endif
							dx = p.x - mSegment[t].cx;
							dy = p.y - mSegment[t].cy;
							mSegment[t].ux += dx * dx;
							mSegment[t].uy += dy * dy;
							mSegment[t].uxy -= dx * dy;
						}
					}
				}
			}
		}
	bool b;
	float c, amax, amin;
	for (p.x=0; p.x<countSegment; p.x++)
		if (mSegment[p.x].active)
		{
			mSegment[p.x].ux = 1.0f/12.0f + (mSegment[p.x].ux / (float)mSegment[p.x].count);
			mSegment[p.x].uy = 1.0f/12.0f + (mSegment[p.x].uy / (float)mSegment[p.x].count);
			mSegment[p.x].uxy = 1.0f/12.0f + (mSegment[p.x].uxy / (float)mSegment[p.x].count);
			dx = mSegment[p.x].ux - mSegment[p.x].uy;
			c = sqrtf(dx * dx + 4.0f * mSegment[p.x].uxy * mSegment[p.x].uxy);
			amax = mSegment[p.x].ux + mSegment[p.x].uy + c;
			amin = mSegment[p.x].ux + mSegment[p.x].uy - c;
			mSegment[p.x].e = sqrtf((amax - amin) / amax);
			//сравнение с сущ-ими образами
			/*b = true;
			for (p.y=0; p.y<countFindedObject; p.y++)
				if (fabs(e-findedObject[p.y].e) < epsEccentricity)
				{
					b = false;
					break;
				}
			if (b)
			{
				mSegment[p.x].count = -1;
				continue;
			}*/
			
			if (mSegment[p.x].uy > mSegment[p.x].ux)
				mSegment[p.x].ux = atanf((mSegment[p.x].uy - mSegment[p.x].ux + c) / (2.0f * mSegment[p.x].uxy));
			else
				mSegment[p.x].ux = atanf((2.0f * mSegment[p.x].uxy) / (mSegment[p.x].ux - mSegment[p.x].uy + c));
			clearMapSegment(p.x);
			mSegment[p.x].angle = mSegment[p.x].ux;
			mSegment[p.x].fObject = -1;
			mSegment[p.x].probability = -1.0f;
			mSegment[p.x].localMaxX = mSegment[p.x].localMaxY = MINNUMBER;
			mSegment[p.x].localMinX = mSegment[p.x].localMinY = MAXNUMBER;
		}
}

inline void ImageProcessor::step3()
{
#if (Debug_OpenCV == 2)
	uchar* elementPtr;
#endif
	int t;
	float dx, dy, sina, cosa;
	CvPoint p;
	for (p.x=0; p.x<size.width; p.x++)
		for (p.y=0; p.y<size.height; p.y++)
		{
			t = image[p.x][p.y].count;
			if (t >= 0)
			{
				if (mSegment[t].active)
				{
					dx = p.x - mSegment[t].cx;
					dy = p.y - mSegment[t].cy;
					sina = sinf(mSegment[t].ux);
					cosa = cosf(mSegment[t].ux);
					image[p.x][p.y].cx = dx * cosa - dy * sina;
					image[p.x][p.y].cy = dx * sina + dy * cosa;
#if (Debug_OpenCV == 2)
					if (debugImage)
					{
						CvPoint p1;
						p1.x = (int)floorf(image[p.x][p.y].cx + mSegment[t].cx);
						p1.y = (int)floorf(image[p.x][p.y].cy + mSegment[t].cy);
						if ((p1.x >= 0) && (p1.x < size.width) && (p1.y >= 0) && (p1.y < size.height))
						{
							elementPtr = (uchar* ) (scrImage->imageData + p1.y * scrImage->widthStep + p1.x * 3);
							elementPtr[2] = r[t];
							elementPtr[1] = g[t];
							elementPtr[0] = b[t];
						}
					}
#endif
					if (image[p.x][p.y].cx < mSegment[t].localMinX) mSegment[t].localMinX = image[p.x][p.y].cx;
					if (image[p.x][p.y].cy < mSegment[t].localMinY) mSegment[t].localMinY = image[p.x][p.y].cy;
					if (image[p.x][p.y].cx > mSegment[t].localMaxX) mSegment[t].localMaxX = image[p.x][p.y].cx;
					if (image[p.x][p.y].cy > mSegment[t].localMaxY) mSegment[t].localMaxY = image[p.x][p.y].cy;
				}
				else
					image[p.x][p.y].count = -1;
			}
		}
	float scale;
	for (p.x=0; p.x<countSegment; p.x++)
		if (mSegment[p.x].active)
		{
			dx = max((- mSegment[p.x].localMinX), mSegment[p.x].localMaxX);
			dy = max((- mSegment[p.x].localMinY), mSegment[p.x].localMaxY);
			mSegment[p.x].scale = mapSize / (dx * 2.0f);
			scale = mapSize / (dy * 2.0f);
			if (scale < mSegment[p.x].scale) mSegment[p.x].scale = scale; 
		}
}

inline void ImageProcessor::step4()
{
	float x, y, dx, dy, s, halfMapSize = mapSize * 0.5f;
	int mS = mapSize - 1;
	int tSegment, begin;
	CvPoint p, newP, newP2;
	for (p.x=0; p.x<size.width; p.x++)
		for (p.y=0; p.y<size.height; p.y++)
		{
			tSegment = image[p.x][p.y].count;
			if (tSegment >= 0)
			{
				x = (image[p.x][p.y].cx) * mSegment[tSegment].scale + halfMapSize;
				y = (image[p.x][p.y].cy) * mSegment[tSegment].scale + halfMapSize;
				newP.x = round(x);
				newP.y = round(y);
				dx = x - newP.x;
				if (newP.x == mapSize)
				{
					newP.x--;
					newP2.x = newP.x;
					s = mSegment[tSegment].scale * halfScalePoint - dx;
					if (s > 0.0f) newP.x -= (int)ceilf(s);
					if (newP.x < 0 ) newP.x = 0;
				}
				else
				{
					s = mSegment[tSegment].scale * halfScalePoint + dx - 1.0f;
					if (s > 0.0f)
					{
						newP2.x = newP.x + (int)(ceilf(s));
						if (newP2.x > mS) newP2.x = mS;
					}
					else newP2.x = newP.x;
					s = mSegment[tSegment].scale * halfScalePoint - dx;
					if (s > 0.0f) newP.x -= (int)ceilf(s);
					if (newP.x < 0 ) newP.x = 0;
				}
				dy = y - newP.y;
				if (newP.y == mapSize)
				{
					newP.y--;
					newP2.y = newP.y;
					s = mSegment[tSegment].scale * halfScalePoint - dy;
					if (s > 0.0f) newP.y -= (int)ceilf(s);
					if (newP.y < 0 ) newP.y = 0;
				}
				else
				{
					s = mSegment[tSegment].scale * halfScalePoint + dy - 1.0f;
					if (s > 0.0f)
					{
						newP2.y = newP.y + (int)(ceilf(s));
						if (newP2.y > mS) newP2.y = mS;
					}
					else newP2.y = newP.y;
					s = mSegment[tSegment].scale * halfScalePoint - dy;
					if (s > 0.0f) newP.y -= (int)ceilf(s);
					if (newP.y < 0 ) newP.y = 0;
				}
				/*if ((newP.x < 0) || (newP2.x >= mapSize) || (newP.y < 0) || (newP2.y >= mapSize))
				{
					continue;
				}*/
				begin = newP.y;
				s = mSegment[tSegment].scale * mSegment[tSegment].scale * halfScalePoint * 2.8284271f;
				for (; newP.x<=newP2.x; newP.x++)
				{
					dx = newP.x - x;
					dx *= dx;
					for (newP.y=begin; newP.y<=newP2.y; newP.y++)
					{
						dy = newP.y - y;
						dy *= dy;
						if ((dx + dy) <= s)
						{
							mSegment[tSegment].map[newP.x][newP.y] = true;
						}
					}
				}
			}
		}
#if (Debug_OpenCV_Map == 1)
	if (debugMap)
	{
		uchar* ptr;
		newP.x = 0;
		newP.y = 0;
		for (tSegment=0; tSegment<countSegment; tSegment++)
		{
			for (p.x=0; p.x<mapSize; p.x++)
				for (p.y=0; p.y<mapSize; p.y++)
				{
					if (mSegment[tSegment].map[p.x][p.y])
					{
						ptr = (uchar*)(map->imageData + map->widthStep * (p.y + newP.y) + (p.x + newP.x));
						ptr[0] = 0;
						ptr[1] = 0;
						ptr = (uchar*)(map->imageData + map->widthStep * (p.y + 1+ newP.y) + (p.x + newP.x));
						ptr[0] = 0;
						ptr[1] = 0;
					}
					else
					{
						ptr = (uchar*)(map->imageData + map->widthStep * (p.y + newP.y) + (p.x + newP.x));
						ptr[0] = 255;
						ptr[1] = 255;
						ptr = (uchar*)(map->imageData + map->widthStep * (p.y + 1 + newP.y) + (p.x + newP.x));
						ptr[0] = 255;
						ptr[1] = 255;
					}
				}
			newP.x += mapSize;
			if (newP.x > mapSize * maxCountSizeMap)
			{
				newP.x = 0;
				newP.y += mapSize;
				if (newP.y > mapSize * maxCountSizeMap) break;
			}
		}
		cvShowImage("Founded Object", map);
	}
#endif
}

inline void ImageProcessor::startTraningFindedObject(int nFindedObject)
{
	if (findedObject[nFindedObject].traning)
		free(findedObject[nFindedObject].traning);
	findedObject[nFindedObject].traning = new Traning();
	int i, j;
	for (i=0; i<mapSize; i++)
	{
		for (j=0; j<mapSize; j++)
		{
			if (findedObject[nFindedObject].map[i][j])
				findedObject[nFindedObject].traning->map[i][j] = 1.0f;
			else
				findedObject[nFindedObject].traning->map[i][j] = 0.0f;
		}
	}
	findedObject[nFindedObject].traning->iter = traningTime;
	printf("Начало обучение символу - %s.\n", findedObject[nFindedObject].name);
}

inline void ImageProcessor::addFindedObject(int nSegment, char name[50])
{
	int i, j;
	findedObject = (FindedObject*)realloc(findedObject, sizeof(FindedObject) * (countFindedObject+1));
	findedObject[countFindedObject].e = mSegment[nSegment].e;
	for (i=0; i<50; i++)
		findedObject[countFindedObject].name[i] = name[i];
	findedObject[countFindedObject].traning = NULL;
	findedObject[countFindedObject].countFillPoint = 0;
	for (i=0; i<mapSize; i++)
		for (j=0; j<mapSize; j++)
		{
			findedObject[countFindedObject].map[i][j] = mSegment[nSegment].map[i][j];
			if (findedObject[countFindedObject].map[i][j])
				findedObject[countFindedObject].countFillPoint++;
		}
	startTraningFindedObject(countFindedObject);
	countFindedObject++;
}

void ImageProcessor::openData()
{
	FILE* f;
	f = fopen("data.bin", "rb");
	if (f != NULL)
	{
		fread(&countFindedObject, sizeof(float), 1, f);
		printf("Открыто образов - %d.\n", countFindedObject);
		findedObject = (FindedObject*)malloc(sizeof(FindedObject) * countFindedObject);
		int i, k;
		for (k=0; k<countFindedObject; k++)
		{
			fread(&findedObject[k].e, sizeof(float), 1, f);
			fread(&findedObject[k].countFillPoint, sizeof(int), 1, f);
			fread(&findedObject[k].name, sizeof(char), 50, f);
			for (i=0; i<mapSize; i++)
				fread(&findedObject[k].map[i], sizeof(bool), mapSize, f);
			findedObject[k].traning = NULL;
			printf("%d - %s\n", (k+1), findedObject[k].name);
		}
	}
	else 
		printf("Не удалось открыть образы.\n");
}

void ImageProcessor::saveData()
{
	FILE* f;
	f = fopen("data.bin", "wb");
	if (f != NULL)
	{
		printf("Образы сохранены.\n");
		fwrite(&countFindedObject, sizeof(float), 1, f);
		int i, k;
		for (k=0; k<countFindedObject; k++)
		{
			fwrite(&findedObject[k].e, sizeof(float), 1, f);
			fwrite(&findedObject[k].countFillPoint, sizeof(int), 1, f);
			fwrite(&findedObject[k].name, sizeof(char), 50, f);
			for (i=0; i<mapSize; i++)
				fwrite(&findedObject[k].map[i], sizeof(bool), mapSize, f);
		}
	}
	else
		printf("Не удалось сохранить образы.\n");
}

void ImageProcessor::addFindedObject(CvPoint p, char name[50])
{
	if (image[p.x][p.y].value)
	{
		int t = image[p.x][p.y].count;
		if (t >= 0)
		{
			if ((mSegment[t].active) && (mSegment[t].fObject >= 0))
			{
				//MessageBox(NULL, _T("Образ обновляется."), _T("Обновление"), MB_OK);
				printf("Образ уже существует - обновление...\n");
				startTraningFindedObject(mSegment[t].fObject);
			}
			else
			{
				printf("Введите имя образа - ");
				scanf("%s", name);
				printf("Образ добавлен.\n");
				//MessageBox(NULL, _T("Образ добавлен."),_T("Успех"),  MB_OK);
				addFindedObject(t, name);
			}
			return;
		}
	}
	//MessageBox(NULL, _T("Образ не найден."), _T("Ошибка"), MB_OK);
	printf("В указанной точке нет образа.\n");
}

inline float ImageProcessor::compareSegment(int nSegment, int nFindedObject)
{
	int i, j, probability = 0;
	for (i=0; i<mapSize; i++)
		for (j=0; j<mapSize; j++)
		{
			if ((findedObject[nFindedObject].map[i][j] == mSegment[nSegment].map[i][j]))
				probability++;
		}
	return (probability) / ((float)(mapSize*mapSize));
}

inline float ImageProcessor::compareSegmentInverse(int nSegment, int nFindedObject)
{
	int i, j, invi, invj, mS = mapSize-1, probability = 0;
	for (i=0; i<mapSize; i++)
	{
		invi = mS - i;
		for (j=0; j<mapSize; j++)
		{
			invj = mS - j;
			if ((findedObject[nFindedObject].map[i][j] == mSegment[nSegment].map[invi][invj]))
				probability++;
		}
	}
	return (probability) / ((float)(mapSize*mapSize));
}

inline void ImageProcessor::compareAllObjects()
{
	int i, j, k;
	float tempP;
	for (i=0; i<countSegment; i++)
	{
		if (mSegment[i].active)
		{
			mSegment[i].active = false;
			for (j=0; j<countFindedObject; j++)
			{
				if (fabs(mSegment[i].e - findedObject[j].e) < epsEccentricity)
				{
					tempP = compareSegment(i, j);
					if (tempP >= probabilityA)
					{
						mSegment[i].fObject = j;
						mSegment[i].probability = tempP;
						mSegment[i].inverse = false;
						mSegment[i].active = true;
						break;
					}
					else
					{
						if (tempP >= probabilityB)
						{
							if (tempP > mSegment[i].probability)
							{
								mSegment[i].fObject = j;
								mSegment[i].probability = tempP;
								mSegment[i].inverse = false;
								mSegment[i].active = true;
							}
						}
						else if (tempP > probabilityC)
						{
							if (!mSegment[i].active)
							{
								for (k=0; k<countPrevResult; k++)
								{
									if (!mPrevResult[k].active)
									{
										if (mPrevResult[k].fObject == j)
										{
											if ((fabs(mPrevResult[k].maxX - mSegment[i].maxX) < epsPrevResult) && 
												(fabs(mPrevResult[k].maxY - mSegment[i].maxY) < epsPrevResult) && 
												(fabs(mPrevResult[k].minX - mSegment[i].minX) < epsPrevResult) && 
												(fabs(mPrevResult[k].minY - mSegment[i].minY) < epsPrevResult))
											{
												//mPrevResult[j].active = true;
												mSegment[i].fObject = j;
												mSegment[i].probability = tempP;
												mSegment[i].inverse = false;
												mSegment[i].active = true;
											}
										}
									}
								}
							}
						}
					}
					tempP = compareSegmentInverse(i, j);
					if (tempP >= probabilityA)
					{
						mSegment[i].fObject = j;
						mSegment[i].probability = tempP;
						mSegment[i].inverse = true;
						mSegment[i].active = true;
						break;
					}
					else
					{
						if (tempP >= probabilityB)
						{
							if (tempP > mSegment[i].probability)
							{
								mSegment[i].fObject = j;
								mSegment[i].probability = tempP;
								mSegment[i].inverse = true;
								mSegment[i].active = true;
							}
						}
						else if (tempP > probabilityC)
						{
							if (!mSegment[i].active)
							{
								for (k=0; k<countPrevResult; k++)
								{
									if (!mPrevResult[k].active)
									{
										if (mPrevResult[k].fObject == j)
										{
											if ((fabs(mPrevResult[k].maxX - mSegment[i].maxX) < epsPrevResult) && 
												(fabs(mPrevResult[k].maxY - mSegment[i].maxY) < epsPrevResult) && 
												(fabs(mPrevResult[k].minX - mSegment[i].minX) < epsPrevResult) && 
												(fabs(mPrevResult[k].minY - mSegment[i].minY) < epsPrevResult))
											{
												//mPrevResult[j].active = true;
												mSegment[i].fObject = j;
												mSegment[i].probability = tempP;
												mSegment[i].inverse = true;
												mSegment[i].active = true;
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
#if (Debug_OpenCV_FindedObject == 1)
	if (debugFindedMap)
	{
		uchar* elementPtr;
		int maxC = countFindedObject, k;
		if (maxC > maxCountFindedObject) maxC = maxCountFindedObject;
		cvRectangle(findedMap, cvPoint(0, mapSize), cvPoint(mapSize * maxCountFindedObject, mapSize + 20), cvScalar(255), -1);
		for (k=0; k<maxC; k++)
		{
			for (i=0; i<mapSize; i++)
			{
				for (j=0; j<mapSize; j++)
				{
					elementPtr = (uchar*)(findedMap->imageData + findedMap->widthStep * (j) + (mapSize * k + i));
					if (findedObject[k].map[i][j])
						elementPtr[0] = 0;
					else
						elementPtr[0] = 255;
				}
			}
			cvPutText(findedMap, findedObject[k].name, cvPoint(k * mapSize, mapSize + 14), &font, cvScalar(0));
		}
		cvShowImage("Finded Object", findedMap);
	}
#endif
}

inline void ImageProcessor::deleteFindedObject(int i)
{
	countFindedObject--;
	findedObject[i] = findedObject[countFindedObject];
}

void ImageProcessor::deleteFindedObject(char* s)
{
	int i;
	for (i=0; i<countFindedObject; i++)
	{
		if (strcmp(findedObject[i].name, s) == 0)
		{
			deleteFindedObject(i);
			i--;
		}
	}
}

inline void ImageProcessor::updateTraning()
{
	int nSegment=0, nFindedObject, i, j, invi, mS = mapSize-1;
	for (; nSegment<countSegment; nSegment++)
	{
		if (mSegment[nSegment].probability >= probabilityTraning)
		{
			nFindedObject = mSegment[nSegment].fObject;
			if (nFindedObject >= 0)
			{
				if (findedObject[nFindedObject].traning != NULL)
				{
					findedObject[nFindedObject].e = findedObject[nFindedObject].e * (1.0f - speedTraning) + mSegment[nSegment].e * speedTraning;
					if (mSegment[nSegment].inverse)
					{
						for (i=0; i<mapSize; i++)
						{
							invi = mS - i;
							for (j=0; j<mapSize; j++)
							{
								if (mSegment[nSegment].map[invi][mS-j])
								{
									findedObject[nFindedObject].traning->map[i][j] += speedTraning;
									if (findedObject[nFindedObject].traning->map[i][j] > 1.0f)
									{
										findedObject[nFindedObject].traning->map[i][j] = 1.0f;
									}
									if (findedObject[nFindedObject].traning->map[i][j] > limitTraning)
										findedObject[nFindedObject].map[i][j] = true;
								}
								else
								{
									findedObject[nFindedObject].traning->map[i][j] -= speedTraning;
									if (findedObject[nFindedObject].traning->map[i][j] < 0.0f)
									{
										findedObject[nFindedObject].traning->map[i][j] = 0.0f;
									}
									if (findedObject[nFindedObject].traning->map[i][j] < limitTraning)
										findedObject[nFindedObject].map[i][j] = false;
								}
							}
						}
					}
					else
					{
						for (i=0; i<mapSize; i++)
						{
							for (j=0; j<mapSize; j++)
							{
								if (mSegment[nSegment].map[i][j])
								{
									findedObject[nFindedObject].traning->map[i][j] += speedTraning;
									if (findedObject[nFindedObject].traning->map[i][j] > 1.0f)
									{
										findedObject[nFindedObject].traning->map[i][j] = 1.0f;
									}
									if (findedObject[nFindedObject].traning->map[i][j] > limitTraning)
										findedObject[nFindedObject].map[i][j] = true;
								}
								else
								{
									findedObject[nFindedObject].traning->map[i][j] -= speedTraning;
									if (findedObject[nFindedObject].traning->map[i][j] < 0.0f)
									{
										findedObject[nFindedObject].traning->map[i][j] = 0.0f;
									}
									if (findedObject[nFindedObject].traning->map[i][j] < limitTraning)
										findedObject[nFindedObject].map[i][j] = false;
								}
							}
						}
					}
					--findedObject[nFindedObject].traning->iter;
					if (findedObject[nFindedObject].traning->iter < 0)
					{
						free(findedObject[nFindedObject].traning);
						findedObject[nFindedObject].traning = NULL;
						printf("Обучение символу завершено - %s.\n", findedObject[nFindedObject].name);
					}
				}
			}
		}
	}
}

void ImageProcessor::process(IplImage* scr)
{
	scrImage = scr;
	initPrevResult();//Текущие результаты становятся предъидущими
	step0();//делает ч/б изображение, вычисляет интегральное изображение
	step1();//бинаризация по отсу, сегментация, начало вычисления центра сегментов, расчет ААВВ
	step2();//конец вычисления центра сегментов, начало вычесление поворота сегмента, определение сегментов, 
	//заранее отсекая ненужные по AABB (слишком большие или маленькие)
	step3();//конец вычисление поворота сегмента, у необходимых точек есть теперь свои новые локальные координаты, 
	//локальный ААВВ, тут же должны бы отбрасываться некоторые сегменты по эксцентриситету момента инерции о_0 (если нет сегментов с таким эксцентриситетом в базе)
	step4();//запись необходимых точек в карты объектов
	compareAllObjects();//сравнение найденных объектов с теми что ищем
	updateTraning();//обновить обучение образам
}

void ImageProcessor::drawInfo()
{
	CvPoint p1, p2;
	int i;
	for (i=0; i<countSegment; i++)
	{
		if (mSegment[i].active)
		{
			if (mSegment[i].probability >= probabilityB)
				cvRectangle(scrImage, cvPoint(mSegment[i].minX, mSegment[i].minY), cvPoint(mSegment[i].maxX, mSegment[i].maxY), 
					CV_RGB(255, 0, 0), 1);
			else
				cvRectangle(scrImage, cvPoint(mSegment[i].minX, mSegment[i].minY), cvPoint(mSegment[i].maxX, mSegment[i].maxY), 
					CV_RGB(0, 0, 255), 1);
			p1.x = (mSegment[i].minX+mSegment[i].maxX)/2;
			p1.y = (mSegment[i].minY+mSegment[i].maxY)/2;
			p2.x = p1.x + 15;
			p2.y = p1.y + 15;
			cvLine(scrImage, p1, p2, CV_RGB(255,0,0));
			p2.y+=10;
			cvPutText(scrImage, findedObject[mSegment[i].fObject].name, p2, &fontText, CV_RGB(0,0,0));
		}
	}
}