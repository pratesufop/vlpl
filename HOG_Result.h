#ifndef __HOG_RESUL__
#define __HOG_RESUL__

#include "cv.h"
using namespace cv;

class HOG_Result
{
	public:
		float score;
		float scale;
		Rect roi;
		int label[4];
		bool sup;

		HOG_Result()
		{
			roi.width = 0;
			roi.height = 0;
			roi.x = 0;
			roi.y = 0;
		}
};


#endif

