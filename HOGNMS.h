#ifndef __HOG_NMS__
#define __HOG_NMS__

#include <iostream>
#include "cvaux.h"
#include "cv.h"
#include <ios>
#include "highgui.h"
#include <vector>
#include <fstream>
#include <ctype.h>
#include <dirent.h>
#include <stdexcept>
#include <ctype.h>

using namespace std;
using namespace cv;

#include "HOG_Result.h"

class HOGNMS
{
	private:
		Point3f  *at, *ms, *tomode, *nmsToMode;

		HOG_Result *nmsResultsLocal;

		float* wt;

		float center, scale;
		float nonmaxSigma[3];
		float nsigma[3];
		float modeEpsilon;
		float epsFinalDist;

		int maxIterations;

		bool isAllocated;

		float sigmoid(float score) { return (score > center) ? scale * (score - center) : 0.0f; }
		void nvalue(Point3f* ms, Point3f* at, float* wt, int length);
		void nvalue(Point3f* ms, Point3f* msnext, Point3f* at, float* wt, int length);
		void fvalue(Point3f* modes, HOG_Result* results, int lengthModes, Point3f* at, float* wt, int length);
		void shiftToMode(Point3f* ms, Point3f* at, float* wt, Point3f *tomode, int length);
		float distqt(Point3f *p1, Point3f *p2);

	public:
		vector<HOG_Result> ComputeNMSResults(Size my_window,vector<HOG_Result> meus_results);

		HOGNMS();
		~HOGNMS(void);
};

#endif

