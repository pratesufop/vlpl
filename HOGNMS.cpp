#include "HOGNMS.h"

HOGNMS::HOGNMS()
{
	center = 0.0f;
	scale = 1.0f;
	//imagino que estes sigmas sejam o que chama no artigo da XIN LI de window bandwidth, pesquisar mais...
	nonmaxSigma[0] = 25.0f;
	nonmaxSigma[1] = 25.0f;
	nonmaxSigma[2] = 4.0f;

	maxIterations = 100;
	modeEpsilon = (float)1e-5;
	epsFinalDist = 1.0f;

	nsigma[0] = nonmaxSigma[0];
	nsigma[1] = nonmaxSigma[1];
	nsigma[2] = logf(nonmaxSigma[2]);

	isAllocated = false;
}

HOGNMS::~HOGNMS()
{
	if (isAllocated)
	{
		delete tomode;
		delete wt;
		delete ms;
		delete at;
		delete nmsResultsLocal;
		delete nmsToMode;
	}
}

void HOGNMS::nvalue(Point3f* ms, Point3f* at, float* wt, int length)
{
	int i, j;
	float dotxmr, w;
	Point3f x, r, ns, numer, denum;

	for (i=0; i<length; i++)
	{
		//inicializando numerador e denominador...
		numer.x = 0; numer.y = 0; numer.z = 0;
		denum.x = 0; denum.y = 0; denum.z = 0;

        //para cada uma das regiões detectadas pelo classificador...
		for (j=0; j<length; j++)
		{
			ns.x = nsigma[0] * expf(at[j].z);// exp(si)*sigmax
			ns.y =  nsigma[1] * expf(at[j].z);//exp(si)*sigmay
			ns.z = nsigma[2]; //apenas o sigma s.

			x.x = at[j].x / ns.x;
			x.y = at[j].y / ns.y;
			x.z = at[j].z / ns.z;

			r.x = at[i].x / ns.x;
			r.y = at[i].y / ns.y;
			r.z = at[i].z / ns.z;


			dotxmr = (x.x - r.x) * (x.x - r.x) + (x.y - r.y) * (x.y - r.y) + (x.z - r.z) * (x.z - r.z); // (x-xi)' * (x-xi) * inv(H) = D^2

			w = wt[j] * expf(-dotxmr/2.0f)/sqrtf(ns.x * ns.y * ns.z); // w= wi* exp( - (D^2)/2)*diag(H)^(-1/2)

			numer.x += w * x.x;
			numer.y += w * x.y;
			numer.z += w * x.z;
			denum.x += w / ns.x;
			denum.y += w / ns.y;
			denum.z += w / ns.z;
		}

		ms[i].x = numer.x / denum.x;
		ms[i].y = numer.y / denum.y;
		ms[i].z = numer.z / denum.z;
	}
}

void HOGNMS::nvalue(Point3f *ms, Point3f* msnext, Point3f* at, float* wt, int length)
{
	int j;
	float dotxmr, w;
	Point3f x, r, ns, numer, denum, toReturn;

	for (j=0; j<length; j++)
	{
		ns.x = nsigma[0] * expf(at[j].z);
		ns.y =  nsigma[1] * expf(at[j].z);
		ns.z = nsigma[2];

		x.x = at[j].x / ns.x;
		x.y = at[j].y / ns.y;
		x.z = at[j].z / ns.z;
		//observa que nesta parte o segundo ponto passa a ser o ms que já havia sido calculado...
		r.x = ms->x / ns.x;
		r.y = ms->y / ns.y;
		r.z = ms->z / ns.z;

		dotxmr = (x.x - r.x) * (x.x - r.x) + (x.y - r.y) * (x.y - r.y) + (x.z - r.z) * (x.z - r.z);

		w = wt[j] * expf(-dotxmr/2.0f)/sqrtf(ns.x * ns.y * ns.z);

		numer.x += w * x.x; numer.y += w * x.y; numer.z += w * x.z;
		denum.x += w / ns.x; denum.y += w / ns.y; denum.z += w / ns.z;
	}

	msnext->x = numer.x / denum.x; msnext->y = numer.y / denum.y; msnext->z = numer.z / denum.z;
}

void HOGNMS::fvalue(Point3f* modes, HOG_Result* results, int lengthModes, Point3f* at, float* wt, int length)
{
	int i, j;
	float no, dotxx;
	Point3f x, ns;
	for (i=0; i<lengthModes; i++)
	{
		no = 0;
		for (j=0; j<length; j++)
		{
			ns.x = nsigma[0] * expf(at[j].z);
			ns.y =  nsigma[1] * expf(at[j].z);
			ns.z = nsigma[2];
			x.x = (at[j].x - modes[i].x) / ns.x;
			x.y = (at[j].y - modes[i].y) / ns.y;
			x.z = (at[j].z - modes[i].z) / ns.z;

			dotxx = x.x * x.x + x.y * x.y + x.z * x.z;

			no += wt[j] * expf(-dotxx/2)/sqrtf(ns.x * ns.y * ns.z);
		}
		results[i].score = no;
	}
}

float HOGNMS::distqt(Point3f *p1, Point3f *p2)
{
	Point3f ns, b;

	ns.x = nsigma[0] * expf(p2->z);
	ns.y = nsigma[1] * expf(p2->z);
	ns.z = nsigma[2];
	b.x = p2->x - p1->x;
	b.y = p2->y - p1->y;
    b.z = p2->z - p1->z;

	b.x /= ns.x;
	b.y /= ns.y;
	b.z /= ns.z;

	return b.x * b.x + b.y * b.y + b.z * b.z;
}

void HOGNMS::shiftToMode(Point3f* ms, Point3f* at, float* wt, Point3f *tomode, int length)
{
	int i, count;
	Point3f ii,II;
	//para cada região detectada
	for (i=0; i<length; i++)
	{
		II = ms[i];
		count = 0;

		do
		{
			ii = II;
			nvalue(&ii, &II, at, wt, length);
			++count;
		} while ( count < maxIterations && distqt(&ii,&II) > modeEpsilon );
		//nesta região é a parte em que é alterado o centro de massa em busca das regiões de maior densidade??

		tomode[i].x = II.x;
		tomode[i].y = II.y;
		tomode[i].z = II.z;
		//mode é uma região de maior densidade, mas para cada região detectada ele tem um mode??
	}
}
//recebe um vector de results.
vector<HOG_Result> HOGNMS::ComputeNMSResults(Size my_window,vector<HOG_Result> meus_results)
{
	int formattedResultsCount = meus_results.size();//Quantas regiões foram detectadas até então...
	int hWindowSizeX = my_window.width;
	int hWindowSizeY = my_window.height;
    vector<HOG_Result> result_final;
	if (!isAllocated)
	{
		wt = new float[hWindowSizeX * hWindowSizeX];
		at = new Point3f[hWindowSizeX * hWindowSizeX];//centro da região retangulo x+w/2 , y+h/2
		ms = new Point3f[hWindowSizeX * hWindowSizeX];
		tomode = new Point3f[hWindowSizeX * hWindowSizeX];
		nmsToMode = new Point3f[hWindowSizeX * hWindowSizeX];
		nmsResultsLocal = new HOG_Result[hWindowSizeX * hWindowSizeX];
		isAllocated = true;
	}

	int i, j;
	float cenx, ceny, nmsOK;

	int nmsResultsCount = 0;

	for (i=0; i<formattedResultsCount; i++)
	{
		wt[i] = this->sigmoid(meus_results.at(i).score);

		cenx = meus_results.at(i).roi.x + meus_results.at(i).roi.width / 2.0f;
		ceny = meus_results.at(i).roi.y + meus_results.at(i).roi.height / 2.0f;

		at[i] = Point3f(cenx, ceny, logf(meus_results.at(i).scale));
	}

	nvalue(ms, at, wt, formattedResultsCount);
	//ms passa a ter o resultado desejado...
	shiftToMode(ms, at, wt, tomode, formattedResultsCount);

	for (i=0; i<formattedResultsCount; i++)
	{
		nmsOK = true;
		for (j=0; j<nmsResultsCount; j++)
		{
			if (distqt(&nmsToMode[j], &tomode[i]) < epsFinalDist)
			{
				nmsOK = false;
				break;
			}
		}

		if (nmsOK)
		{
			nmsResultsLocal[nmsResultsCount].scale = expf(tomode[i].z);
			nmsResultsLocal[nmsResultsCount].roi.width = (int)floorf((float)hWindowSizeX / nmsResultsLocal[nmsResultsCount].scale);

			nmsResultsLocal[nmsResultsCount].roi.height =(int)floorf((float)hWindowSizeY / nmsResultsLocal[nmsResultsCount].scale);

            cout<<nmsResultsLocal[nmsResultsCount].roi.width<< " x "<<nmsResultsLocal[nmsResultsCount].roi.height <<endl;

			nmsResultsLocal[nmsResultsCount].roi.x =(int)ceilf(tomode[i].x - (float) hWindowSizeX * nmsResultsLocal[nmsResultsCount].scale / 2);
			nmsResultsLocal[nmsResultsCount].roi.y =(int)ceilf(tomode[i].y - (float) hWindowSizeY * nmsResultsLocal[nmsResultsCount].scale / 2);

			nmsToMode[nmsResultsCount] = tomode[i];

			result_final.push_back(nmsResultsLocal[nmsResultsCount]);
			nmsResultsCount++;
		}
	}

	fvalue(nmsToMode, nmsResultsLocal, nmsResultsCount, at, wt, formattedResultsCount);
	return result_final;
}
