#ifndef IMAGE_CLASS_H__
#define IMAGE_CLASS_H__
#include "stdlib.h"
#include "math.h"

#include <iostream>
#include <fstream>

#define MIN(i,j)  ((i)<(j)?(i):(j))
#define MAX(i,j)  ((i)>(j)?(i):(j))
#define SWAP(a,b) int _tempr;_tempr=(a);(a)=(b);(b)=_tempr

enum FILE_FORMAT {PGM_RAW,PGM_ASCII,PPM_RAW,PPM_ASCII}; //P2,P5,P3,P6

struct RGB {
	float r; /* RED   Note: r contains the intensity value when MImage is a gray-scale image */
	float g; /* GREEN Note: 0 for a gray-scale image */
	float b; /* BLUE  Note: 0 for a gray-scale image */
};

typedef struct RGB RGBPixel;

class MImage{

	private:
		int MXS;						 /* number of columns */
		int MYS;						 /* number of rows */
		int MZS;						 /* number of channels (1 for grayscale, 3 for color) */
		
		RGBPixel **MImgBuf;  /* buffer containing the image */

	public:
	
		/* Constructors/destructor */
		MImage(void); /* default constructor */
		MImage(int xs,int ys,int zs); 
		MImage(int xs, int ys, int zs, int color);
		MImage(const MImage &copy); /* copy constructor */ 
		~MImage(void);

		/* Various functions */
		int MXSize(void) const{return MXS;};
		int MYSize(void) const{return MYS;};
		int MZSize(void) const{return MZS;};
		void MSetColor(float val, int x, int y) {MImgBuf[x][y].r=val; if(MZS>1){MImgBuf[x][y].g=val;MImgBuf[x][y].b=val;}};
		void MSetColor(float val, int x, int y, int z) {if(z==0) MImgBuf[x][y].r=val; else if(z==1) MImgBuf[x][y].g=val; else MImgBuf[x][y].b=val;};
		float MGetColor(int x, int y) const {return MImgBuf[x][y].r;};
		float MGetColor(int x, int y, int z) const {if(z==0) return MImgBuf[x][y].r; if(z==1) return MImgBuf[x][y].g; return MImgBuf[x][y].b;};
		
		bool MIsEmpty(void) const{if(MImgBuf==NULL || MXS<=0 || MYS<=0 || MZS<=0) return true; return false;};
		bool MSameSize(const MImage &c){if(MXS==c.MXS && MYS==c.MYS && MZS==c.MZS) return true; return false;};
	
		/* I/O */
		void MAllocMemory(int xs,int ys,int zs);
		void MFreeMemory(void);
		
		bool MLoadImage(const char *imageName);
		bool MSaveImage(const char *imageName, FILE_FORMAT format);

		/* Point operations */
		void MThreshold(float value);
		void MRescale(void);
		
		/* Spatial filters */
		void MMeanShift(float SpatialBandWidth, float RangeBandWidth, float tolerance);
				
		/* Segmentation */
		void MMagicWand(int xSeed, int ySeed, float tolerance);
		void Flood(MImage &yOut, int xSeed, int ySeed, float tolerance, RGBPixel &ref);
		void MOptimalThresholding(float *means,float *stddev,float *apriori,int nbClasses);
		void MKMeansSegmentation(float *means,float *stddev,float *apriori,int nbClasses);
		void MSoftKMeansSegmentation(float *means,float *stddev,float *apriori,float beta,int nbClasses);
		void MExpectationMaximization(float *means,float *stddev,float *apriori, int nbClasses);
		void MICMSegmentation(float beta, int nbClasses);
		void MSASegmentation(float beta, float tmin,float tmax,float coolingrate, int nbClasses);
		void MGraphCutSegmentation(float beta);
		void MInteractiveGraphCutSegmentation(MImage &mask, float sigma);
		
		/* Operators */
		void operator= (const MImage &copy);
		void operator= (float val);
		
	private:
		
		float MZeroMeanGaussian2D(float x, float y, float stddev) {return exp(-(x*x+y*y)/(2*stddev*stddev))/(2*3.1416*stddev*stddev);};
		float MGaussian1D(float x, float mean, float stddev) {return exp(-pow(x-mean,2)/(2*stddev*stddev))/(sqrt(2*3.1416)*stddev);};
		float MLogGaussian1D(float x, float mean, float var) const {return 0.5*logf(2*M_PI*var) + pow(x-mean,2)/(2.0*var);};
		float MPottsEnergy(const MImage &img, int x, int y, int label) const;
		float MComputeGlobalEnergy(const MImage &X, float *mean,float *var, float beta,  int nbClasses) const;
};

#endif
