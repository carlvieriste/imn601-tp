#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "MImage.h"

/*
	Mean shift image segmentation
*/
int main(int argc, char **argv)
{
	MImage img;
	
	if(argc<4){
		printf("\n Usage : tp1D image SpatialBandWidth RangeBandWidth\n\n"); 
		return 1;
	}
	
	img.MLoadImage(argv[1]);
	img.MMeanShift(atof(argv[2]),atof(argv[3]),40);
	img.MSaveImage("outMeanShift.pgm",PGM_ASCII);

	return 0;
}
