#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "MImage.h"


/*
	ICM and Simulated anealing
*/
int main(int argc, char **argv)
{
	MImage img1,img2;
	
	if(argc<4){
		printf("\n Usage : tp1C image beta nbClasses\n\n"); 
		return 1;
	}
	
	img1.MLoadImage(argv[1]);

	img2=img1;	
	img2.MICMSegmentation(atof(argv[2]),atoi(argv[3]));
	img2.MRescale();
	img2.MSaveImage("outICM.pgm",PGM_ASCII);

	img2=img1;	
	img2.MSASegmentation(atof(argv[2]),0.01,1000,0.98,atoi(argv[3]));
	img2.MRescale();
	img2.MSaveImage("outSA.pgm",PGM_ASCII);

	return 0;
}
