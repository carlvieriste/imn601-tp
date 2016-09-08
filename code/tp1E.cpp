#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include "MImage.h"

/*
	Graph cut interactive segmentation	
*/ 
int main(int argc, char **argv)
{
	MImage img;
	MImage mask;
	
	if(argc<4){
		printf("\n Usage : tp1E image mask sigma\n\n"); 
		return 1;
	}

	img.MLoadImage(argv[1]);
	mask.MLoadImage(argv[2]);
	
	img.MInteractiveGraphCutSegmentation(mask,atof(argv[3]));

	img.MSaveImage("outIGC.pgm",PGM_ASCII);

	
	return 0;
}
