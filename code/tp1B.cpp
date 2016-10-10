#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include "MImage.h"

/*
	"optimal" thresholding, K-means, Soft K-means, and EM
*/ 
int main(int argc, char **argv)
{
	MImage img1,img2;
	float *means;
	float *stddev;
	float *apriori;
	int nbClasses;
	
	if(argc<4){
		printf("\n Usage : tp1B image beta nbClasses\n\n"); 
		return 1;
	}
	
	nbClasses = atoi(argv[3]);
	means = new float[nbClasses];
	stddev = new float[nbClasses];
	apriori = new float[nbClasses];
	
	/*********************************************************************/
	/*****     K-Means                            ************************/ 
	/*********************************************************************/
	img1.MLoadImage(argv[1]);
	img2 = img1;
	
	printf("\n***********************************\n");
	printf("****    K-Means       *************\n");
	printf("***********************************\n");
	img2.MKMeansSegmentation(means,stddev,apriori,nbClasses);
	
	printf(" Means : "); for(int i=0;i<nbClasses;i++) printf(" %f",means[i]);
	printf("\n Stddev : "); for(int i=0;i<nbClasses;i++) printf(" %f",stddev[i]);
	printf("\n Prior : "); for(int i=0;i<nbClasses;i++) printf(" %f",apriori[i]);
	img2.MRescale();
	img2.MSaveImage("outKMeans.pgm",PGM_ASCII);
	
	/*****     Optimal Thresholding               ************************/ 
	img2 = img1;
	img2.MOptimalThresholding(means,stddev,apriori,nbClasses);
	img2.MRescale();
	img2.MSaveImage("outOptimalThresholdingKM.pgm",PGM_ASCII);
	
	
	/*********************************************************************/
	/*****     Soft K-Means                       ************************/ 
	/*********************************************************************/
	img2 = img1;
	printf("\n***********************************\n");
	printf("****   Soft K-Means       *********\n");
	printf("***********************************\n");
	img2.MSoftKMeansSegmentation(means,stddev,apriori,atof(argv[2]),nbClasses);

	printf(" Means : "); for(int i=0;i<nbClasses;i++) printf(" %f",means[i]);
	printf("\n Stddev : "); for(int i=0;i<nbClasses;i++) printf(" %f",stddev[i]);
	printf("\n Prior : "); for(int i=0;i<nbClasses;i++) printf(" %f",apriori[i]);
	
	/*****     Optimal Thresholding               ************************/ 
	img2 = img1;
	img2.MOptimalThresholding(means,stddev,apriori,nbClasses);
	img2.MRescale();
	img2.MSaveImage("outOptimalThresholdingSKM.pgm",PGM_ASCII);
	
	
	
	/*********************************************************************/
	/*****      Expectation Maximization          ************************/ 
	/*********************************************************************/
	img2 = img1;
	printf("\n***********************************\n");
	printf("****  Expectation-Maximization   ****\n");
	printf("***********************************\n");
	img2.MExpectationMaximization(means,stddev,apriori, nbClasses);

	printf(" Means : "); for(int i=0;i<nbClasses;i++) printf(" %f",means[i]);
	printf("\n Stddev : "); for(int i=0;i<nbClasses;i++) printf(" %f",stddev[i]);
	printf("\n Prior : "); for(int i=0;i<nbClasses;i++) printf(" %f",apriori[i]);
	
	/*****     Optimal Thresholding               ************************/ 
	img2 = img1;
	img2.MOptimalThresholding(means,stddev,apriori,nbClasses);
	img2.MRescale();
	img2.MSaveImage("outOptimalThresholdingEM.pgm",PGM_ASCII);
	
	
	delete []means;
	delete []stddev;
	delete []apriori;
	return 0;
}
