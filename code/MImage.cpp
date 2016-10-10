#include "MImage.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "gc/GCoptimization.h"
#include <vector>
#include <array>
#include <list>
#include <algorithm>

using std::vector;
using std::array;
using std::list;

MImage::MImage(int xs, int ys, int zs)
{
	MXS = 0;
	MYS = 0;
	MZS = 0;
	MImgBuf = NULL;

	if (xs>0 && ys>0 && zs>0)
		MAllocMemory(xs, ys, zs);

	for (int y = 0;y<MYS;y++)
		for (int x = 0;x<MXS;x++)
			MSetColor(0, x, y);
}

MImage::MImage(int xs, int ys, int zs, float color)
{
	MXS = 0;
	MYS = 0;
	MZS = 0;
	MImgBuf = NULL;

	if (xs>0 && ys>0 && zs>0)
		MAllocMemory(xs, ys, zs);

	for (int y = 0;y<MYS;y++)
		for (int x = 0;x<MXS;x++)
			MSetColor(color, x, y);
}

MImage::MImage(void)
{
	MXS = 0;
	MYS = 0;
	MZS = 0;
	MImgBuf=NULL;
}

MImage::MImage(const MImage &copy)
{
 	MXS = 0;
 	MYS = 0;
 	MZS = 0;
 	MImgBuf=NULL;
	if(copy.MIsEmpty()){
		return;
	}

	MAllocMemory(copy.MXS,copy.MYS,copy.MZS);

	for(int y=0;y<MYS;y++)
		for(int x=0;x<MXS;x++)
			MImgBuf[x][y]=copy.MImgBuf[x][y];

}

MImage::~MImage(void)
{
	MFreeMemory();
}

/* =================================================================================
====================================================================================
======================                    I/O                 ======================
====================================================================================
====================================================================================*/

/*
	Allocates a xs x ys x zs image buffer
*/
void MImage::MAllocMemory(int xs,int ys,int zs)
{
	MFreeMemory();
	if(xs<=0||ys<=0||zs<=0){
		printf("Error !! MImage::MAllocMemory\n");
		return;
	}

	MXS = xs;
	MYS = ys;
	MZS = zs;

	MImgBuf = new RGBPixel*[xs];

	for(int i=0;i<xs;i++)
		MImgBuf[i] = new RGBPixel[ys];
}

/*
	Free the image buffer
*/
void MImage::MFreeMemory(void)
{
	if(MImgBuf==NULL || MXS<=0 || MYS<=0 || MZS<=0){
		MXS=MYS=MZS=0;
	}else {
		for(int i=0;i<MXS;i++)
			delete []MImgBuf[i];
		delete []MImgBuf;
		MImgBuf=NULL;
		MXS=MYS=MZS=0;
	}
}

/*
	load a pgm/ppm image
*/
bool MImage::MLoadImage(const char *fileName)
{

	if(fileName==NULL){
		printf("Error!!! MImage::MLoadImage\n");
		return false;
	}
	FILE_FORMAT ff;
	char tmpBuf[500];
	std::ifstream inFile;
	int maxVal,val;
	char valRaw;
	unsigned char color;

	inFile.open (fileName,std::ios::in);
	if (!inFile.is_open()) {
		printf("Error!!! cant open file %s\n",fileName);
		return false;
	}

	inFile.getline(tmpBuf,500);
	switch(tmpBuf[1]){
		case '2':
			ff=PGM_ASCII;
			MZS=1;
		break;
	}

	int nbComm=0;
	inFile.getline(tmpBuf,500);
	while(tmpBuf[0]=='#'){
		nbComm++;
		inFile.getline(tmpBuf,500);
	}
	inFile.close();

	if(ff==PGM_ASCII)
		inFile.open (fileName,std::ios::in);
	else
		inFile.open (fileName,std::ios::in|std::ios::binary);

	inFile.getline(tmpBuf,500);
	while(nbComm>0){
		nbComm--;
		inFile.getline(tmpBuf,500);
	}

	inFile>>MXS;
	inFile>>MYS;
	inFile>>maxVal;

	MAllocMemory(MXS,MYS,MZS);
	switch(ff){

		case PGM_ASCII:
			for(int y=0;y<MYS;y++)
				for(int x=0;x<MXS;x++){
					inFile>>val;
					MImgBuf[x][y].r = (float)val*255.0/maxVal;
				}
		break;

	}

  inFile.close();
 	printf("File %s opened successfully\n",fileName);
 return true;
}

/*
	save a pgm/ppm image
*/
bool MImage::MSaveImage(const char *fileName, FILE_FORMAT ff)
{
	if(!fileName){
		printf("Error!! MImage::MSaveImage\n");
		return false;
	}
	unsigned char val;
	std::ofstream outFile;

	outFile.open (fileName,std::ios::out|std::ios::binary);
	if (!outFile.is_open()) {
		printf("Error!!! cant open file %s\n",fileName);
		return false;
	}
	switch(ff){
		case PGM_ASCII:
			outFile << "P2" << "\n" << MXS << " "  << MYS << "\n"  << "255" << "\n";
			for(int y=0;y<MYS;y++){
				for(int x=0;x<MXS;x++){
					outFile << (unsigned short)(MImgBuf[x][y].r) << " ";
				}
				outFile << "\n";
			}
		break;
	}
 	outFile.close();
	printf("File %s saved successfully\n",fileName);
	return true;
}



/* =================================================================================
====================================================================================
======================           Point Operations             ======================
====================================================================================
====================================================================================*/


/*
	Every pixel with an intensity > 'tvalue' are set to 255.  The other ones are set to 0.
*/
void MImage::MThreshold(float tvalue)
{
	for(int y=0;y<MYS;y++)
		for(int x=0;x<MXS;x++){

			if(MZS>1){

				if((MImgBuf[x][y].r + MImgBuf[x][y].g+ MImgBuf[x][y].b)/3.0>tvalue){
					MImgBuf[x][y].r = 255;
					MImgBuf[x][y].g = 255;
					MImgBuf[x][y].b = 255;
				}else{
					MImgBuf[x][y].r = 0;
					MImgBuf[x][y].g = 0;
					MImgBuf[x][y].b = 0;
				}

			}	else {
				if(MImgBuf[x][y].r>tvalue)
					MImgBuf[x][y].r = 255;
				else
					MImgBuf[x][y].r = 0;
			}
		}
}

/*
	Rescale the image between 0 and 255
*/
void MImage::MRescale(void)
{
	float maxR,minR;
	float maxG,minG;
	float maxB,minB;

	maxR = maxG = maxB = -99999;
	minR = minG = minB =  99999;
	int X,Y;
	for(int y=0;y<MYS;y++)
		for(int x=0;x<MXS;x++){

			if(MImgBuf[x][y].r>maxR)
				maxR = MImgBuf[x][y].r;
			if(MImgBuf[x][y].r<minR)
				minR = MImgBuf[x][y].r;

			if(	MZS>1){
				if(MImgBuf[x][y].g>maxG)
					maxG = MImgBuf[x][y].g;
				if(MImgBuf[x][y].b>maxB)
					maxB = MImgBuf[x][y].b;

				if(MImgBuf[x][y].g<minG)
					minG = MImgBuf[x][y].g;
				if(MImgBuf[x][y].b<minB)
					minB = MImgBuf[x][y].b;
			}
		}

	for(int y=0;y<MYS;y++)
		for(int x=0;x<MXS;x++){
			MImgBuf[x][y].r = (MImgBuf[x][y].r-minR)*255.0/(maxR-minR);

			if(MZS>1){
				MImgBuf[x][y].g = (MImgBuf[x][y].g-minG)*255.0/(maxG-minG);
				MImgBuf[x][y].b = (MImgBuf[x][y].b-minB)*255.0/(maxB-minB);
			}
		}
}



#define SQR(x) ((x)*(x))
typedef std::list<array<float, 3>> cell_t;
/*
	Mean shift filtering
	
	the implementation is inspired of the following paper

	D. Comanicu, P. Meer: "Mean shift: A robust approach toward feature space analysis".
    IEEE Trans. Pattern Anal. Machine Intell., May 2002.

	The resulting filtered image is copied in the current image (this->MImgBuf)
*/
void MImage::MMeanShift(float SpatialBandWidth, float RangeBandWidth, float tolerance)
{
    // Utility lambda functions
    auto sqdist = [](float x, float y, float z, float u, float v, float w) -> float {
        return SQR(x - u) + SQR(y - v) + SQR(z - w);
    };
    auto vec3Add = [](std::array<float, 3>& sum, float x, float y, float z) -> void {
        sum[0] += x;
        sum[1] += y;
        sum[2] += z;
    };
    auto vec3Div = [](std::array<float, 3>& sum, float value) -> void {
        sum[0] /= value;
        sum[1] /= value;
        sum[2] /= value;
    };
    auto vec3Equal = [&sqdist](std::array<float, 3>& v1, std::array<float, 3>& v2) -> bool {
        return sqdist(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]) < 1.0f;
    };
    auto makeVec = [](float x, float y, float z) -> array<float, 3> {
        array<float, 3> vec;
        vec[0] = x;
        vec[1] = y;
        vec[2] = z;
        return vec;
    };

    // A grid is used for finding neighboring 3D points.
    // Each 3D point is inserted in a grid cell and its 26 neighboring cells.
    // The smallest dimension of a cell must be <= SpatialBandWidth, which is used as the size of the window function.
    int gridSizeX(ceil(float(MXSize()) / SpatialBandWidth)),
        gridSizeY(ceil(float(MYSize()) / SpatialBandWidth)),
        gridSizeZ(ceil(255.0f * RangeBandWidth / SpatialBandWidth));
    float cellSizeX(float(MXSize()) / gridSizeX),
          cellSizeY(float(MYSize()) / gridSizeY),
          cellSizeZ(255.0f * RangeBandWidth / gridSizeZ);
    vector<vector<vector<cell_t>>> grid(gridSizeX, vector<vector<cell_t>>(gridSizeY, vector<cell_t>(gridSizeZ)));

    // Lambda function for finding the grid cell containing the 3D position
    auto getGridPos =
            [cellSizeX, cellSizeY, cellSizeZ, gridSizeX, gridSizeY, gridSizeZ](array<float, 3> pos) -> array<int, 3> {
        array<int, 3> grid_pos;
        grid_pos[0] = std::min(int(pos[0] / cellSizeX), gridSizeX-1);
        grid_pos[1] = std::min(int(pos[1] / cellSizeY), gridSizeY-1);
        grid_pos[2] = std::min(int(pos[2] / cellSizeZ), gridSizeZ-1);
        return grid_pos;
    };

    // Insert each 3D point in the grid
    for (int x = 0; x < MXSize(); ++x)
    {
        for (int y = 0; y < MYSize(); ++y)
        {
            float color = MGetColor(x, y) * RangeBandWidth;
            int xCell(x / cellSizeX), yCell(y / cellSizeY), zCell(color / cellSizeZ);

            // Insert in neighboring cells
            for (int nx = xCell-1; nx <= xCell+1; ++nx)
            {
                for (int ny = yCell-1; ny <= yCell+1; ++ny)
                {
                    for (int nz = zCell-1; nz <= zCell+1; ++nz)
                    {
                        if (nx < 0 || ny < 0 || nz < 0 || nx >= gridSizeX || ny >= gridSizeY || nz >= gridSizeZ)
                            continue;
                        grid[nx][ny][nz].push_back(makeVec(x, y, color));
                    }
                }
            }
        }
    }

    array<float, 3> mean, newMean;
    MImage Y(MXSize(), MYSize(), 1);

    // For each pixel, start at the position of the pixel and move to the nearest density maximum
    for (int x = 0; x < MXSize(); ++x)
    {
        std::cout << "Col " << x << std::endl;
        for (int y = 0; y < MYSize(); ++y)
        {
            mean = {float(x), float(y), MGetColor(x, y) * RangeBandWidth};
            // Move the position until it stabilizes
            for (;;)
            {
                // Move the position to the mean of the window
                newMean = {0.0f, 0.0f, 0.0f};
                int num(0);

                // Get all points that could be inside the window
                array<int, 3> grid_pos = getGridPos(mean);
                cell_t& gridCell = grid[grid_pos[0]][grid_pos[1]][grid_pos[2]];
                for (auto p : gridCell)
                {
                    int u(p[0]), v(p[1]); // X,Y position of the neighbor
                    float uvColor = p[2]; // Z position of the neighbor
                    float epanechnikov = sqdist(mean[0], mean[1], mean[2], u, v, uvColor) / SQR(SpatialBandWidth);
                    if (epanechnikov < 1.0f)
                    {
                        vec3Add(newMean, u, v, uvColor);
                        num += 1;
                    }
                }
                vec3Div(newMean, num); // newMean contains the total, divide by num and we get the mean
                if (vec3Equal(mean, newMean))
                    break;
                mean = newMean;
            }
            // Set pixel in "label field" to the color of the density maximum
            Y.MSetColor(mean[2] / RangeBandWidth, x, y);
        }
    }

    operator=(Y);
}
/* =================================================================================
====================================================================================
======================            Feature extraction          ======================
====================================================================================
====================================================================================*/



/*
	Segmentation with magic wand algorithm

	(xSeed, ySeed) is where the region starts growing and
	tolerance is the criteria to stop the region from growing.

*/
void MImage::MMagicWand(int xSeed, int ySeed, float tolerance)
{
	MImage Y(MXS, MYS, 1, -1.0f);
	Flood(Y, xSeed, ySeed, tolerance, MImgBuf[xSeed][ySeed]);

    // Output segmentation result in the image
    for (int x = 0; x < MXS; x++)
    {
        for (int y = 0; y < MYS; y++)
        {
            float grayLevel = float(Y.MGetColor(x, y) > 0.0f) * 255.0f; // Convert boolean result to float
            MSetColor(grayLevel, x, y);
        }
    }
}

void MImage::Flood(MImage &yOut, int xSeed, int ySeed, float tolerance, RGBPixel &xRef)
{
	for (int x = xSeed - 1; x <= xSeed + 1; x++)
	{
		for (int y = ySeed - 1; y <= ySeed + 1; y++)
		{
			// Check coord is inside image AND pixel not visited
            if (x < 0 || x >= MXS || y < 0 || y >= MYS || yOut.MGetColor(x, y) > -1.0f)
                continue;

            if (fabsf(MGetColor(x, y) - xRef.r) < tolerance) // Check inside range
            {
                yOut.MSetColor(1.0f, x, y);
                Flood(yOut, x, y, tolerance, xRef);
            }
            else
            {
                yOut.MSetColor(0.0f, x, y);
            }
		}
	}
}

/*
	N-class segmentation.  Each class is a Gaussian function defined by
	a mean, stddev and prior value.

	The resulting label Field is copied in the current image (this->MImgBuf)
*/
void MImage::MOptimalThresholding(float *means, float *stddev, float *apriori, int nbClasses)
{
	if(nbClasses <= 1)
    {
        std::cout << "Error : number of classes should be at least 2!" << std::endl;
        return;
    }
    
    int classRange = floor(255.0f / nbClasses);
    float sqTwoPi = sqrtf(2.0f * M_PI);
    
    for(int y = 0; y < MYSize(); y++)
    {
        for(int x = 0; x < MXSize(); x++)
        {
            int c = 0;
            float maxProb = 0.0f;
            for(int k = 0; k < nbClasses; k++)
            {
                float diff = powf(MGetColor(x, y) - means[k], 2.0f);
                float prob = apriori[k] * exp2f(-diff / (2.0f * powf(stddev[k], 2.0f))) / (sqTwoPi * stddev[k]);
                if(prob > maxProb)
                {
                    maxProb = prob;
                    c = k;
                }
            }
            MSetColor(c * classRange, x, y);
        }
    }
}


/*
	N-class KMeans segmentation

	Resulting values are copied in parameters 'means','stddev', and 'apriori'
	The 'apriori' parameter contains the proportion of each class.

	The resulting label Field is copied in the current image (this->MImgBuf)
*/
void MImage::MKMeansSegmentation(float *means,float *stddev,float *apriori,int nbClasses)
{
    MImage Y(MXSize(), MYSize(), 1);
    std::vector<int> classSize(nbClasses);
    float* oldMeans = new float[nbClasses];

    // Init average in a uniform manner (instead of random)
    for (int c = 0; c < nbClasses; c++)
        means[c] = float(c) / nbClasses * 255.0f;

    bool meansHaveChanged = true;
    int iter(0);
    while (meansHaveChanged)
    {
        iter++;

        // Set labels
        for (int x = 0; x < MXSize(); x++) {
            for (int y = 0; y < MYSize(); y++) {
                int closestClass = -1;
                float smallestDist = std::numeric_limits<float>::infinity();

                // Check against all averages
                for (int c = 0; c < nbClasses; c++) {
                    float dist = fabsf(MGetColor(x, y) - means[c]);
                    if (dist < smallestDist) {
                        closestClass = c;
                        smallestDist = dist;
                    }
                }

                Y.MSetColor(float(closestClass), x, y);
            }
        }

        // Keep actual means for comparison
        memcpy(oldMeans, means, nbClasses * sizeof(float));

        // Reset means
        for (int c = 0; c < nbClasses; c++) {
            means[c] = 0.0f;
            classSize[c] = 0;
        }

        // Compute means
        for (int x = 0; x < MXSize(); x++) {
            for (int y = 0; y < MYSize(); y++) {
                int label = int(Y.MGetColor(x, y));
                means[label] += MGetColor(x, y);
                classSize[label] += 1;
            }
        }
        for (int c = 0; c < nbClasses; c++) {
            means[c] /= (classSize[c] == 0 ? 1 : classSize[c]);
        }

        // Check if means have changed
        meansHaveChanged = false;
        for (int c = 0; c < nbClasses; c++)
        {
            float dist = fabsf(means[c] - oldMeans[c]);
            if (dist > 0.01f)
            {
                meansHaveChanged = true;
            }
        }

    } // End of while

    // Compute variance
    std::fill(stddev, stddev + nbClasses, 0.0f);
    for (int x = 0; x < MXSize(); x++)
    {
        for (int y = 0; y < MYSize(); y++)
        {
            int c = int(Y.MGetColor(x, y));
            stddev[c] += pow(MGetColor(x, y) - means[c], 2.0f);
        }
    }

    float totalPixels = MXSize() * MYSize();

    // Get standard deviation and 'a priori'
    for (int c = 0; c < nbClasses; c++)
    {
        stddev[c] = sqrtf(stddev[c] / classSize[c]);

        apriori[c] = float(classSize[c]) / totalPixels;
    }

    printf("Iter: %i\n", iter);

    // Copy label field to this image
    operator=(Y);

    delete[] oldMeans;
}


/*
	N-class Soft KMeans segmentation

	Resulting values are copied in parameters 'means' and 'stddev'.
	The 'apriori' parameter contains the proportion of each class.

	The resulting label Field is copied in the current image (this->MImgBuf)
*/
void MImage::MSoftKMeansSegmentation(float *means,float *stddev,float *apriori,float beta, int nbClasses)
{
    std::vector<MImage> Y(nbClasses, MImage(MXSize(), MYSize(), 1));
    std::vector<int> classSize(nbClasses);
    MImage bestClassForSite(MXSize(), MYSize(), 1);
    float* exponentialTerms = new float[nbClasses];

    // Init average in a uniform manner (instead of random)
    for (int c = 0; c < nbClasses; c++)
        means[c] = float(c) / nbClasses * 255.0f;

    bool meansHaveChanged = true;
    int iter(0);
    while (meansHaveChanged)
    {
        iter++;

        // Reset size of each class
        std::fill(classSize.begin(), classSize.end(), 0);

        // Set probabilities
        for (int x = 0; x < MXSize(); x++)
        {
            for (int y = 0; y < MYSize(); y++)
            {
                float pixelValue = MGetColor(x, y);

                // Compute all exponential terms and the denominator
                float denom(0.0f);
                for (int c = 0; c < nbClasses; c++)
                {
                    float d_r = beta * fabsf(pixelValue - means[c]);
                    float term = expf(-d_r);
                    denom += term;
                    exponentialTerms[c] = term;
                }

                // Compute probability of being in class c for pixel (x,y)
                // Update label field using highest probability class
                float highestProb = 0.0f;
                int bestClass = 0;
                for (int c = 0; c < nbClasses; c++)
                {
                    float numer = exponentialTerms[c];

                    float P = numer / denom;

                    Y[c].MSetColor(P, x, y);

                    if (P > highestProb)
                    {
                        highestProb = P;
                        bestClass = c;
                    }
                }
                bestClassForSite.MSetColor(float(bestClass), x, y);
                classSize[bestClass] += 1;
            }
        }

        // Compute new means
        meansHaveChanged = false;
        for (int c = 0; c < nbClasses; c++)
        {
            float numer(0.0f), denom(0.0f);
            for (int x = 0; x < MXSize(); x++)
            {
                for (int y = 0; y < MYSize(); y++)
                {
                    // Get probability that (x,y) belongs to class c
                    float P = Y[c].MGetColor(x, y);

                    numer += P * MGetColor(x, y);
                    denom += P;
                }
            }

            float newMean = numer / denom; // weighted average = (weighted sum) / (sum of weights)
            if (fabsf(newMean - means[c]) > 0.01)
                meansHaveChanged = true;
            means[c] = newMean;
        }
    }

    printf("Iter: %i", iter);

    // Compute variance
    std::fill(stddev, stddev + nbClasses, 0.0f);
    for (int x = 0; x < MXSize(); x++)
    {
        for (int y = 0; y < MYSize(); y++)
        {
            int c = int(bestClassForSite.MGetColor(x, y));
            stddev[c] += pow(MGetColor(x, y) - means[c], 2.0f);
        }
    }

    float totalPixels = MXSize() * MYSize();

    // Get standard deviation and 'a priori'
    for (int c = 0; c < nbClasses; c++)
    {
        stddev[c] = sqrtf(stddev[c] / classSize[c]);

        apriori[c] = float(classSize[c]) / totalPixels;
    }

    // Copy label field in this image
    operator=(bestClassForSite);

    delete[] exponentialTerms;
}


/*
	N-class Expectation maximization segmentation
	
	init values are in 'means', 'stddev' and 'apriori'

	Resulting values are copied in parameters 'means', 'stddev' and 'apriori'
	The 'apriori' parameter contains the proportion of each class.

	The resulting label Field is copied in the current image (this->MImgBuf)
*/
void MImage::MExpectationMaximization(float *means,float *stddev,float *apriori, int nbClasses)
{
    const float NORM_CONSTANT = 1.0f / sqrtf(2.0f * M_PI);
    const float ONE_OVER_NUM_SITES = 1.0f / (MXSize() * MYSize());
    std::vector<float> gaussianProbability(nbClasses);
    std::vector<MImage> P(nbClasses, MImage(MXSize(), MYSize(), 1)); // Contains P(c | x_s)
    auto are_different = [](float a, float b, float eps) -> bool{
        return fabsf(a - b) > eps;
    };

    bool paramsHaveChanged = true;
    while (paramsHaveChanged)
    {
        // 1. Compute P(c | x_s, theta) for all pixels and all classes
        for (int x = 0; x < MXSize(); x++)
        {
            for (int y = 0; y < MYSize(); y++)
            {
                float x_s = MGetColor(x, y);

                // 1.a Compute P(x_s | i, theta_i) * P(i) for each class i (and sum them all)
                float probSum(0.0f);
                for (int c = 0; c < nbClasses; c++)
                {
                    float conditionalProb = NORM_CONSTANT / stddev[c]
                                          * expf(-1.0f * powf(x_s - means[c], 2.0f) / (2.0f * powf(stddev[c], 2.0f)));
                    probSum += gaussianProbability[c] = conditionalProb * apriori[c];
                }

                // 2.a Compute P(c | x_s, theta) for all classes
                for (int c = 0; c < nbClasses; c++)
                {
                    P[c].MSetColor(gaussianProbability[c] / probSum, x, y);
                }
            }
        }

        // 2. Compute gaussian parameters
        paramsHaveChanged = false;
        for (int c = 0; c < nbClasses; c++)
        {
            float meanNumer(0.0f), varianceNumer(0.0f), sumOfCondProbs(0.0f);
            for (int x = 0; x < MXSize(); x++)
            {
                for (int y = 0; y < MYSize(); y++)
                {
                    float x_s = MGetColor(x, y);
                    float conditionalProb = P[c].MGetColor(x, y);

                    sumOfCondProbs += conditionalProb; // This is the denom for both the mean and the variance

                    meanNumer += conditionalProb * x_s;
                    varianceNumer += conditionalProb * powf(x_s - means[c], 2.0f);
                }
            }

            float newMean = meanNumer / sumOfCondProbs;
            float newStddev = sqrtf(varianceNumer / sumOfCondProbs);
            float newApriori = ONE_OVER_NUM_SITES * sumOfCondProbs;

            if (   are_different(newMean, means[c], 0.1f)
                || are_different(newStddev, stddev[c], 0.1f)
                || are_different(newApriori, apriori[c], 0.1f))
            {
                paramsHaveChanged = true;
            }

            means[c] = newMean;
            stddev[c] = newStddev;
            apriori[c] = newApriori;
        }

    } // End of while

    // NOTE: The label field is not copied in the current image.
    // The label field is computed using Optimal Thresholding. (See tp1B.cpp)
}

/*
	N-class ICM segmentation

	beta : Constant multiplying the apriori function

	The resulting label Field is copied in the current image (this->MImgBuf)

*/
void MImage::MICMSegmentation(float beta, int nbClasses)
{
	bool ValueChanged;
	MImage Yprev;
	MImage Y = *this;
	float* means = new float[nbClasses];
	float* stdev = new float[nbClasses];
	float* apriori = new float[nbClasses];
	Y.MKMeansSegmentation(means, stdev, apriori, nbClasses);
	do
	{
		ValueChanged = false;
		Yprev = Y;
		for (int i = 0; i < MXSize(); i++)
		{
			for (int j = 0; j < MYSize(); j++)
			{
				int NearestClass = -1;
				float min = std::numeric_limits<float>::infinity();
				for (int c = 0; c < nbClasses; c++)
				{
                    float Uc = -log(exp(-pow(MGetColor(i, j) - means[c], 2.0f) / (2.0*pow(stdev[c], 2.0))) / (stdev[c] * sqrt(2.0f*M_PI)));
					float Wc = 0.0f;
					for (int m = -1; m <= 1; m++)
					{
						for (int n = -1; n <= 1; n++)
						{
							if (i + m >= 0 && i + m < MXS && j + n >= 0 && j + n < MYS)
							{
								Wc += Y.MGetColor(i + m, j + n) != c ? 1 : 0;
							}
						}
					}
					Wc *= beta;

					if (Uc + Wc < min)
					{
						min = Uc + Wc;
						NearestClass = c;
					}
				}

				ValueChanged |= (fabsf(NearestClass - Y.MGetColor(i, j)) > 1e-5);
				Y.MSetColor(NearestClass, i, j);
			}
		}
	} while (ValueChanged);

	operator=(Y);

	delete means;
	delete stdev;
	delete apriori;
}

/*
	N-class Simulated annealing segmentation

	beta : Constant multiplying the apriori function
	Tmax : Initial temperature (initial temperature)
	Tmin : Minimal temperature allowed (final temperature)
	coolingRate : rate by which the temperature decreases

	The label Field copied in the current image (this->MImgBuf)

*/
void MImage::MSASegmentation(float beta,float Tmin, float Tmax, float coolingRate, int nbClasses)
{
    const float NORM_CONSTANT = 1.0f / sqrtf(2.0f * M_PI);
    const int xNeigh[] = {-1,  0,  1, -1, 1, -1, 0, 1};
    const int yNeigh[] = {-1, -1, -1,  0, 0,  1, 1, 1};
    auto are_different = [](float a, float b, float eps) -> bool{
        return fabsf(a - b) > eps;
    };

    float* means = new float[nbClasses];
    float* stddev = new float[nbClasses];
    float* apriori = new float[nbClasses]; // Will not be used, we give it to the KMeans function
    float T = Tmax;
    double* P = new double[nbClasses];

    // Initialize label field and parameters
    MImage Y = *this;
    Y.MKMeansSegmentation(means, stddev, apriori, nbClasses);

    MImage debug = Y;
    debug.MRescale();
    debug.MSaveImage("SAInitKMeans.pgm", PGM_ASCII);

    for (int x = 0; x < MXSize(); x++)
    {
        for (int y = 0; y < MYSize(); y++)
        {
            Y.MSetColor(roundf(float(std::rand()) / RAND_MAX * (nbClasses - 1)), x, y);
        }
    }

    int idebug(0);
    do
    {
        for (int x = 0; x < MXSize(); x++)
        {
            for (int y = 0; y < MYSize(); y++)
            {
                float x_s = MGetColor(x, y);

                // 1. Compute probability for each class
                double sumOfP(0.0f);
                for (int c = 0; c < nbClasses; c++)
                {
                    float U_c = 0.5f * powf((x_s - means[c]) / stddev[c], 2.0f);

                    int diffNeighbors(0);
                    for(int n = 0; n < 8; n++) // 8 neighbors
                    {
                        int xn = x + xNeigh[n];
                        int yn = y + yNeigh[n];
                        if (xn < 0 || yn < 0 || xn >= MXSize() || yn >= MYSize())
                            continue;
                        if (are_different(Y.MGetColor(xn, yn), Y.MGetColor(x, y), 0.5f)) // We compare "integers"
                            diffNeighbors += 1;
                    }
                    float W_c = beta * diffNeighbors;

                    P[c] = exp(-1.0f * (U_c + W_c) / T);
                    sumOfP += P[c];
                }

                // 2. Choose a class
                double p = double(std::rand()) / RAND_MAX; // Random number in [0,1]
                double sumOfPreviousP(0.0f);

                bool showDebug = false; // = idebug % 1000000 == 0;
                if(showDebug) std::cout << std::endl;

                int c(0);
                for (; c < nbClasses; c++)
                {
                    if(showDebug) std::cout << P[c] << " ";

                    P[c] /= sumOfP; // Normalize P's so that their sum is 1

                    if(showDebug) std::cout << P[c] << " " << sumOfPreviousP << " " << P[c] + sumOfPreviousP << " " << p << std::endl;

                    if (p < sumOfPreviousP + P[c])
                    {
                        break;
                    }
                    sumOfPreviousP += P[c];
                }
                c = std::min(c, nbClasses-1);
                Y.MSetColor(float(c), x, y);

                idebug++;
            }
        }
        T = T * coolingRate;
    } while (T > Tmin);

    // Copy label field in the current instance
    operator=(Y);

    delete[] means;
    delete[] stddev;
    delete[] apriori;
    delete[] P;
}


/*
 * Smooth cost function for use with GCoptimizationGridGraph::setSmoothCost(SmoothCostFnExtra, void*)
*/
double SmoothCost(int s1, int s2, int l1, int l2, void *data)
{
    float* floatData = (float*)data;
    float sigma = floatData[0];
    float* pixData = floatData + 1;

    return double(1000.0f * fabsf(l1 - l2) * expf(-0.5 * powf((pixData[s1] - pixData[s2]) / sigma, 2.0f)));
}

/*
	Interactive graph cut segmentation
	
	the implementation is inspired of the following paper

	Y Boykov and M-P Jolly "Interactive Graph Cuts for Optimal Boundary & Region Segmentation of Objects in N-D images". 
	In International Conference on Computer Vision, (ICCV), vol. I, pp. 105-112, 2001.

	The resulting label Field is copied in the current image (this->MImgBuf)
*/
void MImage::MInteractiveGraphCutSegmentation(MImage &mask, float sigma)
{
    GCoptimizationGridGraph gc(MXSize(), MYSize(), 2);
    float lambda(0.001f);

    // Get means for background and foreground
    float foreSum(0.0f), backSum(0.0f);
    int foreCount(0), backCount(0);
    for (int x = 0; x < MXSize(); ++x)
    {
        for (int y = 0; y < MYSize(); ++y)
        {
            if (mask.MGetColor(x, y) > 250.0f)
            {
                // Foreground selection
                foreSum += MGetColor(x, y);
                foreCount++;
            }
            else if (mask.MGetColor(x, y) > 10.0f)
            {
                // Background selection
                backSum += MGetColor(x, y);
                backCount++;
            }
        }
    }
    float foreground = foreSum / foreCount;
    float background = backSum / backCount;

    // Make 1D vector
    int length = MXSize() * MYSize();
    float* data = new float[length + 1];
    data[0] = sigma;
    float* pixelData = data + 1;
    int pInd = 0;
    for (int x = 0; x < MXSize(); ++x)
    {
        for (int y = 0; y < MYSize(); ++y)
        {
            pixelData[pInd++] = MGetColor(x, y);
        }
    }

    // Data cost
    const float highCost = 1e7;
    for (int x = 0; x < MXSize(); ++x)
    {
        for (int y = 0; y < MYSize(); ++y)
        {
            int pixInd = x * MYSize() + y;

            // Foreground: 1, Background: 0
            if (mask.MGetColor(x, y) > 250.0f)
            {
                // Foreground selection
                gc.setDataCost(pixInd, 0, highCost);
                gc.setDataCost(pixInd, 1, 0.0f); // No cost for choosing foreground
            }
            else if (mask.MGetColor(x, y) > 10.0f)
            {
                // Background selection
                gc.setDataCost(pixInd, 0, 0.0f); // No cost for choosing background
                gc.setDataCost(pixInd, 1, highCost);
            }
            else
            {
                // Not selected
                gc.setDataCost(pixInd, 0, lambda * powf(MGetColor(x, y) - background, 2.0f));
                gc.setDataCost(pixInd, 1, lambda * powf(MGetColor(x, y) - foreground, 2.0f));
            }
        }
    }

    // Smooth cost
    gc.setSmoothCost(SmoothCost, data);

    // Run the algorithm
    gc.expansion(6);

    // Get results
    pInd = 0;
    for (int x = 0; x < MXSize(); ++x)
    {
        for (int y = 0; y < MYSize(); ++y)
        {
            MSetColor(gc.whatLabel(pInd++), x, y);
        }
    }

    MRescale();
}


/* =================================================================================
====================================================================================
======================              Operators                 ======================
====================================================================================
====================================================================================*/
void MImage::operator= (const MImage &copy)
{
	if(copy.MIsEmpty()){
		MFreeMemory();
		return;
	}

	if(!MSameSize(copy))
		MAllocMemory(copy.MXS,copy.MYS,copy.MZS);

	for(int y=0;y<MYS;y++)
		for(int x=0;x<MXS;x++)
			MImgBuf[x][y]=copy.MImgBuf[x][y];

}

void MImage::operator= (float val)
{
	if(MIsEmpty()){
		return;
	}

	for(int y=0;y<MYS;y++)
		for(int x=0;x<MXS;x++)
			for(int z=0;z<MZS;z++)
				MSetColor(val,x,y,z);

}

/* =================================================================================
====================================================================================
======================              Ohters                 ======================
====================================================================================
====================================================================================*/


float MImage::MPottsEnergy(const MImage &img, int x, int y, int label) const
{
}

float MImage::MComputeGlobalEnergy(const MImage &X, float *mean, float *stddev, float beta, int nbClasses) const
{
}

