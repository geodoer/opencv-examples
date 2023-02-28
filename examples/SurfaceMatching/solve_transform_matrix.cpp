/*
 *\brief 根据点对，解算出刚体变换矩阵
 */
#include<opencv2/opencv.hpp>

cv::Mat solveTransformMatrix(const std::vector<cv::Point3d>& A, const std::vector<cv::Point3d>& B)
{
	int numOfPnts = A.size();

	//# 至少3个点对
	if(numOfPnts < 3 || numOfPnts != B.size())
	{
		return {};
	}

	//# 计算质心
	cv::Point3d centroidA, centroidB;
	{
		for (int i{ 0 }; i < numOfPnts; ++i)
		{
			centroidA += A[i];
			centroidB += B[i];
		}
		
		centroidA /= numOfPnts;
		centroidB /= numOfPnts;
	}

	//# 去质心坐标
	cv::Mat ASubCentroid(3, numOfPnts, CV_64FC1);
	cv::Mat BSubCentroid(3, numOfPnts, CV_64FC1);

	for (int i = 0; i < numOfPnts; ++i)//N组点
	{
		//三行
		ASubCentroid.at<double>(0, i) = A[i].x - centroidA.x;
		ASubCentroid.at<double>(1, i) = A[i].y - centroidA.y;
		ASubCentroid.at<double>(2, i) = A[i].z - centroidA.z;

		BSubCentroid.at<double>(0, i) = B[i].x - centroidB.x;
		BSubCentroid.at<double>(1, i) = B[i].y - centroidB.y;
		BSubCentroid.at<double>(2, i) = B[i].z - centroidB.z;
	}

	//# 计算旋转矩阵
	cv::Mat matS = ASubCentroid * BSubCentroid.t();

	cv::Mat matU, matW, matV;
	cv::SVDecomp(matS, matW, matU, matV);

	cv::Mat matTemp = matU * matV;
	double det = cv::determinant(matTemp);//行列式的值

	double datM[] = { 1, 0, 0, 0, 1, 0, 0, 0, det };
	cv::Mat matM(3, 3, CV_64FC1, datM);

	cv::Mat matR = matV.t() * matM * matU.t();

	//# 计算平移量
	double* datR = (double*)(matR.data);
	double delta_X = centroidB.x - (centroidA.x * datR[0] + centroidA.y * datR[1] + centroidA.z * datR[2]);
	double delta_Y = centroidB.y - (centroidA.x * datR[3] + centroidA.y * datR[4] + centroidA.z * datR[5]);
	double delta_Z = centroidB.z - (centroidA.x * datR[6] + centroidA.y * datR[7] + centroidA.z * datR[8]);

	//# 转成4x4矩阵
	cv::Mat R_T = (cv::Mat_<double>(4, 4) <<
		matR.at<double>(0, 0), matR.at<double>(0, 1), matR.at<double>(0, 2), delta_X,
		matR.at<double>(1, 0), matR.at<double>(1, 1), matR.at<double>(1, 2), delta_Y,
		matR.at<double>(2, 0), matR.at<double>(2, 1), matR.at<double>(2, 2), delta_Z,
		0, 0, 0, 1
		);

	return R_T;
}

int main()
{
	std::vector<cv::Point3d> srcPoints
	{
		{1,3,2},
		{2,3,5},
		{3,7,4},
		{4,9,3},
		{5,4,2},
		{6,4,6},
		{7,0,2}
	};
	std::vector<cv::Point3d> dstPoints = srcPoints;

	auto matrix = solveTransformMatrix(srcPoints, dstPoints);
	std::cout << matrix << std::endl;
	return 0;
}