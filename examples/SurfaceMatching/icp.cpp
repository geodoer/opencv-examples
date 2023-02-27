/*
 *@brief:   opencv icp简单示例
 *@time:    2023/2/27
 *@author:  geodoer
 */
#include <iostream>
#include <vector>

#include "opencv2/core/utility.hpp"
#include "opencv2\opencv_modules.hpp"

#include "opencv2\surface_matching.hpp"
#include "opencv2\surface_matching\ppf_helpers.hpp"

using namespace std;
using namespace cv;
using namespace cv::ppf_match_3d;

int main(int argc, char** argv)
{
    string modelAPath = DATA_PATH "icp_test/74.ply";
    string modelBPath = DATA_PATH "icp_test/73.ply";
    string modelATransformPath = modelAPath + "transform.ply";

    Mat modelA = loadPLYSimple(modelAPath.c_str(), 0);
    Mat modelB = loadPLYSimple(modelBPath.c_str(), 0);

    /**
     * @brief: 创建ICP对象
     * @param iterations: 最大迭代次数
     * @param tolerence: 控制ICP算法每次迭代的精度
     * @param rejectionScale: 在ICP算法的 '删除离群点(reject outliers)' 步骤中的scale系数
     * @param numLevels: 金字塔的层数。太深的金字塔层数可以提高计算速度，但最终的精度会降低。过
     于粗略的金字塔，虽然会提高精度，但是在第一次计算时，会带来计算量的问题。一般设在[4, 10]之间内
     较好。
     * @param sampleType: 目前该参数被忽略。
     * @param numMaxCorr: 目前该参数被忽略。
     */
    ICP icp(100, 0.005f, 2.5f, 8);

    int64 t1 = cv::getTickCount();

    /**
	 * @brief: 使用 'Picky ICP' 算法对齐场景和模型点，同时返回残差和姿态
	 * @param srcPc/dstPc: 模型/场景3D坐标+法向量集合。大小为(Nx6)，且目前只支持 CV_32F 类型。
    	场景和模型点数量不用相同。
	 * @param residual: 最终的残差
	 * @param pose: 'srcPc' 到 'dstPc' 点集 的变换矩阵
	 */
    double residual;
    Matx44d pose;
    icp.registerModelToScene(modelA, modelB, residual, pose);

    std::cout << "ICP residual " << residual << std::endl;
    std::cout << "ICP pose " << pose << std::endl;

    Mat result = transformPCPose(modelA, pose);
    writePLY(result, modelATransformPath.c_str());

    int64 t2 = cv::getTickCount();
    cout << endl << "ICP Elapsed Time " << (t2 - t1) / cv::getTickFrequency() << " sec" << endl;

    return 0;
}