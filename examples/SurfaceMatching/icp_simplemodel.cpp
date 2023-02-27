/*
 *@brief:   简单模型的ICP
 *@time:    2023/2/27
 *@author:  geodoer
 *@desc:
    前提假设：两个模型是很简单的，几乎是ctrl+c、ctrl+v而来
    处理：
        1. 将模型中心都平移到原点上
        2. 再进行ICP
 */
#include <iostream>
#include <vector>

#include "opencv2/core/utility.hpp"
#include "opencv2\opencv_modules.hpp"

#include "opencv2\surface_matching.hpp"
#include "opencv2\surface_matching\ppf_helpers.hpp"
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::ppf_match_3d;

void computeBbox(cv::Mat pc, cv::Vec2f& xRange, cv::Vec2f& yRange, cv::Vec2f& zRange)
{
    cv::Mat pcPts = pc.colRange(0, 3);
    int num = pcPts.rows;

    float* points = (float*)pcPts.data;

    xRange[0] = points[0];
    xRange[1] = points[0];
    yRange[0] = points[1];
    yRange[1] = points[1];
    zRange[0] = points[2];
    zRange[1] = points[2];

    for (int ind = 0; ind < num; ind++)
    {
        const float* row = (float*)(pcPts.data + (ind * pcPts.step));
        const float x = row[0];
        const float y = row[1];
        const float z = row[2];

        if (x < xRange[0])
            xRange[0] = x;
        if (x > xRange[1])
            xRange[1] = x;

        if (y < yRange[0])
            yRange[0] = y;
        if (y > yRange[1])
            yRange[1] = y;

        if (z < zRange[0])
            zRange[0] = z;
        if (z > zRange[1])
            zRange[1] = z;
    }
}

cv::Point3f getCenter(const Mat& model)
{
    cv::Vec2f range[3];
    computeBbox(model, range[0], range[1], range[2]);
    
    float center[3];
    for (int i{ 0 }; i < 3; ++i)
    {
        center[i] = (range[i][0] + range[i][1]) * 0.5;
    }

    return { center[0], center[1], center[2] };
}

Matx44d getTranslation(const cv::Point3f& p)
{
    return {
        1, 0, 0, p.x,
        0, 1, 0, p.y,
        0, 0, 1, p.z,
        0, 0, 0, 1
    };
}

int main(int argc, char** argv)
{
    string modelAPath = DATA_PATH "icp_test/74.obj.0.050000.ply";
    string modelBPath = DATA_PATH "icp_test/73.obj.0.050000.ply";
    string modelATransformPath = modelAPath + ".transform.ply";
    
    Mat modelA = loadPLYSimple(modelAPath.c_str(), 0);
    Mat modelB = loadPLYSimple(modelBPath.c_str(), 0);

    // 将模型中心平移到原点上 => 转成模型坐标系（模型坐标系的原点 = 世界坐标系中，box的中心点）
    Mat modelALocal, modelBLocal;
    auto centerA = getCenter(modelA);
	auto centerB = getCenter(modelB);
    {
        auto matrixA = getTranslation(-centerA);
        auto matrixB = getTranslation(-centerB);

        modelALocal = transformPCPose(modelA, matrixA);
        modelBLocal = transformPCPose(modelB, matrixB);

        //保存查看
        writePLY(modelALocal, (modelAPath + ".local.ply").c_str());
        writePLY(modelBLocal, (modelBPath + ".local.ply").c_str());
    }

    int64 t1 = cv::getTickCount();

    ICP icp(100, 0.005f, 2.5f, 8);
    double residual;
    Matx44d pose;
    icp.registerModelToScene(modelALocal, modelBLocal, residual, pose);

    int64 t2 = cv::getTickCount();
    cout << endl << "ICP Elapsed Time " << (t2 - t1) / cv::getTickFrequency() << " sec" << endl;

    std::cout << "ICP residual " << residual << std::endl;
    std::cout << "ICP pose " << pose << std::endl;

    Mat result_LocalAToB  = transformPCPose(modelALocal, pose);
    Mat result_AToB = transformPCPose(result_LocalAToB, getTranslation(centerB));
    writePLY(result_AToB, modelATransformPath.c_str());

    return 0;
}