/*
 *@brief:   批量匹配
 *@time:    2023/2/27
 *@author:  geodoer
 *@desc:
 *  1. 遍历in_dir下的所有文件files
 *  2. 取files[0]作为模板（会将物体中心移动到原点），对其他file逐一进行匹配，计算出变换矩阵
 *  3. 将files[0] + 变换矩阵 的结果输出，与原始数据进行比对
 */
#include "opencv2/surface_matching.hpp"
#include <iostream>
#include "opencv2/surface_matching/ppf_helpers.hpp"
#include "opencv2/core/utility.hpp"

using namespace std;
using namespace cv;
using namespace ppf_match_3d;

struct Timing
{
    Timing(const std::string& title)
        : begin(cv::getTickCount())
        , title(title)
    {
    }
    ~Timing()
    {
        auto end = cv::getTickCount();

        std::cout << endl << title << " "
            << (double)(end - begin) / cv::getTickFrequency()
            << " sec";
    }

    std::string title;
    int64 begin;
};

#include<filesystem>
template<typename FileRanges>
void getFiles(const std::string& in_dir, FileRanges& files)
{
    for (const auto& item : std::filesystem::directory_iterator(in_dir))
    {
        if (std::filesystem::is_directory(item.status()))
        {
            continue;
        }

        auto path = item.path().string();
        files.emplace_back(path);
    }
}

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

Mat translateCenter(const Mat& model)
{
    auto center = getCenter(model);
    auto matrix = getTranslation(-center);
    return transformPCPose(model, matrix);;
}

void process(const std::string& dir, const std::string& out_dir)
{
    std::vector<std::string> files;
    getFiles(dir, files);

    //取第一个作为模板，其他作为匹配
    Mat templateModel = loadPLYSimple(files[0].c_str(), 1);
    templateModel = translateCenter(templateModel);

    //与其他的进行匹配
    int errorCnt = 0;

    for (int i{ 1 }, size = files.size();
        i < size;
        ++i)
    {
        const auto& file = files[i];
        auto fileName = std::filesystem::path(file).filename().string();

        Mat model = loadPLYSimple(file.c_str(), 1);
        auto center = getCenter(model);
        model = translateCenter(model);

        //匹配
        ppf_match_3d::PPF3DDetector detector;
        {
            //Timing t("[Traning template model] ");
            detector.trainModel(templateModel);
        }

        std::vector<Pose3DPtr> results;
        detector.match(model, results, 1.0/40.0, 0.05);

        if (results.empty())
        {
            ++errorCnt;
            std::cout << "Error! No result" << file << std::endl;
            continue;
        }

        //ICP精配准
        ICP icp(100, 0.005f, 2.5f, 8);
        icp.registerModelToScene(templateModel, model, results);

        //取最佳的结果
        int minIdx = -1;
        double minValue = std::numeric_limits<double>::max();

        for (int i{ 0 }, size = results.size(); i < size; ++i)
        {
            const auto& result = results[i];
            const auto& residual = result->residual;

            if (residual > 1)
            {
                continue; //超出1直接判定为不行
            }

            if (residual > minValue)
            {
                continue;
            }

            minIdx = i;
            minValue = residual;
        }

        if (minIdx == -1)
        {
            ++errorCnt;
            std::cout << "Error! residual error!" << file << std::endl;
            continue;
        }

        //输出
        auto path = out_dir + "/" + fileName;
        Mat result_LocalAToB = transformPCPose(templateModel, results[minIdx]->pose);
        Mat result_AToB = transformPCPose(result_LocalAToB, getTranslation(center));
        writePLY(result_AToB, path.c_str());
    }

    std::cout << "[Error Cnt] " << errorCnt << "," << files.size() - 1 << std::endl;
}

int main(int argc, char** argv)
{
    std::string in_dir = DATA_PATH "models/0_0.05";
    std::string out_dir = in_dir + "_ppf";

    if (std::filesystem::exists(out_dir))
    {
        std::filesystem::remove_all(out_dir);
    }
    std::filesystem::create_directories(out_dir);

    {
        Timing t("[Total time] " + in_dir);
        process(in_dir, out_dir);
    }

    return 0;
}
