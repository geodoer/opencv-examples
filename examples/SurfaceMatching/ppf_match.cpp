//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2014, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
// Author: Tolga Birdal <tbirdal AT gmail.com>

#include "opencv2/surface_matching.hpp"
#include <iostream>
#include "opencv2/surface_matching/ppf_helpers.hpp"
#include "opencv2/core/utility.hpp"

using namespace std;
using namespace cv;
using namespace ppf_match_3d;

static void help(const string& errorMessage)
{
    cout << "Program init error : " << errorMessage << endl;
    cout << "\nUsage : ppf_matching [input model file] [input scene file]" << endl;
    cout << "\nPlease start again with new parameters" << endl;
}

int main(int argc, char** argv)
{
    // welcome message
    cout << "****************************************************" << endl;
    cout << "* Surface Matching demonstration : demonstrates the use of surface matching"
        " using point pair features." << endl;
    cout << "* The sample loads a model and a scene, where the model lies in a different"
        " pose than the training.\n* It then trains the model and searches for it in the"
        " input scene. The detected poses are further refined by ICP\n* and printed to the "
        " standard output." << endl;
    cout << "****************************************************" << endl;

    string modelFileName;
    string sceneFileName;

    if (argc < 3)
    {
        //help("Not enough input arguments");
        //exit(1);

#if 0
        //[测试1] 模型 + 模型，进行匹配
        modelFileName = DATA_PATH "ppf_icp_test/model0.05.ply";
        sceneFileName = DATA_PATH "ppf_icp_test/SHTL_SNB_CT_11_0.05.ply";
#else
        //[测试2] 模型 + 模型，进行匹配
        modelFileName   = DATA_PATH "ppf_icp_test_2model/74.obj.0.050000.ply";
        sceneFileName   = DATA_PATH "ppf_icp_test_2model/73.obj.0.050000.ply";
#endif
    }
    else
    {
        modelFileName = (string)argv[1];
        sceneFileName = (string)argv[2];
    }

#if (defined __x86_64__ || defined _M_X64)
    cout << "Running on 64 bits" << endl;
#else
    cout << "Running on 32 bits" << endl;
#endif

#ifdef _OPENMP
    cout << "Running with OpenMP" << endl;
#else
    cout << "Running without OpenMP and without TBB" << endl;
#endif

    Mat pc = loadPLYSimple(modelFileName.c_str(), 1);

    // 训练模型
    // Now train the model
    cout << "Training..." << endl;
    int64 tick1 = cv::getTickCount();
    ppf_match_3d::PPF3DDetector detector;
    detector.trainModel(pc);
    int64 tick2 = cv::getTickCount();
    cout << endl << "Training complete in "
        << (double)(tick2 - tick1) / cv::getTickFrequency()
        << " sec" << endl << "Loading model..." << endl;

    // 读取场景，并在场景中匹配，返回N个位姿
    // Read the scene
    Mat pcTest = loadPLYSimple(sceneFileName.c_str(), 1);

    // Match the model to the scene and get the pose
    cout << endl << "Starting matching..." << endl;
    vector<Pose3DPtr> results;
    tick1 = cv::getTickCount();
    detector.match(pcTest, results, 1.0 / 40.0, 0.05);
    tick2 = cv::getTickCount();
    cout << endl << "PPF Elapsed Time " <<
        (tick2 - tick1) / cv::getTickFrequency() << " sec" << endl;

    //check results size from match call above
    size_t results_size = results.size();
    cout << "Number of matching poses: " << results_size;
    if (results_size == 0) {
        cout << endl << "No matching poses found. Exiting." << endl;
        exit(0);
    }

    //output 
	for (size_t i = 0; i < results.size(); i++)
	{
		Pose3DPtr result = results[i];
		cout << "Pose Result " << i << endl;
		result->printPose();

		if (result->residual > 1)
		{
			continue;
		}

		auto path = modelFileName + ".ppf_" + std::to_string(i) + ".ply";
		Mat pct = transformPCPose(pc, result->pose);
		writePLY(pct, path.c_str());
	}

    // 用PPF的位姿作为初始矩阵，使用ICP进行精配准
    // Create an instance of ICP
    ICP icp(100, 0.005f, 2.5f, 8);
    int64 t1 = cv::getTickCount();

    // Register for all selected poses
    icp.registerModelToScene(pc, pcTest, results);
    int64 t2 = cv::getTickCount();

    cout << endl << "ICP Elapsed Time " <<
        (t2 - t1) / cv::getTickFrequency() << " sec" << endl;

    cout << "Poses: " << endl;
    // debug first five poses
    //全部输出。但residual太大的不输出
    for (size_t i = 0; i < results.size(); i++)
    {
        Pose3DPtr result = results[i];

        cout << "Pose Result " << i << endl;
        result->printPose();

        if (result->residual > 1)
        {
            continue; //误差太大的不输出
        }

        auto path = modelFileName + ".icp_" + std::to_string(i) + ".ply";
        Mat pct = transformPCPose(pc, result->pose);
        writePLY(pct, path.c_str());
    }
    //只输出最佳的
    {
        int minIdx = -1;
        double minValue = std::numeric_limits<double>::max();

        for (int i{ 0 }, size = results.size(); i < size; ++i)
        {
            const auto& result = results[i];
            const auto& residual = result->residual;

            if (residual > minValue)
            {
                continue;
            }

            minIdx = i;
            minValue = residual;
        }

        auto path = modelFileName + ".icp_" + std::to_string(minIdx) + "_optimal.ply";
        Mat pct = transformPCPose(pc, results[minIdx]->pose);
        writePLY(pct, path.c_str());
    }

    return 0;

}
