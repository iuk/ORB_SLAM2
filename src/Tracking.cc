/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Tracking.h"

#include <unistd.h>

#include <iostream>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Converter.h"
#include "FrameDrawer.h"
#include "Initializer.h"
#include "Map.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "PnPsolver.h"

using namespace std;

namespace ORB_SLAM2 {

Tracking::Tracking(System *pSys, ORBVocabulary *pVoc,
                   FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer,
                   Map *pMap, KeyFrameDatabase *pKFDB,
                   const string &strSettingPath, const int sensor)
    : mState(NO_IMAGES_YET),
      mSensor(sensor),
      mbOnlyTracking(false),
      mbVO(false),
      mpORBVocabulary(pVoc),
      mpKeyFrameDB(pKFDB),
      mpInitializer(static_cast<Initializer *>(NULL)),
      mpSystem(pSys),
      mpViewer(NULL),
      mpFrameDrawer(pFrameDrawer),
      mpMapDrawer(pMapDrawer),
      mpMap(pMap),
      mnLastRelocFrameId(0) {
  // Load camera parameters from settings file

  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
  float fx = fSettings["Camera.fx"];
  float fy = fSettings["Camera.fy"];
  float cx = fSettings["Camera.cx"];
  float cy = fSettings["Camera.cy"];

  cv::Mat K         = cv::Mat::eye(3, 3, CV_32F);
  K.at<float>(0, 0) = fx;
  K.at<float>(1, 1) = fy;
  K.at<float>(0, 2) = cx;
  K.at<float>(1, 2) = cy;
  K.copyTo(mK);

  cv::Mat DistCoef(4, 1, CV_32F);
  DistCoef.at<float>(0) = fSettings["Camera.k1"];
  DistCoef.at<float>(1) = fSettings["Camera.k2"];
  DistCoef.at<float>(2) = fSettings["Camera.p1"];
  DistCoef.at<float>(3) = fSettings["Camera.p2"];
  const float k3        = fSettings["Camera.k3"];
  if (k3 != 0) {
    DistCoef.resize(5);
    DistCoef.at<float>(4) = k3;
  }
  DistCoef.copyTo(mDistCoef);

  mbf = fSettings["Camera.bf"];

  float fps = fSettings["Camera.fps"];
  if (fps == 0)
    fps = 30;

  // Max/Min Frames to insert keyframes and to check relocalisation
  mMinFrames = 0;
  mMaxFrames = fps;

  cout << endl
       << "Camera Parameters: " << endl;
  cout << "- fx: " << fx << endl;
  cout << "- fy: " << fy << endl;
  cout << "- cx: " << cx << endl;
  cout << "- cy: " << cy << endl;
  cout << "- k1: " << DistCoef.at<float>(0) << endl;
  cout << "- k2: " << DistCoef.at<float>(1) << endl;
  if (DistCoef.rows == 5)
    cout << "- k3: " << DistCoef.at<float>(4) << endl;
  cout << "- p1: " << DistCoef.at<float>(2) << endl;
  cout << "- p2: " << DistCoef.at<float>(3) << endl;
  cout << "- fps: " << fps << endl;

  int nRGB = fSettings["Camera.RGB"];
  mbRGB    = nRGB;

  if (mbRGB)
    cout << "- color order: RGB (ignored if grayscale)" << endl;
  else
    cout << "- color order: BGR (ignored if grayscale)" << endl;

  // Load ORB parameters

  int nFeatures      = fSettings["ORBextractor.nFeatures"];
  float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
  int nLevels        = fSettings["ORBextractor.nLevels"];
  int fIniThFAST     = fSettings["ORBextractor.iniThFAST"];
  int fMinThFAST     = fSettings["ORBextractor.minThFAST"];

  mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

  if (sensor == System::STEREO)
    mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

  if (sensor == System::MONOCULAR)
    mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

  cout << endl
       << "ORB Extractor Parameters: " << endl;
  cout << "- Number of Features: " << nFeatures << endl;
  cout << "- Scale Levels: " << nLevels << endl;
  cout << "- Scale Factor: " << fScaleFactor << endl;
  cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
  cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

  if (sensor == System::STEREO || sensor == System::RGBD) {
    mThDepth = mbf * (float)fSettings["ThDepth"] / fx;
    cout << endl
         << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
  }

  if (sensor == System::RGBD) {
    mDepthMapFactor = fSettings["DepthMapFactor"];
    if (fabs(mDepthMapFactor) < 1e-5)
      mDepthMapFactor = 1;
    else
      mDepthMapFactor = 1.0f / mDepthMapFactor;
  }
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper) {
  mpLocalMapper = pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing) {
  mpLoopClosing = pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer) {
  mpViewer = pViewer;
}

cv::Mat
Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp) {
  mImGray             = imRectLeft;
  cv::Mat imGrayRight = imRectRight;

  if (mImGray.channels() == 3) {
    if (mbRGB) {
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
    } else {
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
    }
  } else if (mImGray.channels() == 4) {
    if (mbRGB) {
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
    } else {
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
    }
  }

  mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

  Track();

  return mCurrentFrame.mTcw.clone();
}

cv::Mat
Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp) {
  mImGray         = imRGB;
  cv::Mat imDepth = imD;

  if (mImGray.channels() == 3) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
  } else if (mImGray.channels() == 4) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
  }

  if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
    imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

  mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

  Track();

  return mCurrentFrame.mTcw.clone();
}

cv::Mat
Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp) {
  // 首先 转成 gray 图
  mImGray = im;

  if (mImGray.channels() == 3) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
  } else if (mImGray.channels() == 4) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
  }

  // 第一步：创建帧数据
  // 1. 提关键点 2. 去畸变 3. 存到64*48的格子中 4. 保存到 mCurrentFrame 中
  if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)  // 第一帧数据
    mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor,
                          mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
  else  // 其它数据
    mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft,
                          mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

  // 第二步：跟踪
  Track();

  // 返回当前帧位姿
  return mCurrentFrame.mTcw.clone();
}

void Tracking::Track() {
  if (mState == NO_IMAGES_YET) {
    mState = NOT_INITIALIZED;
  }

  mLastProcessedState = mState;

  // Get Map Mutex -> Map cannot be changed
  unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

  // 步骤1：初始化
  if (mState == NOT_INITIALIZED) {
    if (mSensor == System::STEREO || mSensor == System::RGBD)
      StereoInitialization();
    else
      MonocularInitialization();

    mpFrameDrawer->Update(this);

    if (mState != OK)
      return;
  }
  // 步骤2：跟踪
  // System is initialized. Track Frame.
  else {
    // bOK为临时变量，用于表示每个函数是否执行成功
    bool bOK;
    // Initial camera pose estimation using motion model
    // or relocalization (if tracking is lost)
    // 在viewer中有个开关 menuLocalizationMode，有它控制是否 ActivateLocalizationMode，
    // 并最终管控mbOnlyTracking
    // mbOnlyTracking等于false表示正常VO模式（有地图更新），
    // mbOnlyTracking等于true表示用户手动选择定位模式
    if (!mbOnlyTracking)  // 正常模式，有地图更新
    {
      // Local Mapping is activated. This is the normal behaviour, unless
      // you explicitly activate the "only tracking" mode.
      // 如果处于正常状态
      if (mState == OK) {
        // Local Mapping might have changed some MapPoints tracked in last frame
        // 检查并更新LastFrame被替换的MapPoints
        // 替换操作发生在 Fuse函数和SearchAndFuse函数
        CheckReplacedInLastFrame();

        // 步骤2.1：跟踪上一帧或者参考帧或者重定位
        // 运动模型是空的或刚完成重定位
        // mCurrentFrame.mnId<mnLastRelocFrameId+2这个判断不应该有
        // mnLastRelocFrameId上一次重定位的那一帧
        if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
          // 将上一帧的位姿作为当前帧的初始位姿
          // 通过BoW的方式在参考帧中找当前帧特征点的匹配点
          // 优化每个特征点都对应3D点重投影误差即可得到位姿
          bOK = TrackReferenceKeyFrame();
        }
        // mVelocity不为空，选择TrackWithMotionModel
        else {
          // 根据恒速模型设定当前帧的初始位姿
          // 通过投影的方式在参考帧中找当前帧特征点的匹配点
          // 优化每个特征点所对应3D点的投影误差即可得到位姿
          bOK = TrackWithMotionModel();
          if (!bOK)
            // TrackReferenceKeyFrame是跟踪参考帧，不能根据固定运动速度模型预测当前帧的位姿态，通过bow加速匹配（SearchByBow）
            // 最后通过优化得到优化后的位姿
            bOK = TrackReferenceKeyFrame();
        }
      }
      // 如果处于跟丢状态
      else {
        // BOW搜索，PnP求解位姿
        bOK = Relocalization();
      }
    }
    // 只进行跟踪tracking，局部地图不工作
    else {
      // Localization Mode: Local Mapping is deactivated
      // 步骤2.1：跟踪上一帧或者参考帧或者重定位
      // tracking跟丢了
      if (mState == LOST) {
        bOK = Relocalization();
      } 
      // 如果没跟丢
      else {
        if (!mbVO) {
          // In last frame we tracked enough MapPoints in the map

          if (!mVelocity.empty()) {
            bOK = TrackWithMotionModel();
          } else {
            bOK = TrackReferenceKeyFrame();
          }
        } else {
          // In last frame we tracked mainly "visual odometry" points.

          // We compute two camera poses, one from motion model and one doing relocalization.
          // If relocalization is sucessfull we choose that solution, otherwise we retain
          // the "visual odometry" solution.

          bool bOKMM    = false;
          bool bOKReloc = false;
          vector<MapPoint *> vpMPsMM;
          vector<bool> vbOutMM;
          cv::Mat TcwMM;
          if (!mVelocity.empty()) {
            bOKMM   = TrackWithMotionModel();
            vpMPsMM = mCurrentFrame.mvpMapPoints;
            vbOutMM = mCurrentFrame.mvbOutlier;
            TcwMM   = mCurrentFrame.mTcw.clone();
          }
          bOKReloc = Relocalization();

          if (bOKMM && !bOKReloc) {
            mCurrentFrame.SetPose(TcwMM);
            mCurrentFrame.mvpMapPoints = vpMPsMM;
            mCurrentFrame.mvbOutlier   = vbOutMM;

            if (mbVO) {
              for (int i = 0; i < mCurrentFrame.N; i++) {
                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i]) {
                  mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                }
              }
            }
          } else if (bOKReloc) {
            mbVO = false;
          }

          bOK = bOKReloc || bOKMM;
        }
      }
    }

    // 将最新的关键帧作为reference frame
    mCurrentFrame.mpReferenceKF = mpReferenceKF;

    // If we have an initial estimation of the camera pose and matching. Track the local map.
    // 步骤2.2：在帧间匹配得到初始的姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
    // local map:当前帧、当前帧的MapPoints、当前关键帧与其它关键帧共视关系
    // 在步骤2.1中主要是两两跟踪（恒速模型跟踪上一帧、跟踪参考帧），
    // 这里搜索局部关键帧后搜集所有局部MapPoints，
    // 然后将局部MapPoints和当前帧进行投影匹配，得到更多匹配的MapPoints后进行Pose优化
    // 如果即定位又建图
    if (!mbOnlyTracking) {
      if (bOK)
        bOK = TrackLocalMap();
    }
    // 如果只定位
    else {
      // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
      // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
      // the camera we will use the local map again.
      // 重定位成功
      if (bOK && !mbVO)
        bOK = TrackLocalMap();
    }

    if (bOK)
      mState = OK;
    else
      mState = LOST;

    // Update drawer
    mpFrameDrawer->Update(this);

    // If tracking were good, check if we insert a keyframe
    if (bOK) {
      // Update motion model
      if (!mLastFrame.mTcw.empty()) {
        cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
        mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
        mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
        mVelocity = mCurrentFrame.mTcw * LastTwc;
      } else
        mVelocity = cv::Mat();

      mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

      // Clean VO matches
      for (int i = 0; i < mCurrentFrame.N; i++) {
        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
        if (pMP)
          if (pMP->Observations() < 1) {
            mCurrentFrame.mvbOutlier[i]   = false;
            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
          }
      }

      // Delete temporal MapPoints
      for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end();
           lit != lend; lit++) {
        MapPoint *pMP = *lit;
        delete pMP;
      }
      mlpTemporalPoints.clear();

      // Check if we need to insert a new keyframe
      if (NeedNewKeyFrame())
        CreateNewKeyFrame();

      // We allow points with high innovation (considererd outliers by the Huber Function)
      // pass to the new keyframe, so that bundle adjustment will finally decide
      // if they are outliers or not. We don't want next frame to estimate its position
      // with those points so we discard them in the frame.
      for (int i = 0; i < mCurrentFrame.N; i++) {
        if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
          mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
      }
    }

    // Reset if the camera get lost soon after initialization
    if (mState == LOST) {
      if (mpMap->KeyFramesInMap() <= 5) {
        cout << "Track lost soon after initialisation, reseting..." << endl;
        mpSystem->Reset();
        return;
      }
    }

    if (!mCurrentFrame.mpReferenceKF)
      mCurrentFrame.mpReferenceKF = mpReferenceKF;

    mLastFrame = Frame(mCurrentFrame);
  }

  // Store frame pose information to retrieve the complete camera trajectory afterwards.
  if (!mCurrentFrame.mTcw.empty()) {
    cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
    mlRelativeFramePoses.push_back(Tcr);
    mlpReferences.push_back(mpReferenceKF);
    mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
    mlbLost.push_back(mState == LOST);
  } else {
    // This can happen if tracking is lost
    mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
    mlpReferences.push_back(mlpReferences.back());
    mlFrameTimes.push_back(mlFrameTimes.back());
    mlbLost.push_back(mState == LOST);
  }
}

void Tracking::StereoInitialization() {
  if (mCurrentFrame.N > 500) {
    // Set Frame pose to the origin
    mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

    // Create KeyFrame
    KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    // Insert KeyFrame in the map
    mpMap->AddKeyFrame(pKFini);

    // Create MapPoints and asscoiate to KeyFrame
    for (int i = 0; i < mCurrentFrame.N; i++) {
      float z = mCurrentFrame.mvDepth[i];
      if (z > 0) {
        cv::Mat x3D      = mCurrentFrame.UnprojectStereo(i);
        MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap);
        pNewMP->AddObservation(pKFini, i);
        pKFini->AddMapPoint(pNewMP, i);
        pNewMP->ComputeDistinctiveDescriptors();
        pNewMP->UpdateNormalAndDepth();
        mpMap->AddMapPoint(pNewMP);

        mCurrentFrame.mvpMapPoints[i] = pNewMP;
      }
    }

    cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

    mpLocalMapper->InsertKeyFrame(pKFini);

    mLastFrame       = Frame(mCurrentFrame);
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame   = pKFini;

    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints           = mpMap->GetAllMapPoints();
    mpReferenceKF               = pKFini;
    mCurrentFrame.mpReferenceKF = pKFini;

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

    mState = OK;
  }
}

/**
 * @brief 单目的地图初始化
 *
 * 并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
 * 得到初始两帧的匹配、相对运动、初始MapPoints
 */
void Tracking::MonocularInitialization() {
  if (!mpInitializer)  // 如果没有设置初始化器
  {
    // Set Reference Frame
    // 如果当前帧的关键点 > 100
    if (mCurrentFrame.mvKeys.size() > 100) {
      // 步骤1：得到用于初始化的第一帧，初始化需要两帧
      mInitialFrame = Frame(mCurrentFrame);  // 保存初始帧
      // 记录最近的一帧
      mLastFrame = Frame(mCurrentFrame);
      // mvbPrevMatched最大的情况就是所有特征点都被跟踪上
      // 保存初始化第一帧的关键点坐标
      mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
      for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
        mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

      if (mpInitializer)
        delete mpInitializer;

      // 构造函数中，设置了初始化器的参考帧、标准差、最大迭代数
      mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

      return;
    }
    // 如果当前帧的关键点 < 100，啥也不干，等>100 再干
  }

  else  // Try to initialize
  {
    // 步骤2：如果当前帧特征点数大于100，则得到用于单目初始化的第二帧
    // 如果当前帧特征点太少，重新构造初始器
    // 因此只有连续两帧的特征点个数都大于100时，才能继续进行初始化过程
    if ((int)mCurrentFrame.mvKeys.size() <= 100) {
      delete mpInitializer;
      mpInitializer = static_cast<Initializer *>(NULL);
      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
      return;
    }

    // Find correspondences
    // 步骤3：在mInitialFrame与mCurrentFrame中找匹配的特征点对
    // mvbPrevMatched为前一帧的特征点，存储了mInitialFrame中哪些点将进行接下来的匹配
    // mvIniMatches存储mInitialFrame,mCurrentFrame之间匹配的特征点
    // @param1 nnratio  ratio of the best and the second score
    // @param2 checkOri check orientation
    ORBmatcher matcher(0.9, true);
    int nmatches =  // 输出 对应上的点数
        matcher.SearchForInitialization(
            mInitialFrame,   // 初始化第一帧
            mCurrentFrame,   // 初始化第二帧
            mvbPrevMatched,  // 初始化第一帧特征点
            mvIniMatches,    // 输出：初始化第一帧特征点对应第二帧特征点index
            100);            // 搜索窗口

    // Check if there are enough correspondences
    // 如果两帧之间没那么多匹配，则本次初始化失败
    if (nmatches < 100) {
      delete mpInitializer;
      mpInitializer = static_cast<Initializer *>(NULL);
      return;
    }

    cv::Mat Rcw;                  // Current Camera Rotation
    cv::Mat tcw;                  // Current Camera Translation
    vector<bool> vbTriangulated;  // Triangulated Correspondences (mvIniMatches)

    // 步骤5：如果两帧匹配成功。通过H模型或F模型进行单目初始化，
    // 得到两帧间相对运动：R t
    // 初始 MapPoints：以第一帧相机坐标系为世界坐标系的 三维坐标
    if (mpInitializer->Initialize(
            mCurrentFrame,    // 初始化第二帧
            mvIniMatches,     // 输入 1-2帧匹配keypoint的id, 位置是 第1帧的，id 是第二帧的
            Rcw,              // 输出 两帧间相机旋转
            tcw,              // 输出 两帧间相机平移
            mvIniP3D,         // 输出 特征点的 三维点坐标
            vbTriangulated))  // 输出 点对是否可以三角化
    {
      // 步骤6：删除那些无法进行三角化的匹配点
      for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
        if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
          mvIniMatches[i] = -1;
          nmatches--;  // 1-2 对应上的特征点数 --
        }
      }

      // Set Frame Poses
      // 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
      mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

      // 由Rcw和tcw构造Tcw,并赋值给mTcw，mTcw为世界坐标系到该帧的变换矩阵
      cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
      Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
      tcw.copyTo(Tcw.rowRange(0, 3).col(3));
      // 设置第二帧坐标
      mCurrentFrame.SetPose(Tcw);

      // 步骤6：将三角化得到的3D点包装成MapPoints
      // Initialize函数会得到mvIniP3D，
      // mvIniP3D是cv::Point3f类型的一个容器，是个存放3D点的临时变量，
      // CreateInitialMapMonocular将3D点包装成MapPoint类型存入KeyFrame和Map中
      CreateInitialMapMonocular();
    }
  }
}

/**
 * @brief CreateInitialMapMonocular
 *
 * 为单目摄像头三角化生成MapPoints
 */
void Tracking::CreateInitialMapMonocular() {
  // Create KeyFrames
  // 构造关键帧，并关联 地图 和 关键帧数据库
  KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
  KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

  // 步骤1：将初始关键帧的描述子转为BoW
  pKFini->ComputeBoW();
  // 步骤2：将当前关键帧的描述子转为BoW
  pKFcur->ComputeBoW();

  // Insert KFs in the map
  // 步骤3：将关键帧插入到地图
  // 凡是关键帧，都要插入地图
  mpMap->AddKeyFrame(pKFini);
  mpMap->AddKeyFrame(pKFcur);

  // Create MapPoints and asscoiate to keyframes
  // 步骤4：将3D点包装成MapPoints
  for (size_t i = 0; i < mvIniMatches.size(); i++) {
    // =-1的为未找到点对
    if (mvIniMatches[i] < 0)
      continue;

    //Create MapPoint.
    // 通过1-2帧点对算出的三维点坐标
    cv::Mat worldPos(mvIniP3D[i]);

    // 步骤4.1：用3D点构造MapPoint
    MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

    // 步骤4.2：为该MapPoint添加属性：
    // a.观测到该MapPoint的关键帧
    // b.该MapPoint的描述子
    // c.该MapPoint的平均观测方向和深度范围

    // a.表示该MapPoint可以被哪个KeyFrame的哪个特征点观测到
    pMP->AddObservation(pKFini, i);
    pMP->AddObservation(pKFcur, mvIniMatches[i]);

    // b.从众多观测到该MapPoint的特征点中挑选区分读最高的描述子
    // Distinctive 出色的
    pMP->ComputeDistinctiveDescriptors();
    // c.更新该MapPoint平均观测方向以及观测距离的范围
    pMP->UpdateNormalAndDepth();

    // 步骤4.3：表示该KeyFrame的哪个特征点可以观测到哪个3D点
    pKFini->AddMapPoint(pMP, i);
    pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

    //Fill Current Frame structure
    mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
    mCurrentFrame.mvbOutlier[mvIniMatches[i]]   = false;

    //Add to Map
    // 步骤4.4：在地图中添加该MapPoint
    mpMap->AddMapPoint(pMP);
  }

  // Update Connections
  // 步骤5：更新关键帧间的连接关系，对于一个新创建的关键帧都会执行一次关键连接关系更新
  // 在3D点和关键帧之间建立边，每个边有一个权重，边的权重是该关键帧与当前帧公共3D点的个数
  pKFini->UpdateConnections();
  pKFcur->UpdateConnections();

  // Bundle Adjustment
  cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

  // 步骤5：BA优化
  // 经过 BA 优化，map 中的 mappoint 和 keyframe 的位姿都将被更新
  Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

  // Set median depth to 1
  // 关键帧的 中值深度
  // 步骤6：!!!将MapPoints的中值深度归一化到1，并归一化两帧之间变换
  // 单目传感器无法恢复真实的深度，这里将点云中值深度（欧式距离，不是指z）归一化到1
  // 评估关键帧场景深度，q=2表示中值
  // 计算mappoint 在初始第一帧中的深度中值 就是 mappoint 在世界系中的深度(z) 中值
  // 注：初始化第一帧同时为世界坐标系
  float medianDepth    = pKFini->ComputeSceneMedianDepth(2);
  float invMedianDepth = 1.0f / medianDepth;

  // 失败的初始化
  if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100) {
    cout << "Wrong initialization, reseting..." << endl;
    Reset();
    return;
  }

  // Scale points
  // 把3D点的尺度也归一化到1
  // 对 所有 mappoint 点缩放。缩放的结果是  其z的中值 = 1M
  vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
  for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
    if (vpAllMapPoints[iMP]) {
      MapPoint *pMP = vpAllMapPoints[iMP];
      pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
    }
  }

  // Scale initial baseline
  // 根据点云归一化比例缩放平移量
  // 对1-2 帧的平移量进行缩放
  cv::Mat Tc2w               = pKFcur->GetPose();
  Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
  pKFcur->SetPose(Tc2w);

  // 这部分和SteroInitialization()相似
  // localMapper 本体在  system::system 中构造：
  // mpLocalMapper   = new LocalMapping(mpMap, mSensor == MONOCULAR);
  // InsertKeyFrame() 就是 push 到一个 std::list 中
  mpLocalMapper->InsertKeyFrame(pKFini);
  mpLocalMapper->InsertKeyFrame(pKFcur);

  mCurrentFrame.SetPose(pKFcur->GetPose());
  mnLastKeyFrameId = mCurrentFrame.mnId;
  mpLastKeyFrame   = pKFcur;

  mvpLocalKeyFrames.push_back(pKFcur);
  mvpLocalKeyFrames.push_back(pKFini);
  mvpLocalMapPoints           = mpMap->GetAllMapPoints();
  mpReferenceKF               = pKFcur;
  mCurrentFrame.mpReferenceKF = pKFcur;

  mLastFrame = Frame(mCurrentFrame);

  mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

  mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

  mpMap->mvpKeyFrameOrigins.push_back(pKFini);

  mState = OK;
  // 初始化成功，至此，初始化过程完成
}

/**
 * @brief 检查上一帧中的 MapPoints 是否被替换
 * keyframe在local_mapping和loopclosure中存在fuse mappoint。
 * 由于这些mappoint被改变了，且只更新了关键帧的mappoint，对于mLastFrame普通帧，也要检查并更新mappoint
 * @see LocalMapping::SearchInNeighbors()
 */
void Tracking::CheckReplacedInLastFrame() {
  for (int i = 0; i < mLastFrame.N; i++) {
    MapPoint *pMP = mLastFrame.mvpMapPoints[i];

    if (pMP) {
      MapPoint *pRep = pMP->GetReplaced();
      // 如果被标记为被替换为新的 mappoing，则重新指向新的 mappoing
      if (pRep) {
        mLastFrame.mvpMapPoints[i] = pRep;
      }
    }
  }
}

/**
 * @brief 对参考关键帧的MapPoints进行跟踪
 * 
 * 1. 计算当前帧的词包，将当前帧的特征点分到特定层的nodes上
 * 2. 对属于同一node的描述子进行匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return 如果通过重投影误差检测的匹配数大于10，返回true
 */
bool Tracking::TrackReferenceKeyFrame() {
  // Compute Bag of Words vector
  // 步骤1：将当前帧的描述子转化为BoW向量
  mCurrentFrame.ComputeBoW();

  // We perform first an ORB matching with the reference keyframe
  // If enough matches are found we setup a PnP solver
  ORBmatcher matcher(0.7, true);
  vector<MapPoint *> vpMapPointMatches;

  // 步骤2：通过特征点的BoW加快当前帧与参考帧之间的特征点匹配
  // 特征点的匹配关系由MapPoints进行维护
  int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

  if (nmatches < 15)
    return false;

  mCurrentFrame.mvpMapPoints = vpMapPointMatches;
  // 步骤3:将上一帧的位姿态作为当前帧位姿的初始值
  mCurrentFrame.SetPose(mLastFrame.mTcw);

  // 步骤4:通过优化3D-2D的重投影误差来获得位姿
  Optimizer::PoseOptimization(&mCurrentFrame);

  // Discard outliers
  // 步骤5：剔除优化后的outlier匹配点（MapPoints）
  int nmatchesMap = 0;
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (mCurrentFrame.mvbOutlier[i]) {
        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
        mCurrentFrame.mvbOutlier[i]   = false;
        pMP->mbTrackInView            = false;
        pMP->mnLastFrameSeen          = mCurrentFrame.mnId;
        nmatches--;
      } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
        nmatchesMap++;
    }
  }

  return nmatchesMap >= 10;
}

/**
 * @brief 双目或rgbd摄像头根据深度值为上一帧产生新的MapPoints
 *
 * 在双目和rgbd情况下，选取一些深度小一些的点（可靠一些） \n
 * 可以通过深度值产生一些新的MapPoints
 */
void Tracking::UpdateLastFrame() {
  // Update pose according to reference keyframe
  // 步骤1：更新最近一帧的位姿
  KeyFrame *pRef = mLastFrame.mpReferenceKF;
  cv::Mat Tlr    = mlRelativeFramePoses.back();

  mLastFrame.SetPose(Tlr * pRef->GetPose());
  // 如果上一帧为关键帧，或者单目的情况，则退出
  if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || !mbOnlyTracking)
    return;

  // 步骤2：对于双目或rgbd摄像头，为上一帧临时生成新的MapPoints
  // 注意这些MapPoints不加入到Map中，在tracking的最后会删除
  // 跟踪过程中需要将将上一帧的MapPoints投影到当前帧可以缩小匹配范围，加快当前帧与上一帧进行特征点匹配

  // Create "visual odometry" MapPoints
  // We sort points according to their measured depth by the stereo/RGB-D sensor
  // 步骤2.1：得到上一帧有深度值的特征点
  vector<pair<float, int> > vDepthIdx;
  vDepthIdx.reserve(mLastFrame.N);
  for (int i = 0; i < mLastFrame.N; i++) {
    float z = mLastFrame.mvDepth[i];
    if (z > 0) {
      vDepthIdx.push_back(make_pair(z, i));
    }
  }

  if (vDepthIdx.empty())
    return;
  // 步骤2.2：按照深度从小到大排序
  sort(vDepthIdx.begin(), vDepthIdx.end());

  // We insert all close points (depth<mThDepth)
  // If less than 100 close points, we insert the 100 closest ones.
  // 步骤2.3：将距离比较近的点包装成MapPoints
  int nPoints = 0;
  for (size_t j = 0; j < vDepthIdx.size(); j++) {
    int i = vDepthIdx[j].second;

    bool bCreateNew = false;

    MapPoint *pMP = mLastFrame.mvpMapPoints[i];
    if (!pMP)
      bCreateNew = true;
    else if (pMP->Observations() < 1) {
      bCreateNew = true;
    }

    if (bCreateNew) {
      // 这些生成MapPoints后并没有通过：
      // a.AddMapPoint、
      // b.AddObservation、
      // c.ComputeDistinctiveDescriptors、
      // d.UpdateNormalAndDepth添加属性，
      // 这些MapPoint仅仅为了提高双目和RGBD的跟踪成功率
      cv::Mat x3D      = mLastFrame.UnprojectStereo(i);
      MapPoint *pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

      mLastFrame.mvpMapPoints[i] = pNewMP;
      // 标记为临时添加的MapPoint，之后在CreateNewKeyFrame之前会全部删除
      mlpTemporalPoints.push_back(pNewMP);
      nPoints++;
    } else {
      nPoints++;
    }

    if (vDepthIdx[j].first > mThDepth && nPoints > 100)
      break;
  }
}

/**
 * @brief 根据匀速度模型对上一帧的MapPoints进行跟踪
 * 
 * 1. 非单目情况，需要对上一帧产生一些新的MapPoints（临时）
 * 2. 将上一帧的MapPoints投影到当前帧的图像平面上，在投影的位置进行区域匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return 如果匹配数大于10，返回true
 * @see V-B Initial Pose Estimation From Previous Frame
 */
bool Tracking::TrackWithMotionModel() {
  ORBmatcher matcher(0.9, true);

  // Update last frame pose according to its reference keyframe
  // Create "visual odometry" points if in Localization Mode
  // 步骤1：对于双目或rgbd摄像头，根据深度值为上一关键帧生成新的MapPoints
  // （跟踪过程中需要将当前帧与上一帧进行特征点匹配，将上一帧的MapPoints投影到当前帧可以缩小匹配范围）
  // 在跟踪过程中，去除outlier的MapPoint，如果不及时增加MapPoint会逐渐减少
  // 这个函数的功能就是补充增加RGBD和双目相机上一帧的MapPoints数
  UpdateLastFrame();
  // 根据Const Velocity Model(认为这两帧之间的相对运动和之前两帧间相对运动相同)估计当前帧的位姿
  // mVelocity为最近一次前后帧位姿之差
  mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

  fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

  // Project points seen in previous frame
  int th;
  if (mSensor != System::STEREO)
    th = 15;
  else
    th = 7;

  // 步骤2：根据匀速度模型进行对上一帧的MapPoints进行跟踪
  // 根据上一帧特征点对应的3D点投影的位置缩小特征点匹配范围
  int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

  // If few matches, uses a wider window search
  // 如果跟踪的点少，则扩大搜索半径再来一次
  if (nmatches < 20) {
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
    nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR);
  }

  if (nmatches < 20)
    return false;

  // Optimize frame pose with all matches
  // 步骤3：优化位姿，only-pose BA优化
  Optimizer::PoseOptimization(&mCurrentFrame);

  // Discard outliers
  // 步骤4：优化位姿后剔除outlier的mvpMapPoints
  int nmatchesMap = 0;
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (mCurrentFrame.mvbOutlier[i]) {
        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
        mCurrentFrame.mvbOutlier[i]   = false;
        pMP->mbTrackInView            = false;
        pMP->mnLastFrameSeen          = mCurrentFrame.mnId;
        nmatches--;
      } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
        nmatchesMap++;
    }
  }

  if (mbOnlyTracking) {
    mbVO = nmatchesMap < 10;
    return nmatches > 20;
  }

  return nmatchesMap >= 10;
}

bool Tracking::TrackLocalMap() {
  // We have an estimation of the camera pose and some map points tracked in the frame.
  // We retrieve the local map and try to find matches to points in the local map.

  UpdateLocalMap();

  SearchLocalPoints();

  // Optimize Pose
  Optimizer::PoseOptimization(&mCurrentFrame);
  mnMatchesInliers = 0;

  // Update MapPoints Statistics
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (!mCurrentFrame.mvbOutlier[i]) {
        mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
        if (!mbOnlyTracking) {
          if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
            mnMatchesInliers++;
        } else
          mnMatchesInliers++;
      } else if (mSensor == System::STEREO)
        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
    }
  }

  // Decide if the tracking was succesful
  // More restrictive if there was a relocalization recently
  if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
    return false;

  if (mnMatchesInliers < 30)
    return false;
  else
    return true;
}

bool Tracking::NeedNewKeyFrame() {
  if (mbOnlyTracking)
    return false;

  // If Local Mapping is freezed by a Loop Closure do not insert keyframes
  if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
    return false;

  const int nKFs = mpMap->KeyFramesInMap();

  // Do not insert keyframes if not enough frames have passed from last relocalisation
  if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
    return false;

  // Tracked MapPoints in the reference keyframe
  int nMinObs = 3;
  if (nKFs <= 2)
    nMinObs = 2;
  int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

  // Local Mapping accept keyframes?
  bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

  // Check how many "close" points are being tracked and how many could be potentially created.
  int nNonTrackedClose = 0;
  int nTrackedClose    = 0;
  if (mSensor != System::MONOCULAR) {
    for (int i = 0; i < mCurrentFrame.N; i++) {
      if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth) {
        if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
          nTrackedClose++;
        else
          nNonTrackedClose++;
      }
    }
  }

  bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

  // Thresholds
  float thRefRatio = 0.75f;
  if (nKFs < 2)
    thRefRatio = 0.4f;

  if (mSensor == System::MONOCULAR)
    thRefRatio = 0.9f;

  // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
  const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
  // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
  const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
  //Condition 1c: tracking is weak
  const bool c1c = mSensor != System::MONOCULAR && (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
  // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
  const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) && mnMatchesInliers > 15);

  if ((c1a || c1b || c1c) && c2) {
    // If the mapping accepts keyframes, insert keyframe.
    // Otherwise send a signal to interrupt BA
    if (bLocalMappingIdle) {
      return true;
    } else {
      mpLocalMapper->InterruptBA();
      if (mSensor != System::MONOCULAR) {
        if (mpLocalMapper->KeyframesInQueue() < 3)
          return true;
        else
          return false;
      } else
        return false;
    }
  } else
    return false;
}

void Tracking::CreateNewKeyFrame() {
  if (!mpLocalMapper->SetNotStop(true))
    return;

  KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

  mpReferenceKF               = pKF;
  mCurrentFrame.mpReferenceKF = pKF;

  if (mSensor != System::MONOCULAR) {
    mCurrentFrame.UpdatePoseMatrices();

    // We sort points by the measured depth by the stereo/RGBD sensor.
    // We create all those MapPoints whose depth < mThDepth.
    // If there are less than 100 close points we create the 100 closest.
    vector<pair<float, int> > vDepthIdx;
    vDepthIdx.reserve(mCurrentFrame.N);
    for (int i = 0; i < mCurrentFrame.N; i++) {
      float z = mCurrentFrame.mvDepth[i];
      if (z > 0) {
        vDepthIdx.push_back(make_pair(z, i));
      }
    }

    if (!vDepthIdx.empty()) {
      sort(vDepthIdx.begin(), vDepthIdx.end());

      int nPoints = 0;
      for (size_t j = 0; j < vDepthIdx.size(); j++) {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
        if (!pMP)
          bCreateNew = true;
        else if (pMP->Observations() < 1) {
          bCreateNew                    = true;
          mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
        }

        if (bCreateNew) {
          cv::Mat x3D      = mCurrentFrame.UnprojectStereo(i);
          MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap);
          pNewMP->AddObservation(pKF, i);
          pKF->AddMapPoint(pNewMP, i);
          pNewMP->ComputeDistinctiveDescriptors();
          pNewMP->UpdateNormalAndDepth();
          mpMap->AddMapPoint(pNewMP);

          mCurrentFrame.mvpMapPoints[i] = pNewMP;
          nPoints++;
        } else {
          nPoints++;
        }

        if (vDepthIdx[j].first > mThDepth && nPoints > 100)
          break;
      }
    }
  }

  mpLocalMapper->InsertKeyFrame(pKF);

  mpLocalMapper->SetNotStop(false);

  mnLastKeyFrameId = mCurrentFrame.mnId;
  mpLastKeyFrame   = pKF;
}

/**
 * @brief 对Local MapPoints进行跟踪
 * 
 * 在局部地图中查找在当前帧视野范围内的点，将视野范围内的点和当前帧的特征点进行投影匹配
 */
void Tracking::SearchLocalPoints() {
  // Do not search map points already matched
  // 步骤1：遍历当前帧的mvpMapPoints，标记这些MapPoints不参与之后的搜索
  // 因为当前的mvpMapPoints一定在当前帧的视野中
  for (vector<MapPoint *>::iterator vit  = mCurrentFrame.mvpMapPoints.begin(),
                                    vend = mCurrentFrame.mvpMapPoints.end();
       vit != vend; vit++)
  // 遍历当前帧的 所有 mappoint
  {
    MapPoint *pMP = *vit;
    if (pMP) {
      if (pMP->isBad()) {
        *vit = static_cast<MapPoint *>(NULL);
      } else {
        // 更新能观测到该点的帧数加1
        pMP->IncreaseVisible();
        // 标记该点被当前帧观测到
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        // 标记该点将来不被投影，因为已经匹配过
        pMP->mbTrackInView = false;
      }
    }
  }

  int nToMatch = 0;

  // Project points in frame and check its visibility
  // 步骤2：将所有局部MapPoints投影到当前帧，判断是否在视野范围内，然后进行投影匹配
  for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end();
       vit != vend; vit++) {
    // 已经被当前帧观测到MapPoint不再判断是否能被当前帧观测到
    MapPoint *pMP = *vit;
    if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
      continue;
    if (pMP->isBad())
      continue;
    // Project (this fills MapPoint variables for matching)
    // 步骤2.1：判断LocalMapPoints中的点是否在在视野内
    if (mCurrentFrame.isInFrustum(pMP, 0.5)) {
      // 观测到该点的帧数加1，该MapPoint在某些帧的视野范围内
      pMP->IncreaseVisible();
      // 只有在视野范围内的MapPoints才参与之后的投影匹配
      nToMatch++;
    }
  }

  if (nToMatch > 0) {
    ORBmatcher matcher(0.8);
    int th = 1;
    if (mSensor == System::RGBD)
      th = 3;
    // If the camera has been relocalised recently, perform a coarser search
    // 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
    if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
      th = 5;
    // 步骤2.2：对视野范围内的MapPoints通过投影进行特征点匹配
    matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
  }
}

/**
 * @brief 更新LocalMap
 *
 * 局部地图包括： \n
 * - K1个关键帧、K2个临近关键帧和参考关键帧
 * - 由这些关键帧观测到的MapPoints
 */
void Tracking::UpdateLocalMap() {
  // This is for visualization
  // 这行程序放在UpdateLocalPoints函数后面是不是好一些
  mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

  // Update
  // 更新局部关键帧和局部MapPoints
  UpdateLocalKeyFrames();
  UpdateLocalPoints();
}

/**
 * @brief 更新局部关键点，called by UpdateLocalMap()
 * 
 * 局部关键帧mvpLocalKeyFrames的MapPoints，更新mvpLocalMapPoints
 */
void Tracking::UpdateLocalPoints() {
  // 步骤1：清空局部MapPoints
  mvpLocalMapPoints.clear();
  // 步骤2：遍历局部关键帧mvpLocalKeyFrames
  for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
       itKF != itEndKF; itKF++) {
    KeyFrame *pKF                  = *itKF;
    const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();
    // 步骤2：将局部关键帧的MapPoints添加到mvpLocalMapPoints
    for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end();
         itMP != itEndMP; itMP++) {
      MapPoint *pMP = *itMP;
      if (!pMP)
        continue;
      // mnTrackReferenceForFrame防止重复添加局部MapPoint
      if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
        continue;
      if (!pMP->isBad()) {
        mvpLocalMapPoints.push_back(pMP);
        pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
      }
    }
  }
}

/**
 * @brief 更新局部关键帧，called by UpdateLocalMap()
 *
 * 遍历当前帧的MapPoints，将观测到这些MapPoints的关键帧和相邻的关键帧取出，更新mvpLocalKeyFrames
 */
void Tracking::UpdateLocalKeyFrames() {
  // Each map point vote for the keyframes in which it has been observed
  // 步骤1：遍历当前帧的MapPoints，记录所有能观测到当前帧MapPoints的关键帧
  map<KeyFrame *, int> keyframeCounter;

  // 遍历当前帧的 keypoint
  for (int i = 0; i < mCurrentFrame.N; i++) {
    // 如果该 keypoint 有对应的 mappoint
    if (mCurrentFrame.mvpMapPoints[i]) {
      // 取出 该mappoint
      MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
      if (!pMP->isBad()) {
        // 取出能观察到该MapPoints的所有 keyframe
        const map<KeyFrame *, size_t> observations = pMP->GetObservations();
        // 遍历所有的 共视 keyframe
        for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end();
             it != itend; it++)
          // 加入到 keyframe 统计器 中
          keyframeCounter[it->first]++;
      }
      // 如果该 mappoint 是个坏点
      else {
        mCurrentFrame.mvpMapPoints[i] = NULL;
      }
    }
  }

  if (keyframeCounter.empty())
    return;

  int max          = 0;
  KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

  // 步骤2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧有三个策略
  // 先清空局部关键帧
  // 制作新的 localkeyframes(局部关键帧)
  mvpLocalKeyFrames.clear();
  mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

  // All keyframes that observe a map point are included in the local map.
  // Also check which keyframe shares most points
  // V-D K1: shares the map points with current frame
  // 策略1：能观测到当前帧MapPoints的关键帧作为局部关键帧
  // 遍历 keyframe 统计器
  for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end();
       it != itEnd; it++) {
    KeyFrame *pKF = it->first;

    if (pKF->isBad())
      continue;

    if (it->second > max) {
      max    = it->second;
      pKFmax = pKF;
    }

    mvpLocalKeyFrames.push_back(it->first);
    pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
  }

  // Include also some not-already-included keyframes that are
  // neighbors to already-included keyframes
  // V-D K2: neighbors to K1 in the covisibility graph
  // 策略2：与策略1得到的局部关键帧共视程度很高的关键帧作为局部关键帧
  for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
       itKF != itEndKF; itKF++) {
    // Limit the number of keyframes
    if (mvpLocalKeyFrames.size() > 80)
      break;

    KeyFrame *pKF = *itKF;

    // 策略2.1:最佳共视的10帧
    const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

    for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end();
         itNeighKF != itEndNeighKF; itNeighKF++) {
      KeyFrame *pNeighKF = *itNeighKF;
      if (!pNeighKF->isBad()) {
        if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
          mvpLocalKeyFrames.push_back(pNeighKF);
          pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
          break;
        }
      }
    }
    // 策略2.2:自己的子关键帧
    const set<KeyFrame *> spChilds = pKF->GetChilds();
    for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++) {
      KeyFrame *pChildKF = *sit;
      if (!pChildKF->isBad()) {
        if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
          mvpLocalKeyFrames.push_back(pChildKF);
          pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
          break;
        }
      }
    }

    // 策略2.3:自己的父关键帧
    KeyFrame *pParent = pKF->GetParent();
    if (pParent) {
      // mnTrackReferenceForFrame防止重复添加局部关键帧
      if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
        mvpLocalKeyFrames.push_back(pParent);
        pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        break;
      }
    }
  }

  // V-D Kref： shares the most map points with current frame
  // 步骤3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
  if (pKFmax) {
    mpReferenceKF               = pKFmax;
    mCurrentFrame.mpReferenceKF = mpReferenceKF;
  }
}

bool Tracking::Relocalization() {
  // Compute Bag of Words Vector
  // 步骤1：计算当前帧特征点的Bow映射
  mCurrentFrame.ComputeBoW();

  // Relocalization is performed when tracking is lost
  // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
  // 步骤2：找到与当前帧相似的候选关键帧
  vector<KeyFrame *> vpCandidateKFs =
      mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

  if (vpCandidateKFs.empty())
    return false;

  const int nKFs = vpCandidateKFs.size();

  // We perform first an ORB matching with each candidate
  // If enough matches are found we setup a PnP solver
  ORBmatcher matcher(0.75, true);

  vector<PnPsolver *> vpPnPsolvers;
  vpPnPsolvers.resize(nKFs);

  vector<vector<MapPoint *> > vvpMapPointMatches;
  vvpMapPointMatches.resize(nKFs);

  vector<bool> vbDiscarded;
  vbDiscarded.resize(nKFs);

  int nCandidates = 0;

  // 遍历所有候选关键帧
  for (int i = 0; i < nKFs; i++) {
    KeyFrame *pKF = vpCandidateKFs[i];
    if (pKF->isBad())
      vbDiscarded[i] = true;
    else {
      // 步骤3：通过BoW进行匹配
      int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
      if (nmatches < 15) {
        vbDiscarded[i] = true;
        continue;
      } else {
        PnPsolver *pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
        pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
        vpPnPsolvers[i] = pSolver;
        nCandidates++;
      }
    }
  }

  // Alternatively perform some iterations of P4P RANSAC
  // Until we found a camera pose supported by enough inliers
  bool bMatch = false;
  ORBmatcher matcher2(0.9, true);

  while (nCandidates > 0 && !bMatch) {
    for (int i = 0; i < nKFs; i++) {
      if (vbDiscarded[i])
        continue;

      // Perform 5 Ransac Iterations
      vector<bool> vbInliers;
      int nInliers;
      bool bNoMore;
      // 步骤4：通过EPnP算法估计姿态
      PnPsolver *pSolver = vpPnPsolvers[i];
      cv::Mat Tcw        = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

      // If Ransac reachs max. iterations discard keyframe
      if (bNoMore) {
        vbDiscarded[i] = true;
        nCandidates--;
      }

      // If a Camera Pose is computed, optimize
      if (!Tcw.empty()) {
        Tcw.copyTo(mCurrentFrame.mTcw);

        set<MapPoint *> sFound;

        const int np = vbInliers.size();

        for (int j = 0; j < np; j++) {
          if (vbInliers[j]) {
            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
            sFound.insert(vvpMapPointMatches[i][j]);
          } else
            mCurrentFrame.mvpMapPoints[j] = NULL;
        }
        // 步骤5：通过PoseOptimization对姿态进行优化求解
        int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

        if (nGood < 10)
          continue;

        for (int io = 0; io < mCurrentFrame.N; io++)
          if (mCurrentFrame.mvbOutlier[io])
            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

        // If few inliers, search by projection in a coarse window and optimize again
        // 步骤6：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
        if (nGood < 50) {
          int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

          if (nadditional + nGood >= 50) {
            nGood = Optimizer::PoseOptimization(&mCurrentFrame);

            // If many inliers but still not enough, search by projection again in a narrower window
            // the camera has been already optimized with many points
            if (nGood > 30 && nGood < 50) {
              sFound.clear();
              for (int ip = 0; ip < mCurrentFrame.N; ip++)
                if (mCurrentFrame.mvpMapPoints[ip])
                  sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
              nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

              // Final optimization
              if (nGood + nadditional >= 50) {
                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                for (int io = 0; io < mCurrentFrame.N; io++)
                  if (mCurrentFrame.mvbOutlier[io])
                    mCurrentFrame.mvpMapPoints[io] = NULL;
              }
            }
          }
        }

        // If the pose is supported by enough inliers stop ransacs and continue
        if (nGood >= 50) {
          bMatch = true;
          break;
        }
      }// if果通过 pnp 获得初始 位姿
    }// for遍历所有 候选重定位关键帧
  }// while 还没有找到匹配的重定位帧

  // 如果没有找到重定位匹配帧
  if (!bMatch) {
    return false;
  } 
  // 如果找到了
  else {
    mnLastRelocFrameId = mCurrentFrame.mnId;
    return true;
  }
}

void Tracking::Reset() {
  cout << "System Reseting" << endl;
  if (mpViewer) {
    mpViewer->RequestStop();
    while (!mpViewer->isStopped())
      usleep(3000);
  }

  // Reset Local Mapping
  cout << "Reseting Local Mapper...";
  mpLocalMapper->RequestReset();
  cout << " done" << endl;

  // Reset Loop Closing
  cout << "Reseting Loop Closing...";
  mpLoopClosing->RequestReset();
  cout << " done" << endl;

  // Clear BoW Database
  cout << "Reseting Database...";
  mpKeyFrameDB->clear();
  cout << " done" << endl;

  // Clear Map (this erase MapPoints and KeyFrames)
  mpMap->clear();

  KeyFrame::nNextId = 0;
  Frame::nNextId    = 0;
  mState            = NO_IMAGES_YET;

  if (mpInitializer) {
    delete mpInitializer;
    mpInitializer = static_cast<Initializer *>(NULL);
  }

  mlRelativeFramePoses.clear();
  mlpReferences.clear();
  mlFrameTimes.clear();
  mlbLost.clear();

  if (mpViewer)
    mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath) {
  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
  float fx = fSettings["Camera.fx"];
  float fy = fSettings["Camera.fy"];
  float cx = fSettings["Camera.cx"];
  float cy = fSettings["Camera.cy"];

  cv::Mat K         = cv::Mat::eye(3, 3, CV_32F);
  K.at<float>(0, 0) = fx;
  K.at<float>(1, 1) = fy;
  K.at<float>(0, 2) = cx;
  K.at<float>(1, 2) = cy;
  K.copyTo(mK);

  cv::Mat DistCoef(4, 1, CV_32F);
  DistCoef.at<float>(0) = fSettings["Camera.k1"];
  DistCoef.at<float>(1) = fSettings["Camera.k2"];
  DistCoef.at<float>(2) = fSettings["Camera.p1"];
  DistCoef.at<float>(3) = fSettings["Camera.p2"];
  const float k3        = fSettings["Camera.k3"];
  if (k3 != 0) {
    DistCoef.resize(5);
    DistCoef.at<float>(4) = k3;
  }
  DistCoef.copyTo(mDistCoef);

  mbf = fSettings["Camera.bf"];

  Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag) {
  mbOnlyTracking = flag;
}

}  // namespace ORB_SLAM2
