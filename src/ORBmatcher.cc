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

#include "ORBmatcher.h"

#include <limits.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#ifdef __APPLE__
#include <stdint.h>
#else
#include <stdint-gcc.h>
#endif

using namespace std;

namespace ORB_SLAM2 {

const int ORBmatcher::TH_HIGH      = 100;
const int ORBmatcher::TH_LOW       = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

/**
 * Constructor
 * @param nnratio  ratio of the best and the second score
 * @param checkOri check orientation
 */
ORBmatcher::ORBmatcher(float nnratio, bool checkOri)
    : mfNNratio(nnratio), mbCheckOrientation(checkOri) {
}

/**
 * @brief 对于每个局部3D点通过投影在小范围内找到和最匹配的2D点。
 * 从而实现Frame对Local MapPoint的跟踪。用于tracking过程中实现当前帧对局部3D点的跟踪。
 * 将Local MapPoint投影到当前帧中, 由此增加当前帧的MapPoints \n
 * 在SearchLocalPoints()中已经将Local MapPoints重投影（isInFrustum()）到当前帧 \n
 * 并标记了这些点是否在当前帧的视野中，即mbTrackInView \n
 * 对这些MapPoints，在其投影点附近根据描述子距离选取匹配，以及最终的方向投票机制进行剔除
 * @param  F           当前帧
 * @param  vpMapPoints Local MapPoints
 * @param  th          搜索范围因子：r = r * th * ScaleFactor
 * @return             成功匹配的数量
 * @see SearchLocalPoints() isInFrustum()
 */
int ORBmatcher::SearchByProjection(
    Frame &F,
    const vector<MapPoint *> &vpMapPoints,
    const float th) {
  int nmatches = 0;

  const bool bFactor = th != 1.0;

  // 遍历所有 local map points
  for (size_t iMP = 0; iMP < vpMapPoints.size(); iMP++) {
    MapPoint *pMP = vpMapPoints[iMP];
    // 判断该点是否要投影
    if (!pMP->mbTrackInView)
      continue;

    if (pMP->isBad())
      continue;
    // step1：通过距离预测特征点所在的金字塔层数
    const int &nPredictedLevel = pMP->mnTrackScaleLevel;

    // The size of the window will depend on the viewing direction
    // step2：根据观测到该3D点的视角确定搜索窗口的大小, 若相机正对这该3D点则r取一个较小的值（mTrackViewCos>0.998?2.5:4.0）
    float r = RadiusByViewingCos(pMP->mTrackViewCos);

    if (bFactor)
      r *= th;
    // (pMP->mTrackProjX, pMP->mTrackProjY)：图像特征点坐标
    // r*F.mvScaleFactors[nPredictedLevel]：搜索范围
    // nPredictedLevel-1：miniLevel
    // nPredictedLevel：maxLevel
    // step3：在2D投影点附近一定范围内搜索属于miniLevel~maxLevel层的特征点 ---> vIndices
    const vector<size_t> vIndices =
        F.GetFeaturesInArea(pMP->mTrackProjX, pMP->mTrackProjY, r * F.mvScaleFactors[nPredictedLevel], nPredictedLevel - 1, nPredictedLevel);

    if (vIndices.empty())
      continue;

    const cv::Mat MPdescriptor = pMP->GetDescriptor();

    int bestDist   = 256;
    int bestLevel  = -1;
    int bestDist2  = 256;
    int bestLevel2 = -1;
    int bestIdx    = -1;

    // Get best and second matches with near keypoints
    // step4：在vIndices内找到最佳匹配与次佳匹配，如果最优匹配误差小于阈值，且最优匹配明显优于次优匹配，则匹配3D点-2D特征点匹配关联成功
    for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) {
      const size_t idx = *vit;
      // 如果Frame中的该兴趣点已经有对应的MapPoint了，则退出该次循环
      if (F.mvpMapPoints[idx])
        if (F.mvpMapPoints[idx]->Observations() > 0)
          continue;

      if (F.mvuRight[idx] > 0) {
        const float er = fabs(pMP->mTrackProjXR - F.mvuRight[idx]);
        if (er > r * F.mvScaleFactors[nPredictedLevel])
          continue;
      }

      const cv::Mat &d = F.mDescriptors.row(idx);

      const int dist = DescriptorDistance(MPdescriptor, d);
      // 记录最优匹配和次优匹配
      if (dist < bestDist) {
        bestDist2  = bestDist;
        bestDist   = dist;
        bestLevel2 = bestLevel;
        bestLevel  = F.mvKeysUn[idx].octave;
        bestIdx    = idx;
      } else if (dist < bestDist2) {
        bestLevel2 = F.mvKeysUn[idx].octave;
        bestDist2  = dist;
      }
    }

    // Apply ratio to second match (only if best and second are in the same scale level)
    if (bestDist <= TH_HIGH) {
      if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
        continue;
      // 为Frame中的兴趣点增加对应的MapPoint
      F.mvpMapPoints[bestIdx] = pMP;
      nmatches++;
    }
  }

  return nmatches;
}

float ORBmatcher::RadiusByViewingCos(const float &viewCos) {
  if (viewCos > 0.998)
    return 2.5;
  else
    return 4.0;
}

bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF2) {
  // Epipolar line in second image l = x1'F12 = [a b c]
  const float a = kp1.pt.x * F12.at<float>(0, 0) + kp1.pt.y * F12.at<float>(1, 0) + F12.at<float>(2, 0);
  const float b = kp1.pt.x * F12.at<float>(0, 1) + kp1.pt.y * F12.at<float>(1, 1) + F12.at<float>(2, 1);
  const float c = kp1.pt.x * F12.at<float>(0, 2) + kp1.pt.y * F12.at<float>(1, 2) + F12.at<float>(2, 2);

  const float num = a * kp2.pt.x + b * kp2.pt.y + c;

  const float den = a * a + b * b;

  if (den == 0)
    return false;

  const float dsqr = num * num / den;

  return dsqr < 3.84 * pKF2->mvLevelSigma2[kp2.octave];
}

/**
 * @brief 通过语法树加速关键帧与当前帧之间的特征点匹配
 * 
 * 通过bow对pKF和F中的特征点进行快速匹配（不属于同一node的特征点直接跳过匹配） \n
 * 对属于同一node的特征点通过描述子距离进行匹配 \n
 * 根据匹配，用pKF中特征点对应的MapPoint更新F中特征点对应的MapPoints \n
 * 每个特征点都对应一个MapPoint，因此pKF中每个特征点的MapPoint也就是F中对应点的MapPoint \n
 * 通过距离阈值、比例阈值和角度投票进行剔除误匹配
 * @param  pKF               KeyFrame
 * @param  F                 Current Frame
 * @param  vpMapPointMatches F中MapPoints对应的匹配，NULL表示未匹配
 * @return                   成功匹配的数量
 */
int ORBmatcher::SearchByBoW(
    KeyFrame *pKF,
    Frame &F,
    vector<MapPoint *> &vpMapPointMatches) {
  const vector<MapPoint *> vpMapPointsKF = pKF->GetMapPointMatches();

  vpMapPointMatches = vector<MapPoint *>(F.N, static_cast<MapPoint *>(NULL));

  const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

  int nmatches = 0;

  vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++)
    rotHist[i].reserve(500);
  const float factor = 1.0f / HISTO_LENGTH;

  // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
  // 将属于同一节点(特定层)的ORB特征进行匹配
  DBoW2::FeatureVector::const_iterator KFit  = pKF->mFeatVec.begin();
  DBoW2::FeatureVector::const_iterator KFend = pKF->mFeatVec.end();
  DBoW2::FeatureVector::const_iterator Fit   = F.mFeatVec.begin();
  DBoW2::FeatureVector::const_iterator Fend  = F.mFeatVec.end();

  while (KFit != KFend && Fit != Fend) {
    //步骤1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)
    if (KFit->first == Fit->first) {
      const vector<unsigned int> vIndicesKF = KFit->second;
      const vector<unsigned int> vIndicesF  = Fit->second;
      // 步骤2：遍历KF中属于该node的特征点
      for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++) {
        const unsigned int realIdxKF = vIndicesKF[iKF];
        // 取出KF中该特征对应的MapPoint
        MapPoint *pMP = vpMapPointsKF[realIdxKF];

        if (!pMP)
          continue;

        if (pMP->isBad())
          continue;
        // 取出KF中该特征对应的描述子
        const cv::Mat &dKF = pKF->mDescriptors.row(realIdxKF);

        int bestDist1 = 256;  // 最好的距离（最小距离）
        int bestIdxF  = -1;
        int bestDist2 = 256;  // 倒数第二好距离（倒数第二小距离）

        // 步骤3：遍历F中属于该node的特征点，找到了最佳匹配点
        for (size_t iF = 0; iF < vIndicesF.size(); iF++) {
          const unsigned int realIdxF = vIndicesF[iF];
          // 表明这个点已经被匹配过了，不再匹配，加快速度
          if (vpMapPointMatches[realIdxF])
            continue;
          // 取出F中该特征对应的描述子
          const cv::Mat &dF = F.mDescriptors.row(realIdxF);
          // 求描述子的距离
          const int dist = DescriptorDistance(dKF, dF);
          // dist < bestDist1 < bestDist2，更新bestDist1 bestDist2
          if (dist < bestDist1) {
            bestDist2 = bestDist1;
            bestDist1 = dist;
            bestIdxF  = realIdxF;
          }
          // bestDist1 < dist < bestDist2，更新bestDist2
          else if (dist < bestDist2) {
            bestDist2 = dist;
          }
        }
        // 步骤4：根据阈值 和 角度投票剔除误匹配
        // 匹配距离（误差）小于阈值
        if (bestDist1 <= TH_LOW) {
          // trick!
          // 最佳匹配比次佳匹配明显要好，那么最佳匹配才真正靠谱
          if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2)) {
            // 步骤5：更新特征点的MapPoint
            vpMapPointMatches[bestIdxF] = pMP;

            const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

            if (mbCheckOrientation) {
              // trick!
              // angle：每个特征点在提取描述子时的旋转主方向角度，如果图像旋转了，这个角度将发生改变
              // 所有的特征点的角度变化应该是一致的，通过直方图统计得到最准确的角度变化值
              float rot = kp.angle - F.mvKeys[bestIdxF].angle;
              if (rot < 0.0)
                rot += 360.0f;
              int bin = round(rot * factor);
              if (bin == HISTO_LENGTH)
                bin = 0;
              assert(bin >= 0 && bin < HISTO_LENGTH);
              rotHist[bin].push_back(bestIdxF);
            }
            nmatches++;
          }
        }
      }

      KFit++;
      Fit++;
    } else if (KFit->first < Fit->first) {
      KFit = vFeatVecKF.lower_bound(Fit->first);
    } else {
      Fit = F.mFeatVec.lower_bound(KFit->first);
    }
  }
  // 根据方向剔除误匹配的点
  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;
    // 计算rotHist中最大的三个的index
    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);
    // 如果特征点的旋转角度变化量属于这三个组，则保留
    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i == ind1 || i == ind2 || i == ind3)
        continue;
      // 将除了ind1 ind2 ind3以外的匹配点去掉
      for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
        vpMapPointMatches[rotHist[i][j]] = static_cast<MapPoint *>(NULL);
        nmatches--;
      }
    }
  }

  return nmatches;
}

int ORBmatcher::SearchByProjection(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, vector<MapPoint *> &vpMatched, int th) {
  // Get Calibration Parameters for later projection
  const float &fx = pKF->fx;
  const float &fy = pKF->fy;
  const float &cx = pKF->cx;
  const float &cy = pKF->cy;

  // Decompose Scw
  cv::Mat sRcw    = Scw.rowRange(0, 3).colRange(0, 3);
  const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
  cv::Mat Rcw     = sRcw / scw;
  cv::Mat tcw     = Scw.rowRange(0, 3).col(3) / scw;
  cv::Mat Ow      = -Rcw.t() * tcw;

  // Set of MapPoints already found in the KeyFrame
  set<MapPoint *> spAlreadyFound(vpMatched.begin(), vpMatched.end());
  spAlreadyFound.erase(static_cast<MapPoint *>(NULL));

  int nmatches = 0;

  // For each Candidate MapPoint Project and Match
  for (int iMP = 0, iendMP = vpPoints.size(); iMP < iendMP; iMP++) {
    MapPoint *pMP = vpPoints[iMP];

    // Discard Bad MapPoints and already found
    if (pMP->isBad() || spAlreadyFound.count(pMP))
      continue;

    // Get 3D Coords.
    cv::Mat p3Dw = pMP->GetWorldPos();

    // Transform into Camera Coords.
    cv::Mat p3Dc = Rcw * p3Dw + tcw;

    // Depth must be positive
    if (p3Dc.at<float>(2) < 0.0)
      continue;

    // Project into Image
    const float invz = 1 / p3Dc.at<float>(2);
    const float x    = p3Dc.at<float>(0) * invz;
    const float y    = p3Dc.at<float>(1) * invz;

    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    if (!pKF->IsInImage(u, v))
      continue;

    // Depth must be inside the scale invariance region of the point
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    cv::Mat PO              = p3Dw - Ow;
    const float dist        = cv::norm(PO);

    if (dist < minDistance || dist > maxDistance)
      continue;

    // Viewing angle must be less than 60 deg
    cv::Mat Pn = pMP->GetNormal();

    if (PO.dot(Pn) < 0.5 * dist)
      continue;

    int nPredictedLevel = pMP->PredictScale(dist, pKF);

    // Search in a radius
    const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

    const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty())
      continue;

    // Match to the most similar keypoint in the radius
    const cv::Mat dMP = pMP->GetDescriptor();

    int bestDist = 256;
    int bestIdx  = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) {
      const size_t idx = *vit;
      if (vpMatched[idx])
        continue;

      const int &kpLevel = pKF->mvKeysUn[idx].octave;

      if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
        continue;

      const cv::Mat &dKF = pKF->mDescriptors.row(idx);

      const int dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx  = idx;
      }
    }

    if (bestDist <= TH_LOW) {
      vpMatched[bestIdx] = pMP;
      nmatches++;
    }
  }

  return nmatches;
}

// 初始化时假设F1和F2图像变化不大，在windowSize范围进行匹配，外部调用中windowSize = 100
int ORBmatcher::SearchForInitialization(
    Frame &F1,                           // 初始化第一帧
    Frame &F2,                           // 初始化第二帧
    vector<cv::Point2f> &vbPrevMatched,  // 输入时是第一帧关键点，输出时是第二帧关键点
    vector<int> &vnMatches12,            // 一二帧间匹配
    int windowSize)                      // 窗口大小 =100
{
  int nmatches = 0;
  vnMatches12  = vector<int>(F1.mvKeysUn.size(), -1);  // F1 各关键点最近的 F2关键点

  //30 长的数组，储存int vector，方向差 的直方图，30个bin, bin 中存 F1 的keypoint id
  vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++)
    rotHist[i].reserve(500);                 // 每个vector预分配500长度
  const float factor = 1.0f / HISTO_LENGTH;  // =1/30

  vector<int> vMatchedDistance(F2.mvKeysUn.size(), INT_MAX);
  vector<int> vnMatches21(F2.mvKeysUn.size(), -1);  // F2 各关键点最近的 F1关键点

  // 遍历 Frame1 关键点
  for (size_t i1 = 0, iend1 = F1.mvKeysUn.size(); i1 < iend1; i1++) {
    cv::KeyPoint kp1 = F1.mvKeysUn[i1];
    int level1       = kp1.octave;

    // 只要 Level = 0 层的 ORB 特征点。也就是金字塔中最原始的那层。
    if (level1 > 0)
      continue;

    // 在 F1 关键点的小窗口附近提取出 F2 的所有关键点
    vector<size_t> vIndices2 =
        F2.GetFeaturesInArea(vbPrevMatched[i1].x, vbPrevMatched[i1].y,
                             windowSize, level1, level1);

    if (vIndices2.empty())
      continue;

    // F1 关键点的描述子
    cv::Mat d1 = F1.mDescriptors.row(i1);

    int bestDist  = INT_MAX;  // 最好的关键点描述子距离
    int bestDist2 = INT_MAX;  // 第二好的关键点描述子距离
    int bestIdx2  = -1;       // 第二好的 id

    // 遍历窗口内 F2 关键点
    for (vector<size_t>::iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++) {
      size_t i2 = *vit;
      // F2 关键点的描述子
      cv::Mat d2 = F2.mDescriptors.row(i2);
      // 描述子距离
      int dist = DescriptorDistance(d1, d2);

      // 如果是一个无效距离?
      if (vMatchedDistance[i2] <= dist)
        continue;
      // 如果比最好的好 , 更新最佳
      if (dist < bestDist) {
        bestDist2 = bestDist;  // 第二好的距离
        bestDist  = dist;      // 最好的距离
        bestIdx2  = i2;        // F2最好匹配点的 id
      }
      // 如果比第二好的好， 更新次佳
      else if (dist < bestDist2) {
        bestDist2 = dist;
      }
    }  // 完成  遍历窗口内 F2 关键点

    // 步骤4：根据阈值 和 角度投票剔除误匹配
    if (bestDist <= TH_LOW) {
      // trick!
      // 最佳匹配比次佳匹配明显要好，那么最佳匹配才真正靠谱
      if (bestDist < (float)bestDist2 * mfNNratio) {
        if (vnMatches21[bestIdx2] >= 0) {
          vnMatches12[vnMatches21[bestIdx2]] = -1;
          nmatches--;
        }
        vnMatches12[i1]            = bestIdx2;
        vnMatches21[bestIdx2]      = i1;
        vMatchedDistance[bestIdx2] = bestDist;
        nmatches++;

        if (mbCheckOrientation)  // 更新方向差值直方图
        {
          float rot = F1.mvKeysUn[i1].angle - F2.mvKeysUn[bestIdx2].angle;
          if (rot < 0.0)
            rot += 360.0f;
          int bin = round(rot * factor);
          if (bin == HISTO_LENGTH)
            bin = 0;
          assert(bin >= 0 && bin < HISTO_LENGTH);
          rotHist[bin].push_back(i1);
        }
      }
    }
  }  //完成 遍历 Frame1 关键点, 完成关键点对的提取

  // 依据旋转直方图删除不好的匹配
  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    // 取出3个最大的bin 的id，最大就是 vector 最长
    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    // 这里认为前三大的bin 中的匹配是好的匹配，删除剩下的匹配
    // 遍历所有 bin
    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i == ind1 || i == ind2 || i == ind3)
        continue;
      // 遍历剩下 bin 中左右 F1 的keypoint id
      for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
        int idx1 = rotHist[i][j];
        if (vnMatches12[idx1] >= 0) {
          vnMatches12[idx1] = -1;
          nmatches--;
        }
      }
    }
  }

  //Update prev matched
  for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
    if (vnMatches12[i1] >= 0)
      vbPrevMatched[i1] = F2.mvKeysUn[vnMatches12[i1]].pt;

  return nmatches;
}

int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12) {
  const vector<cv::KeyPoint> &vKeysUn1  = pKF1->mvKeysUn;
  const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
  const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
  const cv::Mat &Descriptors1           = pKF1->mDescriptors;

  const vector<cv::KeyPoint> &vKeysUn2  = pKF2->mvKeysUn;
  const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
  const vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();
  const cv::Mat &Descriptors2           = pKF2->mDescriptors;

  vpMatches12 = vector<MapPoint *>(vpMapPoints1.size(), static_cast<MapPoint *>(NULL));
  vector<bool> vbMatched2(vpMapPoints2.size(), false);

  vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++)
    rotHist[i].reserve(500);

  const float factor = 1.0f / HISTO_LENGTH;

  int nmatches = 0;

  DBoW2::FeatureVector::const_iterator f1it  = vFeatVec1.begin();
  DBoW2::FeatureVector::const_iterator f2it  = vFeatVec2.begin();
  DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
  DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

  while (f1it != f1end && f2it != f2end) {
    if (f1it->first == f2it->first) {
      for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++) {
        const size_t idx1 = f1it->second[i1];

        MapPoint *pMP1 = vpMapPoints1[idx1];
        if (!pMP1)
          continue;
        if (pMP1->isBad())
          continue;

        const cv::Mat &d1 = Descriptors1.row(idx1);

        int bestDist1 = 256;
        int bestIdx2  = -1;
        int bestDist2 = 256;

        for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++) {
          const size_t idx2 = f2it->second[i2];

          MapPoint *pMP2 = vpMapPoints2[idx2];

          if (vbMatched2[idx2] || !pMP2)
            continue;

          if (pMP2->isBad())
            continue;

          const cv::Mat &d2 = Descriptors2.row(idx2);

          int dist = DescriptorDistance(d1, d2);

          if (dist < bestDist1) {
            bestDist2 = bestDist1;
            bestDist1 = dist;
            bestIdx2  = idx2;
          } else if (dist < bestDist2) {
            bestDist2 = dist;
          }
        }

        if (bestDist1 < TH_LOW) {
          if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2)) {
            vpMatches12[idx1]    = vpMapPoints2[bestIdx2];
            vbMatched2[bestIdx2] = true;

            if (mbCheckOrientation) {
              float rot = vKeysUn1[idx1].angle - vKeysUn2[bestIdx2].angle;
              if (rot < 0.0)
                rot += 360.0f;
              int bin = round(rot * factor);
              if (bin == HISTO_LENGTH)
                bin = 0;
              assert(bin >= 0 && bin < HISTO_LENGTH);
              rotHist[bin].push_back(idx1);
            }
            nmatches++;
          }
        }
      }

      f1it++;
      f2it++;
    } else if (f1it->first < f2it->first) {
      f1it = vFeatVec1.lower_bound(f2it->first);
    } else {
      f2it = vFeatVec2.lower_bound(f1it->first);
    }
  }

  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i == ind1 || i == ind2 || i == ind3)
        continue;
      for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
        vpMatches12[rotHist[i][j]] = static_cast<MapPoint *>(NULL);
        nmatches--;
      }
    }
  }

  return nmatches;
}

/**
 * @brief 利用基本矩阵F12，在pKF1和pKF2之间找特征匹配。
 * 作用：当pKF1中特征点没有对应的3D点时，通过匹配的特征点产生新的3d点
 * @param pKF1          关键帧1
 * @param pKF2          关键帧2
 * @param F12           基础矩阵
 * @param vMatchedPairs 存储匹配特征点对，特征点用其在关键帧中的索引表示
 * @param bOnlyStereo   在双目和rgbd情况下，要求特征点在右图存在匹配
 * @return              成功匹配的数量
 */
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1,
                                       KeyFrame *pKF2,
                                       cv::Mat F12,
                                       vector<pair<size_t, size_t>> &vMatchedPairs,
                                       const bool bOnlyStereo) {
  const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
  const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

  // Compute epipole in second image
  // 计算KF1的相机中心在KF2图像平面的坐标，即极点坐标
  cv::Mat Cw       = pKF1->GetCameraCenter();  //世界系下，1帧的位置 // tw2c1
  cv::Mat R2w      = pKF2->GetRotation();      //2帧系下，世界系的姿态 // Rc2w
  cv::Mat t2w      = pKF2->GetTranslation();   //2帧系下，世界系的位置 // tc2w
  cv::Mat C2       = R2w * Cw + t2w;           //2帧系下，1帧的位置 // tc2c1 KF1的相机中心在KF2坐标系的表示
  const float invz = 1.0f / C2.at<float>(2);

  // 步骤0：得到KF1的相机光心在KF2中的坐标（KF1在KF2中的极点坐标）
  // 1帧 在 2帧归一化平面下的坐标
  const float ex = pKF2->fx * C2.at<float>(0) * invz + pKF2->cx;
  const float ey = pKF2->fy * C2.at<float>(1) * invz + pKF2->cy;

  // Find matches between not tracked keypoints
  // Matching speed-up by ORB Vocabulary
  // Compare only ORB that share the same node

  int nmatches = 0;
  vector<bool> vbMatched2(pKF2->N, false);
  vector<int> vMatches12(pKF1->N, -1);

  // 旋转直方图，长度30
  vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++)
    rotHist[i].reserve(500);

  const float factor = 1.0f / HISTO_LENGTH;
  // We perform the matching over ORB that belong to the same vocabulary node
  // (at a certain level)
  // 将属于同一节点(特定层)的ORB特征进行匹配
  // FeatureVector的数据结构类似于：{(node1,feature_vector1) (node2,feature_vector2)...}
  // f1it->first对应node编号，f1it->second对应属于该node的所有特特征点编号
  DBoW2::FeatureVector::const_iterator f1it  = vFeatVec1.begin();
  DBoW2::FeatureVector::const_iterator f2it  = vFeatVec2.begin();
  DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
  DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

  // 步骤1：遍历pKF1和pKF2中的node节点
  while (f1it != f1end && f2it != f2end) {
    if (f1it->first == f2it->first)
    // 如果f1it和f2it属于同一个node节点
    {
      // 步骤2：遍历该node节点下(f1it->first)的所有特征点
      for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++) {
        // 获取pKF1中属于该node节点的所有特征点索引
        const size_t idx1 = f1it->second[i1];

        // 步骤2.1：通过特征点索引idx1在pKF1中取出对应的MapPoint
        MapPoint *pMP1 = pKF1->GetMapPoint(idx1);

        // If there is already a MapPoint skip
        // ！！！！！！由于寻找的是未匹配的特征点，所以pMP1应该为NULL
        if (pMP1)
          continue;

        // 如果mvuRight中的值大于0，表示是双目，且该特征点有深度值
        const bool bStereo1 = pKF1->mvuRight[idx1] >= 0;

        // 不考虑双目深度的有效性
        if (bOnlyStereo)
          if (!bStereo1)
            continue;

        // 步骤2.2：通过特征点索引idx1在pKF1中取出对应的特征点
        const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];

        // 步骤2.3：通过特征点索引idx1在pKF1中取出对应的特征点的描述子
        const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

        int bestDist = TH_LOW;
        int bestIdx2 = -1;

        // 步骤3：遍历该node节点下(f2it->first)的所有特征点
        for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++) {
          // 获取pKF2中属于该node节点的所有特征点索引
          size_t idx2 = f2it->second[i2];

          // 步骤3.1：通过特征点索引idx2在pKF2中取出对应的MapPoint
          MapPoint *pMP2 = pKF2->GetMapPoint(idx2);

          // If we have already matched or there is a MapPoint skip
          // 如果pKF2当前特征点索引idx2已经被匹配过或者对应的3d点非空
          // 那么这个索引idx2就不能被考虑
          if (vbMatched2[idx2] || pMP2)
            continue;

          const bool bStereo2 = pKF2->mvuRight[idx2] >= 0;

          if (bOnlyStereo)
            if (!bStereo2)
              continue;

          // 步骤3.2：通过特征点索引idx2在pKF2中取出对应的特征点的描述子
          const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

          // 计算idx1与idx2在两个关键帧中对应特征点的描述子距离
          const int dist = DescriptorDistance(d1, d2);

          if (dist > TH_LOW || dist > bestDist)
            continue;

          // 步骤3.3：通过特征点索引idx2在pKF2中取出对应的特征点
          const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

          if (!bStereo1 && !bStereo2) {
            const float distex = ex - kp2.pt.x;
            const float distey = ey - kp2.pt.y;
            // ！！！！该特征点距离极点太近，表明kp2对应的MapPoint距离pKF1相机太近
            if (distex * distex + distey * distey < 100 * pKF2->mvScaleFactors[kp2.octave])
              continue;
          }

          // 步骤4：计算特征点kp2到kp1极线（kp1对应pKF2的一条极线）的距离是否小于阈值
          if (CheckDistEpipolarLine(kp1, kp2, F12, pKF2)) {
            bestIdx2 = idx2;
            bestDist = dist;
          }
        }

        // 步骤1、2、3、4总结下来就是：将左图像的每个特征点与右图像同一node节点的所有特征点
        // 依次检测，判断是否满足对极几何约束，满足约束就是匹配的特征点

        // 详见SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)函数步骤4
        if (bestIdx2 >= 0) {
          const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
          vMatches12[idx1]        = bestIdx2;
          vbMatched2[bestIdx2]    = true;
          nmatches++;

          if (mbCheckOrientation) {
            float rot = kp1.angle - kp2.angle;
            if (rot < 0.0)
              rot += 360.0f;
            int bin = round(rot * factor);
            if (bin == HISTO_LENGTH)
              bin = 0;
            assert(bin >= 0 && bin < HISTO_LENGTH);
            rotHist[bin].push_back(idx1);
          }
        }
      }

      f1it++;
      f2it++;
    } else if (f1it->first < f2it->first) {
      f1it = vFeatVec1.lower_bound(f2it->first);
    } else {
      f2it = vFeatVec2.lower_bound(f1it->first);
    }
  }

  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i == ind1 || i == ind2 || i == ind3)
        continue;
      for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
        vMatches12[rotHist[i][j]] = -1;
        nmatches--;
      }
    }
  }

  vMatchedPairs.clear();
  vMatchedPairs.reserve(nmatches);

  for (size_t i = 0, iend = vMatches12.size(); i < iend; i++) {
    if (vMatches12[i] < 0)
      continue;
    vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
  }

  return nmatches;
}

/**
 * @brief 将MapPoints投影（用关键帧的位姿）到关键帧pKF中，并判断是否有重复的MapPoints
 * 1.如果MapPoint能匹配关键帧的特征点，并且该点有对应的MapPoint，
 * 那么将两个MapPoint合并（选择观测数多的）
 * 2.如果MapPoint能匹配关键帧的特征点，并且该点没有对应的MapPoint，那么为该点添加MapPoint
 * @param  pKF         相邻关键帧
 * @param  vpMapPoints 当前关键帧的MapPoints
 * @param  th          搜索半径的因子
 * @return             重复MapPoints的数量
 */
int ORBmatcher::Fuse(KeyFrame *pKF,
                     const vector<MapPoint *> &vpMapPoints,
                     const float th) {
  cv::Mat Rcw = pKF->GetRotation();
  cv::Mat tcw = pKF->GetTranslation();

  const float &fx = pKF->fx;
  const float &fy = pKF->fy;
  const float &cx = pKF->cx;
  const float &cy = pKF->cy;
  const float &bf = pKF->mbf;

  cv::Mat Ow = pKF->GetCameraCenter();

  int nFused = 0;

  const int nMPs = vpMapPoints.size();

  // 遍历所有的MapPoints
  for (int i = 0; i < nMPs; i++) {
    MapPoint *pMP = vpMapPoints[i];

    if (!pMP)
      continue;

    if (pMP->isBad() || pMP->IsInKeyFrame(pKF))
      continue;

    cv::Mat p3Dw = pMP->GetWorldPos();
    cv::Mat p3Dc = Rcw * p3Dw + tcw;

    // Depth must be positive
    if (p3Dc.at<float>(2) < 0.0f)
      continue;

    const float invz = 1 / p3Dc.at<float>(2);
    const float x    = p3Dc.at<float>(0) * invz;
    const float y    = p3Dc.at<float>(1) * invz;

    const float u = fx * x + cx;
    const float v = fy * y + cy;  // 步骤1：得到MapPoint在图像上的投影坐标

    // Point must be inside the image
    if (!pKF->IsInImage(u, v))
      continue;

    const float ur = u - bf * invz;

    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    cv::Mat PO              = p3Dw - Ow;
    const float dist3D      = cv::norm(PO);

    // Depth must be inside the scale pyramid of the image
    if (dist3D < minDistance || dist3D > maxDistance)
      continue;

    // Viewing angle must be less than 60 deg
    cv::Mat Pn = pMP->GetNormal();

    if (PO.dot(Pn) < 0.5 * dist3D)
      continue;

    int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

    // Search in a radius
    const float radius = th * pKF->mvScaleFactors[nPredictedLevel];  // 步骤2：根据MapPoint的深度确定尺度，从而确定搜索范围

    const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty())
      continue;

    // Match to the most similar keypoint in the radius

    const cv::Mat dMP = pMP->GetDescriptor();

    int bestDist = 256;
    int bestIdx  = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)  // 步骤3：遍历搜索范围内的features
    {
      const size_t idx = *vit;

      const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

      const int &kpLevel = kp.octave;

      if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
        continue;

      // 计算MapPoint投影的坐标与这个区域特征点的距离，如果偏差很大，直接跳过特征点匹配
      if (pKF->mvuRight[idx] >= 0) {
        // Check reprojection error in stereo
        const float &kpx = kp.pt.x;
        const float &kpy = kp.pt.y;
        const float &kpr = pKF->mvuRight[idx];
        const float ex   = u - kpx;
        const float ey   = v - kpy;
        const float er   = ur - kpr;
        const float e2   = ex * ex + ey * ey + er * er;

        if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 7.8)
          continue;
      } else {
        const float &kpx = kp.pt.x;
        const float &kpy = kp.pt.y;
        const float ex   = u - kpx;
        const float ey   = v - kpy;
        const float e2   = ex * ex + ey * ey;

        // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
        if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 5.99)
          continue;
      }

      const cv::Mat &dKF = pKF->mDescriptors.row(idx);

      const int dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist)  // 找MapPoint在该区域最佳匹配的特征点
      {
        bestDist = dist;
        bestIdx  = idx;
      }
    }

    // If there is already a MapPoint replace otherwise add new measurement
    if (bestDist <= TH_LOW)  // 找到了MapPoint在该区域最佳匹配的特征点
    {
      MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);
      if (pMPinKF)  // 如果这个点有对应的MapPoint
      {
        if (!pMPinKF->isBad())  // 如果这个MapPoint不是bad，选择哪一个呢？
        {
          if (pMPinKF->Observations() > pMP->Observations())
            pMP->Replace(pMPinKF);
          else
            pMPinKF->Replace(pMP);
        }
      } else  // 如果这个点没有对应的MapPoint
      {
        pMP->AddObservation(pKF, bestIdx);
        pKF->AddMapPoint(pMP, bestIdx);
      }
      nFused++;
    }
  }

  return nFused;
}

int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint) {
  // Get Calibration Parameters for later projection
  const float &fx = pKF->fx;
  const float &fy = pKF->fy;
  const float &cx = pKF->cx;
  const float &cy = pKF->cy;

  // Decompose Scw
  cv::Mat sRcw    = Scw.rowRange(0, 3).colRange(0, 3);
  const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
  cv::Mat Rcw     = sRcw / scw;
  cv::Mat tcw     = Scw.rowRange(0, 3).col(3) / scw;
  cv::Mat Ow      = -Rcw.t() * tcw;

  // Set of MapPoints already found in the KeyFrame
  const set<MapPoint *> spAlreadyFound = pKF->GetMapPoints();

  int nFused = 0;

  const int nPoints = vpPoints.size();

  // For each candidate MapPoint project and match
  for (int iMP = 0; iMP < nPoints; iMP++) {
    MapPoint *pMP = vpPoints[iMP];

    // Discard Bad MapPoints and already found
    if (pMP->isBad() || spAlreadyFound.count(pMP))
      continue;

    // Get 3D Coords.
    cv::Mat p3Dw = pMP->GetWorldPos();

    // Transform into Camera Coords.
    cv::Mat p3Dc = Rcw * p3Dw + tcw;

    // Depth must be positive
    if (p3Dc.at<float>(2) < 0.0f)
      continue;

    // Project into Image
    const float invz = 1.0 / p3Dc.at<float>(2);
    const float x    = p3Dc.at<float>(0) * invz;
    const float y    = p3Dc.at<float>(1) * invz;

    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    if (!pKF->IsInImage(u, v))
      continue;

    // Depth must be inside the scale pyramid of the image
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    cv::Mat PO              = p3Dw - Ow;
    const float dist3D      = cv::norm(PO);

    if (dist3D < minDistance || dist3D > maxDistance)
      continue;

    // Viewing angle must be less than 60 deg
    cv::Mat Pn = pMP->GetNormal();

    if (PO.dot(Pn) < 0.5 * dist3D)
      continue;

    // Compute predicted scale level
    const int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

    // Search in a radius
    const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

    const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty())
      continue;

    // Match to the most similar keypoint in the radius

    const cv::Mat dMP = pMP->GetDescriptor();

    int bestDist = INT_MAX;
    int bestIdx  = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin(); vit != vIndices.end(); vit++) {
      const size_t idx   = *vit;
      const int &kpLevel = pKF->mvKeysUn[idx].octave;

      if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
        continue;

      const cv::Mat &dKF = pKF->mDescriptors.row(idx);

      int dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx  = idx;
      }
    }

    // If there is already a MapPoint replace otherwise add new measurement
    if (bestDist <= TH_LOW) {
      MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);
      if (pMPinKF) {
        if (!pMPinKF->isBad())
          vpReplacePoint[iMP] = pMPinKF;
      } else {
        pMP->AddObservation(pKF, bestIdx);
        pKF->AddMapPoint(pMP, bestIdx);
      }
      nFused++;
    }
  }

  return nFused;
}

int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th) {
  const float &fx = pKF1->fx;
  const float &fy = pKF1->fy;
  const float &cx = pKF1->cx;
  const float &cy = pKF1->cy;

  // Camera 1 from world
  cv::Mat R1w = pKF1->GetRotation();
  cv::Mat t1w = pKF1->GetTranslation();

  //Camera 2 from world
  cv::Mat R2w = pKF2->GetRotation();
  cv::Mat t2w = pKF2->GetTranslation();

  //Transformation between cameras
  cv::Mat sR12 = s12 * R12;
  cv::Mat sR21 = (1.0 / s12) * R12.t();
  cv::Mat t21  = -sR21 * t12;

  const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
  const int N1                          = vpMapPoints1.size();

  const vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();
  const int N2                          = vpMapPoints2.size();

  vector<bool> vbAlreadyMatched1(N1, false);
  vector<bool> vbAlreadyMatched2(N2, false);

  for (int i = 0; i < N1; i++) {
    MapPoint *pMP = vpMatches12[i];
    if (pMP) {
      vbAlreadyMatched1[i] = true;
      int idx2             = pMP->GetIndexInKeyFrame(pKF2);
      if (idx2 >= 0 && idx2 < N2)
        vbAlreadyMatched2[idx2] = true;
    }
  }

  vector<int> vnMatch1(N1, -1);
  vector<int> vnMatch2(N2, -1);

  // Transform from KF1 to KF2 and search
  for (int i1 = 0; i1 < N1; i1++) {
    MapPoint *pMP = vpMapPoints1[i1];

    if (!pMP || vbAlreadyMatched1[i1])
      continue;

    if (pMP->isBad())
      continue;

    cv::Mat p3Dw  = pMP->GetWorldPos();
    cv::Mat p3Dc1 = R1w * p3Dw + t1w;
    cv::Mat p3Dc2 = sR21 * p3Dc1 + t21;

    // Depth must be positive
    if (p3Dc2.at<float>(2) < 0.0)
      continue;

    const float invz = 1.0 / p3Dc2.at<float>(2);
    const float x    = p3Dc2.at<float>(0) * invz;
    const float y    = p3Dc2.at<float>(1) * invz;

    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    if (!pKF2->IsInImage(u, v))
      continue;

    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const float dist3D      = cv::norm(p3Dc2);

    // Depth must be inside the scale invariance region
    if (dist3D < minDistance || dist3D > maxDistance)
      continue;

    // Compute predicted octave
    const int nPredictedLevel = pMP->PredictScale(dist3D, pKF2);

    // Search in a radius
    const float radius = th * pKF2->mvScaleFactors[nPredictedLevel];

    const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty())
      continue;

    // Match to the most similar keypoint in the radius
    const cv::Mat dMP = pMP->GetDescriptor();

    int bestDist = INT_MAX;
    int bestIdx  = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) {
      const size_t idx = *vit;

      const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

      if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
        continue;

      const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

      const int dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx  = idx;
      }
    }

    if (bestDist <= TH_HIGH) {
      vnMatch1[i1] = bestIdx;
    }
  }

  // Transform from KF2 to KF2 and search
  for (int i2 = 0; i2 < N2; i2++) {
    MapPoint *pMP = vpMapPoints2[i2];

    if (!pMP || vbAlreadyMatched2[i2])
      continue;

    if (pMP->isBad())
      continue;

    cv::Mat p3Dw  = pMP->GetWorldPos();
    cv::Mat p3Dc2 = R2w * p3Dw + t2w;
    cv::Mat p3Dc1 = sR12 * p3Dc2 + t12;

    // Depth must be positive
    if (p3Dc1.at<float>(2) < 0.0)
      continue;

    const float invz = 1.0 / p3Dc1.at<float>(2);
    const float x    = p3Dc1.at<float>(0) * invz;
    const float y    = p3Dc1.at<float>(1) * invz;

    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    if (!pKF1->IsInImage(u, v))
      continue;

    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const float dist3D      = cv::norm(p3Dc1);

    // Depth must be inside the scale pyramid of the image
    if (dist3D < minDistance || dist3D > maxDistance)
      continue;

    // Compute predicted octave
    const int nPredictedLevel = pMP->PredictScale(dist3D, pKF1);

    // Search in a radius of 2.5*sigma(ScaleLevel)
    const float radius = th * pKF1->mvScaleFactors[nPredictedLevel];

    const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty())
      continue;

    // Match to the most similar keypoint in the radius
    const cv::Mat dMP = pMP->GetDescriptor();

    int bestDist = INT_MAX;
    int bestIdx  = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) {
      const size_t idx = *vit;

      const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

      if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
        continue;

      const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

      const int dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx  = idx;
      }
    }

    if (bestDist <= TH_HIGH) {
      vnMatch2[i2] = bestIdx;
    }
  }

  // Check agreement
  int nFound = 0;

  for (int i1 = 0; i1 < N1; i1++) {
    int idx2 = vnMatch1[i1];

    if (idx2 >= 0) {
      int idx1 = vnMatch2[idx2];
      if (idx1 == i1) {
        vpMatches12[i1] = vpMapPoints2[idx2];
        nFound++;
      }
    }
  }

  return nFound;
}

/**
 * @brief 对上一帧每个3D点通过投影在小范围内找到和最匹配的2D点。从而实现当前帧CurrentFrame对上一帧LastFrame 3D点的匹配跟踪。用于tracking中前后帧跟踪
 *
 * 上一帧中包含了MapPoints，对这些MapPoints进行tracking，由此增加当前帧的MapPoints \n
 * 1. 将上一帧的MapPoints投影到当前帧(根据速度模型可以估计当前帧的Tcw)
 * 2. 在投影点附近根据描述子距离选取匹配，以及最终的方向投票机制进行剔除
 * @param  CurrentFrame 当前帧
 * @param  LastFrame    上一帧
 * @param  th           阈值
 * @param  bMono        是否为单目
 * @return              成功匹配的数量
 * @see SearchByBoW()
 */
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame,
                                   const float th, const bool bMono) {
  int nmatches = 0;

  // Rotation Histogram (to check rotation consistency)
  // 30 个分区的角度直方图
  vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++)
    rotHist[i].reserve(500);
  const float factor = 1.0f / HISTO_LENGTH;

  // 当前帧的 Camera_World
  const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
  const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0, 3).col(3);

  // 当前帧的 World_Camera
  const cv::Mat twc = -Rcw.t() * tcw;

  // 上一帧Camera World
  const cv::Mat Rlw = LastFrame.mTcw.rowRange(0, 3).colRange(0, 3);
  const cv::Mat tlw = LastFrame.mTcw.rowRange(0, 3).col(3);

  // 上一帧camera _ 当前帧 Camera
  const cv::Mat tlc = Rlw * twc + tlw;

  // 判断前进还是后退，并以此预测特征点在当前帧所在的金字塔层数
  const bool bForward  = tlc.at<float>(2) > CurrentFrame.mb && !bMono;   // 非单目情况，如果Z大于基线，则表示前进
  const bool bBackward = -tlc.at<float>(2) > CurrentFrame.mb && !bMono;  // 非单目情况，如果Z小于基线，则表示后退

  for (int i = 0; i < LastFrame.N; i++) {
    // 取上一帧看到的 MapPoint
    MapPoint *pMP = LastFrame.mvpMapPoints[i];

    if (pMP) {
      if (!LastFrame.mvbOutlier[i]) {
        // 对上一帧有效的MapPoints进行跟踪
        // Project
        cv::Mat x3Dw = pMP->GetWorldPos();
        // 得上一帧看到的MapPoint在当前帧系下的坐标
        cv::Mat x3Dc = Rcw * x3Dw + tcw;

        const float xc    = x3Dc.at<float>(0);
        const float yc    = x3Dc.at<float>(1);
        const float invzc = 1.0 / x3Dc.at<float>(2);

        if (invzc < 0)
          continue;

        // 投影到图像，得 uv，需要内参数参与计算
        float u = CurrentFrame.fx * xc * invzc + CurrentFrame.cx;
        float v = CurrentFrame.fy * yc * invzc + CurrentFrame.cy;

        if (u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX)
          continue;
        if (v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY)
          continue;

        int nLastOctave = LastFrame.mvKeys[i].octave;

        // Search in a window. Size depends on scale
        float radius = th * CurrentFrame.mvScaleFactors[nLastOctave];

        vector<size_t> vIndices2;
        // NOTE 尺度越大,图像越小
        // 以下可以这么理解，例如一个有一定面积的圆点，在某个尺度n下它是一个特征点
        // 当前进时，圆点的面积增大，在某个尺度m下它是一个特征点，由于面积增大，则需要在更高的尺度下才能检测出来
        // 因此m>=n，对应前进的情况，nCurOctave>=nLastOctave。后退的情况可以类推
        // 前进,则上一帧兴趣点在所在的尺度nLastOctave<=nCurOctave
        if (bForward)
          vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nLastOctave);
        // 后退,则上一帧兴趣点在所在的尺度0<=nCurOctave<=nLastOctave
        else if (bBackward)
          vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, 0, nLastOctave);
        // 在[nLastOctave-1, nLastOctave+1]中搜索
        else
          vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nLastOctave - 1, nLastOctave + 1);

        if (vIndices2.empty())
          continue;

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx2 = -1;
        // 遍历满足条件的特征点
        for (vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++) {
          // 如果该特征点已经有对应的MapPoint了,则退出该次循环
          const size_t i2 = *vit;
          if (CurrentFrame.mvpMapPoints[i2])
            if (CurrentFrame.mvpMapPoints[i2]->Observations() > 0)
              continue;

          if (CurrentFrame.mvuRight[i2] > 0) {
            // 双目和rgbd的情况，需要保证右图的点也在搜索半径以内
            const float ur = u - CurrentFrame.mbf * invzc;
            const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
            if (er > radius)
              continue;
          }

          const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

          const int dist = DescriptorDistance(dMP, d);

          if (dist < bestDist) {
            bestDist = dist;
            bestIdx2 = i2;
          }
        }
        // 详见SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)函数步骤4
        if (bestDist <= TH_HIGH) {
          CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
          nmatches++;

          if (mbCheckOrientation) {
            float rot = LastFrame.mvKeysUn[i].angle - CurrentFrame.mvKeysUn[bestIdx2].angle;
            if (rot < 0.0)
              rot += 360.0f;
            int bin = round(rot * factor);
            if (bin == HISTO_LENGTH)
              bin = 0;
            assert(bin >= 0 && bin < HISTO_LENGTH);
            rotHist[bin].push_back(bestIdx2);
          }
        }
      }
    }
  }

  //Apply rotation consistency
  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i != ind1 && i != ind2 && i != ind3) {
        for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
          CurrentFrame.mvpMapPoints[rotHist[i][j]] = static_cast<MapPoint *>(NULL);
          nmatches--;
        }
      }
    }
  }

  return nmatches;
}

// 对当前帧每个3D点通过投影在小范围内找到和最匹配的2D点。从而实现当前帧CurrentFrame对关键帧3D点的匹配跟踪。用于重定位时特征点匹配
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint *> &sAlreadyFound, const float th, const int ORBdist) {
  int nmatches = 0;

  const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
  const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0, 3).col(3);
  const cv::Mat Ow  = -Rcw.t() * tcw;

  // Rotation Histogram (to check rotation consistency)
  vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++)
    rotHist[i].reserve(500);
  const float factor = 1.0f / HISTO_LENGTH;

  const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

  for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
    MapPoint *pMP = vpMPs[i];

    if (pMP) {
      if (!pMP->isBad() && !sAlreadyFound.count(pMP)) {
        //Project
        cv::Mat x3Dw = pMP->GetWorldPos();
        cv::Mat x3Dc = Rcw * x3Dw + tcw;

        const float xc    = x3Dc.at<float>(0);
        const float yc    = x3Dc.at<float>(1);
        const float invzc = 1.0 / x3Dc.at<float>(2);

        const float u = CurrentFrame.fx * xc * invzc + CurrentFrame.cx;
        const float v = CurrentFrame.fy * yc * invzc + CurrentFrame.cy;

        if (u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX)
          continue;
        if (v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY)
          continue;

        // Compute predicted scale level
        cv::Mat PO   = x3Dw - Ow;
        float dist3D = cv::norm(PO);

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();

        // Depth must be inside the scale pyramid of the image
        if (dist3D < minDistance || dist3D > maxDistance)
          continue;

        int nPredictedLevel = pMP->PredictScale(dist3D, &CurrentFrame);

        // Search in a window
        const float radius = th * CurrentFrame.mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel - 1, nPredictedLevel + 1);

        if (vIndices2.empty())
          continue;

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx2 = -1;

        for (vector<size_t>::const_iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++) {
          const size_t i2 = *vit;
          if (CurrentFrame.mvpMapPoints[i2])
            continue;

          const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

          const int dist = DescriptorDistance(dMP, d);

          if (dist < bestDist) {
            bestDist = dist;
            bestIdx2 = i2;
          }
        }

        if (bestDist <= ORBdist) {
          CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
          nmatches++;

          if (mbCheckOrientation) {
            float rot = pKF->mvKeysUn[i].angle - CurrentFrame.mvKeysUn[bestIdx2].angle;
            if (rot < 0.0)
              rot += 360.0f;
            int bin = round(rot * factor);
            if (bin == HISTO_LENGTH)
              bin = 0;
            assert(bin >= 0 && bin < HISTO_LENGTH);
            rotHist[bin].push_back(bestIdx2);
          }
        }
      }
    }
  }

  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i != ind1 && i != ind2 && i != ind3) {
        for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
          CurrentFrame.mvpMapPoints[rotHist[i][j]] = NULL;
          nmatches--;
        }
      }
    }
  }

  return nmatches;
}

void ORBmatcher::ComputeThreeMaxima(vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3) {
  int max1 = 0;
  int max2 = 0;
  int max3 = 0;

  for (int i = 0; i < L; i++) {
    const int s = histo[i].size();
    if (s > max1) {
      max3 = max2;
      max2 = max1;
      max1 = s;
      ind3 = ind2;
      ind2 = ind1;
      ind1 = i;
    } else if (s > max2) {
      max3 = max2;
      max2 = s;
      ind3 = ind2;
      ind2 = i;
    } else if (s > max3) {
      max3 = s;
      ind3 = i;
    }
  }

  if (max2 < 0.1f * (float)max1) {
    ind2 = -1;
    ind3 = -1;
  } else if (max3 < 0.1f * (float)max1) {
    ind3 = -1;
  }
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
  const int *pa = a.ptr<int32_t>();
  const int *pb = b.ptr<int32_t>();

  int dist = 0;

  for (int i = 0; i < 8; i++, pa++, pb++) {
    unsigned int v = *pa ^ *pb;
    v              = v - ((v >> 1) & 0x55555555);
    v              = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
  }

  return dist;
}

}  // namespace ORB_SLAM2
