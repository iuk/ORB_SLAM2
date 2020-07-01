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

#include "MapPoint.h"

#include <mutex>

#include "ORBmatcher.h"

namespace ORB_SLAM2 {

long unsigned int MapPoint::nNextId = 0;
mutex MapPoint::mGlobalMutex;

/**
 * @brief 给定坐标与keyframe构造MapPoint
 *
 * 双目：StereoInitialization()，CreateNewKeyFrame()，LocalMapping::CreateNewMapPoints()
 * 单目：CreateInitialMapMonocular()，LocalMapping::CreateNewMapPoints()
 * @param Pos    MapPoint的坐标（wrt世界坐标系）
 * @param pRefKF KeyFrame
 * @param pMap   Map
 */
MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap)
    : mnFirstKFid(pRefKF->mnId),
      mnFirstFrame(pRefKF->mnFrameId),
      nObs(0),
      mnTrackReferenceForFrame(0),
      mnLastFrameSeen(0),
      mnBALocalForKF(0),
      mnFuseCandidateForKF(0),
      mnLoopPointForKF(0),
      mnCorrectedByKF(0),
      mnCorrectedReference(0),
      mnBAGlobalForKF(0),
      mpRefKF(pRefKF),
      mnVisible(1),
      mnFound(1),
      mbBad(false),
      mpReplaced(static_cast<MapPoint *>(NULL)),
      mfMinDistance(0),
      mfMaxDistance(0),
      mpMap(pMap) {
  Pos.copyTo(mWorldPos);
  mNormalVector = cv::Mat::zeros(3, 1, CV_32F);

  // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
  unique_lock<mutex> lock(mpMap->mMutexPointCreation);
  mnId = nNextId++;
}

/**
 * @brief 给定坐标与frame构造MapPoint
 *
 * 双目：UpdateLastFrame()
 * @param Pos    MapPoint的坐标（wrt世界坐标系）
 * @param pMap   Map
 * @param pFrame Frame
 * @param idxF   MapPoint在Frame中的索引，即对应的特征点的编号
 */
MapPoint::MapPoint(const cv::Mat &Pos, Map *pMap, Frame *pFrame, const int &idxF)
    : mnFirstKFid(-1),
      mnFirstFrame(pFrame->mnId),
      nObs(0),
      mnTrackReferenceForFrame(0),
      mnLastFrameSeen(0),
      mnBALocalForKF(0),
      mnFuseCandidateForKF(0),
      mnLoopPointForKF(0),
      mnCorrectedByKF(0),
      mnCorrectedReference(0),
      mnBAGlobalForKF(0),
      mpRefKF(static_cast<KeyFrame *>(NULL)),
      mnVisible(1),
      mnFound(1),
      mbBad(false),
      mpReplaced(NULL),
      mpMap(pMap) {
  Pos.copyTo(mWorldPos);
  cv::Mat Ow    = pFrame->GetCameraCenter();
  mNormalVector = mWorldPos - Ow;
  mNormalVector = mNormalVector / cv::norm(mNormalVector);

  cv::Mat PC                   = Pos - Ow;
  const float dist             = cv::norm(PC);
  const int level              = pFrame->mvKeysUn[idxF].octave;
  const float levelScaleFactor = pFrame->mvScaleFactors[level];
  const int nLevels            = pFrame->mnScaleLevels;

  mfMaxDistance = dist * levelScaleFactor;
  mfMinDistance = mfMaxDistance / pFrame->mvScaleFactors[nLevels - 1];

  pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

  // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
  unique_lock<mutex> lock(mpMap->mMutexPointCreation);
  mnId = nNextId++;
}

void MapPoint::SetWorldPos(const cv::Mat &Pos) {
  unique_lock<mutex> lock2(mGlobalMutex);
  unique_lock<mutex> lock(mMutexPos);
  Pos.copyTo(mWorldPos);
}

cv::Mat MapPoint::GetWorldPos() {
  unique_lock<mutex> lock(mMutexPos);
  return mWorldPos.clone();
}

cv::Mat MapPoint::GetNormal() {
  unique_lock<mutex> lock(mMutexPos);
  return mNormalVector.clone();
}

KeyFrame *MapPoint::GetReferenceKeyFrame() {
  unique_lock<mutex> lock(mMutexFeatures);
  return mpRefKF;
}

/**
 * @brief 添加观测
 *
 * 记录哪些KeyFrame的那个特征点能观测到该MapPoint \n
 * 并增加观测的相机数目nObs，单目+1，双目或者grbd+2
 * 这个函数是建立关键帧共视关系的核心函数，能共同观测到某些MapPoints的关键帧是共视关键帧
 * @param pKF KeyFrame
 * @param idx MapPoint在KeyFrame中的索引
 */
void MapPoint::AddObservation(KeyFrame *pKF, size_t idx) {
  unique_lock<mutex> lock(mMutexFeatures);
  // 一个 map point 只能被一个 keyframe 观测到一次
  if (mObservations.count(pKF)) // 如果已经统计过了，跳过
    return;
  // 记录下能观测到该MapPoint的KF和该MapPoint在KF中的索引
  mObservations[pKF] = idx;

  if (pKF->mvuRight[idx] >= 0)
    nObs += 2;  // 双目或者grbd
  else
    nObs++;  // 单目
}

void MapPoint::EraseObservation(KeyFrame *pKF) {
  bool bBad = false;
  {
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF)) {
      int idx = mObservations[pKF];
      if (pKF->mvuRight[idx] >= 0)
        nObs -= 2;
      else
        nObs--;

      mObservations.erase(pKF);

      if (mpRefKF == pKF)
        mpRefKF = mObservations.begin()->first;

      // If only 2 observations or less, discard point
      if (nObs <= 2)
        bBad = true;
    }
  }

  if (bBad)
    SetBadFlag();
}

map<KeyFrame *, size_t> MapPoint::GetObservations() {
  unique_lock<mutex> lock(mMutexFeatures);
  return mObservations;
}

int MapPoint::Observations() {
  unique_lock<mutex> lock(mMutexFeatures);
  return nObs;
}

void MapPoint::SetBadFlag() {
  map<KeyFrame *, size_t> obs;
  {
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    mbBad = true;
    obs   = mObservations;
    mObservations.clear();
  }
  for (map<KeyFrame *, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++) {
    KeyFrame *pKF = mit->first;
    pKF->EraseMapPointMatch(mit->second);
  }

  mpMap->EraseMapPoint(this);
}

MapPoint *MapPoint::GetReplaced() {
  unique_lock<mutex> lock1(mMutexFeatures);
  unique_lock<mutex> lock2(mMutexPos);
  return mpReplaced;
}

void MapPoint::Replace(MapPoint *pMP) {
  if (pMP->mnId == this->mnId)
    return;

  int nvisible, nfound;
  map<KeyFrame *, size_t> obs;
  {
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    obs = mObservations;
    mObservations.clear();
    mbBad      = true;
    nvisible   = mnVisible;
    nfound     = mnFound;
    mpReplaced = pMP;
  }

  for (map<KeyFrame *, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++) {
    // Replace measurement in keyframe
    KeyFrame *pKF = mit->first;

    if (!pMP->IsInKeyFrame(pKF)) {
      pKF->ReplaceMapPointMatch(mit->second, pMP);
      pMP->AddObservation(pKF, mit->second);
    } else {
      pKF->EraseMapPointMatch(mit->second);
    }
  }
  pMP->IncreaseFound(nfound);
  pMP->IncreaseVisible(nvisible);
  pMP->ComputeDistinctiveDescriptors();

  mpMap->EraseMapPoint(this);
}

bool MapPoint::isBad() {
  unique_lock<mutex> lock(mMutexFeatures);
  unique_lock<mutex> lock2(mMutexPos);
  return mbBad;
}

void MapPoint::IncreaseVisible(int n) {
  unique_lock<mutex> lock(mMutexFeatures);
  mnVisible += n;
}

void MapPoint::IncreaseFound(int n) {
  unique_lock<mutex> lock(mMutexFeatures);
  mnFound += n;
}

float MapPoint::GetFoundRatio() {
  unique_lock<mutex> lock(mMutexFeatures);
  return static_cast<float>(mnFound) / mnVisible;
}

/**
 * @brief 计算具有代表的描述子
 *
 * 由于一个MapPoint会被许多相机观测到，因此在插入关键帧后，需要判断是否更新当前点的最适合的描述子 \n
 * 先获得当前点的所有描述子，然后计算描述子之间的两两距离，最好的描述子与其他描述子应该具有最小的距离中值
 * @see III - C3.3
 */
void MapPoint::ComputeDistinctiveDescriptors() {
  // Retrieve all observed descriptors
  vector<cv::Mat> vDescriptors;

  map<KeyFrame *, size_t> observations;

  {
    unique_lock<mutex> lock1(mMutexFeatures);
    if (mbBad)
      return;
    observations = mObservations;
  }

  if (observations.empty())
    return;

  vDescriptors.reserve(observations.size());

  // 遍历观测到3d点的所有关键帧，获得orb描述子，并插入到vDescriptors中
  for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
    KeyFrame *pKF = mit->first;

    if (!pKF->isBad())
      vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
  }

  if (vDescriptors.empty())
    return;

  // Compute distances between them
  // 获得这些描述子两两之间的距离
  const size_t N = vDescriptors.size();

  float Distances[N][N];
  for (size_t i = 0; i < N; i++) {
    Distances[i][i] = 0;
    for (size_t j = i + 1; j < N; j++) {
      int distij      = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
      Distances[i][j] = distij;
      Distances[j][i] = distij;
    }
  }

  // Take the descriptor with least median distance to the rest
  int BestMedian = INT_MAX;
  int BestIdx    = 0;
  for (size_t i = 0; i < N; i++) {
    // 第i个描述子到其它所有所有描述子之间的距离
    vector<int> vDists(Distances[i], Distances[i] + N);
    sort(vDists.begin(), vDists.end());

    // 获得中值
    int median = vDists[0.5 * (N - 1)];

    // 寻找最小的中值
    if (median < BestMedian) {
      BestMedian = median;
      BestIdx    = i;
    }
  }

  {
    unique_lock<mutex> lock(mMutexFeatures);

    // 最好的描述子，该描述子相对于其他描述子有最小的距离中值
    // 简化来讲，中值代表了这个描述子到其它描述子的平均距离
    // 最好的描述子就是和其它描述子的平均距离最小
    mDescriptor = vDescriptors[BestIdx].clone();
  }
}

cv::Mat MapPoint::GetDescriptor() {
  unique_lock<mutex> lock(mMutexFeatures);
  return mDescriptor.clone();
}

int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF) {
  unique_lock<mutex> lock(mMutexFeatures);
  if (mObservations.count(pKF))
    return mObservations[pKF];
  else
    return -1;
}

bool MapPoint::IsInKeyFrame(KeyFrame *pKF) {
  unique_lock<mutex> lock(mMutexFeatures);
  return (mObservations.count(pKF));
}

/**
 * @brief 更新平均观测方向以及观测距离范围
 *
 * 由于一个MapPoint会被许多相机观测到，因此在插入关键帧后，需要更新相应变量
 * mNormalVector：3D点被观测的平均方向
 * mfMaxDistance：观测到该3D点的最大距离
 * mfMinDistance：观测到该3D点的最小距离
 * @see III - C2.2 c2.4
 */
void MapPoint::UpdateNormalAndDepth() {
  map<KeyFrame *, size_t> observations;
  KeyFrame *pRefKF;
  cv::Mat Pos;
  {
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    if (mbBad)
      return;

    observations = mObservations;      // 获得观测到该3d点的所有关键帧
    pRefKF       = mpRefKF;            // 观测到该点的参考关键帧
    Pos          = mWorldPos.clone();  // 3d点在世界坐标系中的位置
  }

  if (observations.empty())
    return;

  cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
  int n          = 0;
  for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
    KeyFrame *pKF   = mit->first;
    cv::Mat Owi     = pKF->GetCameraCenter();
    cv::Mat normali = mWorldPos - Owi;
    normal          = normal + normali / cv::norm(normali);  // 对所有关键帧对该点的观测方向归一化为单位向量进行求和
    n++;
  }

  cv::Mat PC                   = Pos - pRefKF->GetCameraCenter();  // 参考关键帧相机指向3D点的向量（在世界坐标系下的表示）
  const float dist             = cv::norm(PC);                     // 该点到参考关键帧相机的距离
  const int level              = pRefKF->mvKeysUn[observations[pRefKF]].octave;
  const float levelScaleFactor = pRefKF->mvScaleFactors[level];
  const int nLevels            = pRefKF->mnScaleLevels;  // 金字塔层数

  {
    unique_lock<mutex> lock3(mMutexPos);
    // 另见PredictScale函数前的注释
    mfMaxDistance = dist * levelScaleFactor;                              // 观测到该点的距离最大值
    mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];  // 观测到该点的距离最小值
    mNormalVector = normal / n;                                           // 获得平均的观测方向
  }
}

float MapPoint::GetMinDistanceInvariance() {
  unique_lock<mutex> lock(mMutexPos);
  return 0.8f * mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance() {
  unique_lock<mutex> lock(mMutexPos);
  return 1.2f * mfMaxDistance;
}

int MapPoint::PredictScale(const float &currentDist, KeyFrame *pKF) {
  float ratio;
  {
    unique_lock<mutex> lock(mMutexPos);
    ratio = mfMaxDistance / currentDist;
  }

  int nScale = ceil(log(ratio) / pKF->mfLogScaleFactor);
  if (nScale < 0)
    nScale = 0;
  else if (nScale >= pKF->mnScaleLevels)
    nScale = pKF->mnScaleLevels - 1;

  return nScale;
}

int MapPoint::PredictScale(const float &currentDist, Frame *pF) {
  float ratio;
  {
    unique_lock<mutex> lock(mMutexPos);
    ratio = mfMaxDistance / currentDist;
  }

  int nScale = ceil(log(ratio) / pF->mfLogScaleFactor);
  if (nScale < 0)
    nScale = 0;
  else if (nScale >= pF->mnScaleLevels)
    nScale = pF->mnScaleLevels - 1;

  return nScale;
}

}  // namespace ORB_SLAM2
