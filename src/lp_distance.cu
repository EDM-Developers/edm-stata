#include <cstdio>

#include "lp_distance.cuh"

#include <algorithm>
#include <cub/cub.cuh>

#define MISSING 1.0e+100
#define divup(a, b) (((a) + (b)-1) / (b))

template<typename T, bool isDMAE, int BLOCK_DIM_X>
__global__
void lpDistances(char * const valids, T * const distances,
                 const int npreds, const bool isPanelMode,
                 const double idw, const double missingDistance,
                 const int eacts, const int mnobs,
                 const T* mData, const int* mPanelIds,
                 const T* mpData, const int* mpPanelIds,
                 const char* mopts)
{
  using DistReduce = cub::BlockReduce<T, BLOCK_DIM_X>;
  using MsngReduce = cub::BlockReduce<bool, BLOCK_DIM_X>;

  //TODO const int p = blockIdx.y + blockIdx.z * gridDim.y; //nth prediction
  const int p = blockIdx.y; //nth prediction

  if (p < npreds) {
    __shared__ typename DistReduce::TempStorage dtemp;
    __shared__ typename MsngReduce::TempStorage mtemp;

    const bool imdoZero = missingDistance == 0;

    const int nob = blockIdx.x;

    const T* predsM  = mData + nob * eacts;
    const T* predsMp = mpData + p * eacts;

    bool anyEAmissing = false;
    T dist_i = T(0);

    if (threadIdx.x == 0 && isPanelMode && idw > 0) {
      dist_i += (idw * (mPanelIds[nob] != mpPanelIds[p]));
    }
    for (int x = threadIdx.x; x < eacts; x += blockDim.x)
    {
      T M_ij    = predsM[x];
      T Mp_ij   = predsMp[x];
      bool mopt = mopts[x];
      bool msng = (M_ij == MISSING || Mp_ij == MISSING);
      T diffM   = M_ij - Mp_ij;
      T compM   = M_ij != Mp_ij;
      T distM   = mopt * diffM + (1 - mopt) * compM;
      T dist_ij = msng * (1 - imdoZero) * missingDistance + (1 - msng) * distM;

      if (isDMAE) {
        dist_i += abs(dist_ij) / eacts;
      } else {
        dist_i += dist_ij * dist_ij;
      }
      anyEAmissing = (anyEAmissing || msng);
    }
    __syncthreads();

    dist_i = DistReduce(dtemp).Sum(dist_i);
    anyEAmissing = MsngReduce(mtemp).Sum(int(anyEAmissing)) > 0;

    if (threadIdx.x == 0) {
      anyEAmissing = anyEAmissing && imdoZero;

      dist_i = anyEAmissing * MISSING + (1 - anyEAmissing) * dist_i;

      bool isValid = dist_i != 0 && dist_i != MISSING;

      dist_i = (isDMAE * dist_i + (1 - isDMAE) * sqrt(dist_i));

      valids[nob + p * mnobs] = (char)isValid;
      distances[nob + p * mnobs] = dist_i;
    }
  }
}

// For 32 bit integers
unsigned int powerOf2LE(unsigned int value)
{
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;

  return value ^ (value >> 1);
}

template<typename T, bool isDMAE>
void lpDistances(char * const valids, T * const distances,
                 const int npreds, const bool isPanelMode,
                 const double idw, const double missingDistance,
                 const int eacts, const int mnobs,
                 const T* mData, const int* mPanelIds,
                 const T* mpData, const int* mpPanelIds,
                 const char* mopts, const cudaStream_t stream)
{
  dim3 threads( powerOf2LE(eacts) );

  threads.x = threads.x > 1024 ? 1024 : threads.x;

  dim3 blocks(mnobs, npreds);

  switch(threads.x) {
    case 1024:
      lpDistances<T, isDMAE, 1024> <<<blocks, threads, 0, stream>>>(
              valids, distances, npreds, isPanelMode, idw, missingDistance, eacts, mnobs,
              mData, mPanelIds, mpData, mpPanelIds, mopts);
      break;
    case 512:
      lpDistances<T, isDMAE, 512> <<<blocks, threads, 0, stream>>>(
              valids, distances, npreds, isPanelMode, idw, missingDistance, eacts, mnobs,
              mData, mPanelIds, mpData, mpPanelIds, mopts);
      break;
    case 256:
      lpDistances<T, isDMAE, 256> <<<blocks, threads, 0, stream>>>(
              valids, distances, npreds, isPanelMode, idw, missingDistance, eacts, mnobs,
              mData, mPanelIds, mpData, mpPanelIds, mopts);
      break;
    case 128:
      lpDistances<T, isDMAE, 128> <<<blocks, threads, 0, stream>>>(
              valids, distances, npreds, isPanelMode, idw, missingDistance, eacts, mnobs,
              mData, mPanelIds, mpData, mpPanelIds, mopts);
      break;
    case 64:
      lpDistances<T, isDMAE, 64> <<<blocks, threads, 0, stream>>>(
              valids, distances, npreds, isPanelMode, idw, missingDistance, eacts, mnobs,
              mData, mPanelIds, mpData, mpPanelIds, mopts);
      break;
    case 32:
      lpDistances<T, isDMAE, 32> <<<blocks, threads, 0, stream>>>(
              valids, distances, npreds, isPanelMode, idw, missingDistance, eacts, mnobs,
              mData, mPanelIds, mpData, mpPanelIds, mopts);
      break;
    case 16:
      lpDistances<T, isDMAE, 16> <<<blocks, threads, 0, stream>>>(
              valids, distances, npreds, isPanelMode, idw, missingDistance, eacts, mnobs,
              mData, mPanelIds, mpData, mpPanelIds, mopts);
      break;
    default:
      lpDistances<T, isDMAE, 8> <<<blocks, threads, 0, stream>>>(
              valids, distances, npreds, isPanelMode, idw, missingDistance, eacts, mnobs,
              mData, mPanelIds, mpData, mpPanelIds, mopts);
      break;
  }
}

template<typename T>
void cuLPDistances(char * const valids, T * const distances,
                   const int npreds, const bool isDMAE, const bool isPanelMode,
                   const double idw, const double missingDistance,
                   const int eacts, const int mnobs,
                   const T* mData, const int* mPanelIds,
                   const T* mpData, const int* mpPanelIds,
                   const char* mopts, const cudaStream_t stream)
{
  if (isDMAE) {
    lpDistances<T, true>(valids, distances, npreds, isPanelMode, idw, missingDistance, eacts, mnobs,
            mData, mPanelIds, mpData, mpPanelIds, mopts, stream);
  } else {
    lpDistances<T, false>(valids, distances, npreds, isPanelMode, idw, missingDistance, eacts, mnobs,
            mData, mPanelIds, mpData, mpPanelIds, mopts, stream);
  }
}

#define INSTANTIATE(T)                                                                      \
template void cuLPDistances(char * const, T * const, const int, const bool, const bool,     \
        const double, const double, const int, const int, const T*, const int*, const T*,   \
        const int*, const char*, const cudaStream_t);

//INSTANTIATE(float)
INSTANTIATE(double)
