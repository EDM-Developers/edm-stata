#include "lp_distance.cuh"

#include <algorithm>
#include <cub/cub.cuh>

#define BLOCK_DIM_X 32
#define MISSING 1.0e+100
#define divup(a, b) (((a) + (b)-1) / (b))

// reduceAlongDimY requires a maximum blockDim.y value of BLOCK_DIM_X to function correctly
// Currently, that is max number of E_actuals in a given manifold expected as of now
template<typename T>
__device__ T reduceAlongDimY(T value, T* workspace) {
  __syncthreads(); // Sync to ensure unfinished accesses don't effect downstream ops

  int loc = threadIdx.x * blockDim.y + threadIdx.y;

  workspace[ loc ] = value;
  __syncthreads();

  for (int rsize = blockDim.y / 2; rsize > 0; rsize /= 2) {
    if (threadIdx.y < rsize) {
      workspace[ loc ] += workspace[ loc + rsize ];
    }
    __syncthreads();
  }
  value = workspace[threadIdx.x * blockDim.y];
  __syncthreads();

  return value; // each nob's reduction result
}

template<typename T, bool isDMAE, int BLOCK_DIM_Y>
__global__
void lpDistances(char * const valids, T * const distances,
                 const int npreds, const bool isPanelMode,
                 const double idw, const double missingDistance,
                 const int eacts, const int mnobs,
                 const T* mData, const int* mPanelIds,
                 const T* mpData, const int* mpPanelIds,
                 const char* mopts)
{
  const int p = blockIdx.y; //nth prediction

  if (p < npreds)
  {
    __shared__ T dists[BLOCK_DIM_X * BLOCK_DIM_Y];
    __shared__ bool markers[BLOCK_DIM_X * BLOCK_DIM_Y];

    const bool isZero = (missingDistance == 0);
    const T* predsMp  = mpData + p * eacts;
    const int nob     = blockDim.x * blockIdx.x + threadIdx.x;

    if (nob < mnobs)
    {
      const T* predsM   = mData + nob * eacts;
      bool anyEAmissing = false;
      T dist_i          = T(0);

      if ( threadIdx.y == 0 && isPanelMode && idw > 0 ) {
        dist_i += (idw * (mPanelIds[nob] != mpPanelIds[p]));
      }
      for (int e = threadIdx.y; e < eacts; e += blockDim.y)
      {
        T M_ij    = predsM[e];
        T Mp_ij   = predsMp[e];
        bool mopt = mopts[e];
        bool msng = (M_ij == MISSING || Mp_ij == MISSING);
        T diffM   = M_ij - Mp_ij;
        T compM   = M_ij != Mp_ij;
        T distM   = mopt * diffM + (1 - mopt) * compM;
        T dist_ij = msng * (1 - isZero) * missingDistance + (1 - msng) * distM;

        if (isDMAE) {
          dist_i += abs(dist_ij) / eacts;
        } else {
          dist_i += dist_ij * dist_ij;
        }
        anyEAmissing = (anyEAmissing || msng);
      }
      __syncthreads();

      dist_i = reduceAlongDimY(dist_i, dists);
      anyEAmissing = reduceAlongDimY(anyEAmissing, markers);

      if (threadIdx.y == 0) {
        anyEAmissing = anyEAmissing && isZero;

        dist_i = anyEAmissing * MISSING + (1 - anyEAmissing) * dist_i;

        bool isValid = dist_i != 0 && dist_i != MISSING;

        dist_i = (isDMAE * dist_i + (1 - isDMAE) * sqrt(dist_i));

        valids[nob + p * mnobs] = (char)isValid;
        distances[nob + p * mnobs] = dist_i;
      }
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
  dim3 threads(BLOCK_DIM_X, powerOf2LE(eacts));

  threads.y = threads.y > 32 ? 32 : threads.y;

  dim3 blocks(divup(mnobs, threads.x), npreds);

  //printf("eacts %d mnobs %d npreds %d \n", eacts, mnobs, npreds);
  //printf("grid %d, %d block %d, %d \n", blocks.x, blocks.y, threads.x, threads.y);

  switch(threads.y) {
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
    case 8:
      lpDistances<T, isDMAE, 8> <<<blocks, threads, 0, stream>>>(
              valids, distances, npreds, isPanelMode, idw, missingDistance, eacts, mnobs,
              mData, mPanelIds, mpData, mpPanelIds, mopts);
      break;
    case 4:
      lpDistances<T, isDMAE, 4> <<<blocks, threads, 0, stream>>>(
              valids, distances, npreds, isPanelMode, idw, missingDistance, eacts, mnobs,
              mData, mPanelIds, mpData, mpPanelIds, mopts);
      break;
    default:
      lpDistances<T, isDMAE, 2> <<<blocks, threads, 0, stream>>>(
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
