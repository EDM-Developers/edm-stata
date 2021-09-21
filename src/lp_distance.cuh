#pragma once

template<typename T>
void cuLPDistances(char* const valids, T* const distances, const int npreds, const bool isDistanceMeanAbsoluteError,
                   const bool isPanelMode, const double idw, const double missingDistance, const int eacts,
                   const int mnobs, const T* mData, const int* mPanelIds, const T* mpData, const int* mpPanelIds,
                   const char* metricOptions, const cudaStream_t);
