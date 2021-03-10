#include "manifold.h"

Manifold ManifoldGenerator::create_manifold(size_t E, const std::vector<bool>& filter, bool prediction) const
{
  std::vector<size_t> inds;
  std::vector<double> y;

  size_t nobs = 0;
  for (size_t i = 0; i < filter.size(); i++) {
    if (filter[i]) {
      inds.push_back(i);
      y.push_back(_y[i]);
      nobs += 1;
    }
  }

  auto flat = std::make_unique<double[]>(nobs * E_actual(E));
  // auto missingMasks = std::make_unique<boost::dynamic_bitset<>[]>(nobs);
  auto missingMasks = std::make_unique<std::bitset<MAX_MANIFOLD_SIZE>[]>(nobs);

  // for (size_t i = 0; i < nobs; i++) {
  //   missingMasks[i].resize(E_actual(E));
  // }

  // Fill in the lagged embedding of x (or co_x) in the first columns
  for (size_t i = 0; i < nobs; i++) {
    for (size_t j = 0; j < E; j++) {
      if (prediction && _copredict) {
        flat[i * E_actual(E) + j] = lagged(_co_x, inds, i, j);
      } else {
        flat[i * E_actual(E) + j] = lagged(_x, inds, i, j);
      }
      missingMasks[i].set(j, flat[i * E_actual(E) + j] == _missing);
    }
  }

  // Put the lagged embedding of dt in the next columns
  for (size_t i = 0; i < nobs; i++) {
    for (size_t j = 0; j < E_dt(E); j++) {
      flat[i * E_actual(E) + E + j] = find_dt(inds, i, j);
      missingMasks[i].set(E + j, flat[i * E_actual(E) + E + j] == _missing);
    }
  }

  // Finally put the extras in the last columns
  for (size_t i = 0; i < nobs; i++) {
    int offset = 0;
    for (size_t k = 0; k < _num_extras; k++) {
      int numLags = _extrasEVarying[k] ? E : 1;
      for (size_t j = 0; j < numLags; j++) {
        flat[i * E_actual(E) + E + E_dt(E) + offset + j] = lagged(_extras[k], inds, i, j);
        missingMasks[i].set(E + E_dt(E) + offset + j, flat[i * E_actual(E) + E + E_dt(E) + offset + j] == _missing);
      }
      offset += numLags;
    }
  }

  return { flat, missingMasks, y, nobs, E, E_dt(E), E_extras(E), E_actual(E), _missing };
}

double ManifoldGenerator::lagged(const std::vector<double>& vec, const std::vector<size_t>& inds, size_t i,
                                 size_t j) const
{
  int index = inds.at(i) - j * _tau;
  if (index < 0) {
    return _missing;
  }
  return vec[index];
}

double ManifoldGenerator::find_dt(const std::vector<size_t>& inds, size_t i, size_t j) const
{
  int ind1 = inds.at(i) + _add_dt0 * _tau - j * _tau;
  int ind2 = ind1 - _tau;

  if ((ind1 >= _t.size()) || (ind2 < 0) || (_t[ind1] == _missing) || (_t[ind2] == _missing) || (_t[ind1] < _t[ind2])) {
    return _missing;
  }
  return _dtWeight * (_t[ind1] - _t[ind2]);
}