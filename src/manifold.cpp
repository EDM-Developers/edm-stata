#include "manifold.h"

Manifold ManifoldGenerator::create_manifold(size_t E, const std::vector<bool>& filter, bool prediction) const
{
  size_t E_dt = (_use_dt) * (E - 1);
  size_t E_actual = E + E_dt + _E_extras;

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

  auto flat = std::make_unique<double[]>(nobs * E_actual);

  // Fill in the lagged embedding of x (or co_x) in the first columns
  for (size_t i = 0; i < nobs; i++) {
    for (size_t j = 0; j < E; j++) {
      if (prediction && _copredict) {
        flat[i * E_actual + j] = find_co_x(inds, i, j);
      } else {
        flat[i * E_actual + j] = find_x(inds, i, j);
      }
    }
  }

  // Put the lagged embedding of dt in the next columns
  for (size_t i = 0; i < nobs; i++) {
    for (size_t j = 0; j < E_dt; j++) {
      flat[i * E_actual + E + j] = find_dt(inds, i, j);
    }
  }

  // Finally put the unlagged extras in the last columns
  for (size_t i = 0; i < nobs; i++) {
    for (size_t j = 0; j < _E_extras; j++) {
      flat[i * E_actual + E + E_dt + j] = find_extras(inds, i, j);
    }
  }

  return { flat, y, nobs, E, E_dt, _E_extras, E_actual, _missing };
}

double ManifoldGenerator::find_x(const std::vector<size_t>& inds, size_t i, size_t j) const
{
  size_t index = inds.at(i);
  if (index < j) {
    return _missing;
  }
  return _x[index - j];
}

double ManifoldGenerator::find_co_x(const std::vector<size_t>& inds, size_t i, size_t j) const
{
  size_t index = inds.at(i);
  if (index < j) {
    return _missing;
  }
  return _co_x[index - j];
}

double ManifoldGenerator::find_dt(const std::vector<size_t>& inds, size_t i, size_t j) const
{
  size_t index = inds.at(i);
  if (index < j + 1 || _t[index - 1] == _missing || _t[index] == _missing) {
    return _missing;
  }
  index -= j;
  return _dtWeight * (_t[index] - _t[index - 1]);
}

double ManifoldGenerator::find_extras(const std::vector<size_t>& inds, size_t i, size_t j) const
{
  size_t index = inds.at(i);
  return _extras.at(j).at(index);
}