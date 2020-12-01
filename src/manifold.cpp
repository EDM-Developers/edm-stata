#include "manifold.h"

Manifold::Manifold(std::vector<double> x, std::vector<double> t, std::vector<std::vector<double>> extras,
                   std::vector<bool> filter, size_t E, double dtWeight, double missing)
  : _x(x)
  , _t(t)
  , _extras(extras)
  , _dtWeight(dtWeight)
  , _missing(missing)
{
  _use_dt = (dtWeight > 0);
  _E_x = E;
  _E_dt = (_use_dt) * (E - 1);
  _E_extras = extras.size();
  _E_actual = _E_x + _E_dt + _E_extras;

  set_filter(filter);
}

void Manifold::set_filter(std::vector<bool> filter)
{
  _filter = filter;
  _filteredIndices.clear();

  _nobs = 0;
  for (size_t i = 0; i < filter.size(); i++) {
    if (filter[i]) {
      _filteredIndices.push_back(i);
      _nobs += 1;
    }
  }

  // Any lagged embeddings we previously computed are now invalid
  _x_flat.clear();
  _dt_flat.clear();
  _extras_flat.clear();
  _combined_flat.clear();
}

double Manifold::find_x(size_t i, size_t j) const
{
  size_t index = _filteredIndices.at(i);
  if (index < j) {
    return _missing;
  }
  return _x[index - j];
}

double Manifold::find_dt(size_t i, size_t j) const
{
  size_t index = _filteredIndices.at(i);
  if (index < j + 1 || _t[index - 1] == _missing || _t[index] == _missing) {
    return _missing;
  }
  index -= j;
  return _dtWeight * (_t[index] - _t[index - 1]);
}

double Manifold::find_extras(size_t i, size_t j) const
{
  size_t index = _filteredIndices.at(i);
  return _extras.at(j).at(index);
}

void Manifold::compute_lagged_embedding() const
{
  if (_combined_flat.size() > 0) {
    return;
  }
  _combined_flat = std::vector<double>(_nobs * _E_actual);

  for (size_t i = 0; i < _nobs; i++) {
    for (size_t j = 0; j < _E_x; j++) {
      _combined_flat[i * _E_actual + j] = find_x(i, j);
    }
  }

  for (size_t i = 0; i < _nobs; i++) {
    for (size_t j = 0; j < _E_dt; j++) {
      _combined_flat[i * _E_actual + j + _E_x] = find_dt(i, j);
    }
  }

  for (size_t i = 0; i < _nobs; i++) {
    for (size_t j = 0; j < _E_extras; j++) {
      _combined_flat[i * _E_actual + j + _E_x + _E_dt] = find_extras(i, j);
    }
  }
}

void Manifold::compute_lagged_embeddings() const
{
  if (_x_flat.size() > 0) {
    return;
  }
  _x_flat = std::vector<double>(_nobs * _E_x);

  for (size_t i = 0; i < _nobs; i++) {
    for (size_t j = 0; j < _E_x; j++) {
      _x_flat[i * _E_x + j] = find_x(i, j);
    }
  }

  _dt_flat = std::vector<double>(_nobs * _E_dt);

  for (size_t i = 0; i < _nobs; i++) {
    for (size_t j = 0; j < _E_dt; j++) {
      _dt_flat[i * _E_dt + j] = find_dt(i, j);
    }
  }

  _extras_flat = std::vector<double>(_nobs * _E_extras);

  for (size_t i = 0; i < _nobs; i++) {
    for (size_t j = 0; j < _E_extras; j++) {
      _extras_flat[i * _E_extras + j] = find_extras(i, j);
    }
  }
}

bool Manifold::any_missing(size_t obsNum) const
{
  bool missing = false;
  for (size_t j = 0; j < _E_actual; j++) {
    if (operator()(obsNum, j) == _missing) {
      missing = true;
      break;
    }
  }
  return missing;
}