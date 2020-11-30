#include "manifold.h"
#include <stdexcept>

template<class T>
void ignore(const T&)
{}

Manifold::Manifold(std::vector<double> x, std::vector<int> t, std::vector<std::vector<double>> extras,
                   std::vector<bool> filter, size_t E, double dtweight, double missing)
  : _x(x)
  , _t(t)
  , _extras(extras)
  , _dtweight(dtweight)
  , _missing(missing)
{
  _use_dt = (dtweight > 0);
  _E_x = E;
  _E_dt = (_use_dt) * (E - 1);
  _E_extras = extras.size();
  _E_actual = _E_x + _E_dt + _E_extras;

  _full_t = (t.back() - t.front()) == (t.size() - 1);

  set_filter(filter);
}

void Manifold::set_filter(std::vector<bool> filter)
{
  _filter = filter;

  _filtered_t.clear();
  _timeToIndex.clear();

  _nobs = 0;
  for (size_t i = 0; i < filter.size(); i++) {
    if (filter[i]) {
      _filtered_t.push_back(_t[i]);
      _nobs += 1;
    }
    _timeToIndex[_t[i]] = i;
  }

  compute_lagged_embedding();
}

size_t Manifold::time_to_index(int time) const
{
  if (_full_t) {
    return (time - _t[0]);
  } else {
    return _timeToIndex.at(time);
  }
}

int Manifold::obs_num_to_time(size_t obsNum) const
{
  return _filtered_t.at(obsNum);
}

double Manifold::find_x(size_t i, size_t j) const
{
  try {
    int referenceTime = obs_num_to_time(i);
    size_t index;
    if (_use_dt) {
      index = time_to_index(referenceTime) - j;
    } else {
      index = time_to_index(referenceTime - (int)j);
    }
    return _x.at(index);
  } catch (const std::out_of_range& e) {
    ignore(e);
    return _missing;
  }
}

double Manifold::find_dt(size_t i, size_t j) const
{
  try {
    int referenceTime = obs_num_to_time(i);
    size_t index = time_to_index(referenceTime) - j;
    return _dtweight * (_t.at(index) - _t.at(index - 1));
  } catch (const std::out_of_range& e) {
    ignore(e);
    return _missing;
  }
}

double Manifold::find_extras(size_t i, size_t j) const
{
  try {
    int referenceTime = obs_num_to_time(i);
    size_t index = time_to_index(referenceTime);
    return _extras.at(j).at(index);
  } catch (const std::out_of_range& e) {
    ignore(e);
    return _missing;
  }
}

void Manifold::compute_lagged_embedding()
{
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

void Manifold::compute_lagged_embeddings()
{
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

double Manifold::x(size_t i, size_t j) const
{
  return _x_flat[i * _E_x + j];
}

double Manifold::dt(size_t i, size_t j) const
{
  return _dt_flat[i * _E_dt + j];
}

double Manifold::extras(size_t i, size_t j) const
{
  return _extras_flat[i * _E_extras + j];
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