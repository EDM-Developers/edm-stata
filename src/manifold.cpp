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
  set_filter(filter);
}

void Manifold::set_filter(std::vector<bool> filter)
{
  _filter = filter;

  _obsNumToTime.clear();
  _timeToIndex.clear();

  _nobs = 0;
  for (size_t i = 0; i < filter.size(); i++) {
    if (filter[i]) {
      _obsNumToTime[_nobs++] = _t[i];
    }
    _timeToIndex[_t[i]] = i;
  }
}

double Manifold::x(size_t i, size_t j) const
{
  try {
    int referenceTime = _obsNumToTime.at(i);
    size_t index;
    if (_use_dt) {
      index = _timeToIndex.at(referenceTime) - j;
    } else {
      index = _timeToIndex.at(referenceTime - (int)j);
    }
    return _x.at(index);
  } catch (const std::out_of_range& e) {
    ignore(e);
    return _missing;
  }
}

double Manifold::dt(size_t i, size_t j) const
{
  try {
    int referenceTime = _obsNumToTime.at(i);
    size_t index = _timeToIndex.at(referenceTime) - j;
    return _dtweight * (_t.at(index) - _t.at(index - 1));
  } catch (const std::out_of_range& e) {
    ignore(e);
    return _missing;
  }
}

double Manifold::extras(size_t i, size_t j) const
{
  try {
    int referenceTime = _obsNumToTime.at(i);
    size_t index = _timeToIndex.at(referenceTime);
    return _extras.at(j).at(index);
  } catch (const std::out_of_range& e) {
    ignore(e);
    return _missing;
  }
}

double Manifold::operator()(size_t i, size_t j) const
{
  if (j < _E_x) {
    return x(i, j);
  } else if (j < _E_x + _E_dt) {
    return dt(i, j - _E_x);
  } else {
    return extras(i, j - (_E_x + _E_dt));
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