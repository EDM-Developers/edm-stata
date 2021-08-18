#pragma warning(disable : 4018)

#include "manifold.h"

// Recursive function to return gcd of a and b
// Lifted from https://www.geeksforgeeks.org/program-find-gcd-floating-point-numbers/
double gcd(double a, double b)
{
  if (a < b)
    return gcd(b, a);

  // base case
  if (fabs(b) < 0.001)
    return a;

  else
    return (gcd(b, a - floor(a / b) * b));
}

double ManifoldGenerator::calculate_time_increment() const
{
  // Find the units which time is measured in.
  // E.g. if time variables are 1, 2, 3, ... then the 'unit' is 1
  // Whereas if time is like 1000, 2000, 4000, 20000 ... then the 'unit' is perhaps 1000.
  double unit = -1;

  // Go through the supplied time index and find the greatest common divisor of the differences between consecutive time
  // points.
  for (int i = 1; i < _t.size(); i++) {

    double timeDiff = _t[i] - _t[i - 1];

    // In the panel data case, we may get consecutive times which are negative at the boundary of panels.
    if (timeDiff <= 0 || _t[i] == MISSING || _t[i - 1] == MISSING) {
      continue;
    }

    // For the first time, just replace sentinel value with the time difference.
    if (unit < 0) {
      unit = timeDiff;
      continue;
    }

    unit = gcd(timeDiff, unit);
  }

  return unit;
}

void ManifoldGenerator::setup_observation_numbers()
{
  if (!_use_dt) {
    // In normal situations (non-dt)
    double unit = calculate_time_increment();
    double minT = *std::min_element(_t.begin(), _t.end());

    // Create a time index which is a discrete count of the number of 'unit' time units.
    for (int i = 0; i < _t.size(); i++) {
      if (_t[i] != MISSING) {
        _observation_number.push_back(std::round((_t[i] - minT) / unit));
      } else {
        _observation_number.push_back(-1);
      }
    }
  } else {
    // In 'dt' mode
    int countUp = 0;
    for (int i = 0; i < _t.size(); i++) {
      if (_t[i] != MISSING && (_allow_missing || (_x[i] != MISSING))) { // TODO: What about co_x missing here?
        _observation_number.push_back(countUp);
        countUp += 1;
      } else {
        _observation_number.push_back(-1);
      }
    }
  }
}

int ManifoldGenerator::get_observation_num(int i)
{
  return _observation_number[i];
}

bool ManifoldGenerator::find_observation_num(int target, int& k, int direction, int panel) const
{
  // Loop either forward or back until we find the right index or give up.
  while (k >= 0 && k < _observation_number.size()) {
    // If in panel mode, make sure we don't wander over a panel boundary.
    if (_panel_mode) {
      if (panel != _panel_ids[k]) {
        return false;
      }
    }

    // Skip over garbage rows which don't have a time recorded.
    if (_observation_number[k] < 0) {
      k += direction;
      continue;
    }

    // If we found the desired row at index k then stop here and report the success.
    if (_observation_number[k] == target) {
      return true;
    }

    // If we've gone past it & therefore this target doesn't exist, give up.
    if (direction > 0 && _observation_number[k] > target) {
      return false;
    }
    if (direction < 0 && _observation_number[k] < target) {
      return false;
    }

    k += direction;
  }

  return false;
}

std::vector<int> ManifoldGenerator::get_lagged_indices(int startIndex, int E, int panel) const
{

  std::vector<int> laggedIndices(E);
  std::fill_n(laggedIndices.begin(), E, -1);

  // For obs i, which indices correspond to looking back 0, tau, ..., (E-1)*tau observations.
  laggedIndices[0] = startIndex;
  int pointStartObsNum = _observation_number[startIndex];

  // Start by going back one index
  int k = startIndex - 1;

  for (int j = 1; j < E; j++) {
    // Find the discrete time we're searching for.
    int targetObsNum = pointStartObsNum - j * _tau;

    if (find_observation_num(targetObsNum, k, -1, panel)) {
      laggedIndices[j] = k;
    }
  }

  return laggedIndices;
}

Manifold ManifoldGenerator::create_manifold(int E, const std::vector<bool>& filter, bool copredict, bool prediction,
                                            bool skipMissing) const
{
  int nobs = 0;
  std::vector<int> pointNumToStartIndex;
  for (int i = 0; i < filter.size(); i++) {
    if (filter[i]) {
      pointNumToStartIndex.push_back(i);
      nobs += 1;
    }
  }

  auto flat = std::make_unique<double[]>(nobs * E_actual(E));

  std::vector<double> y;
  std::vector<int> panelIDs;

  // Fill in the manifold row-by-row (point-by-point)
  int M_i = 0;
  double target;

  for (int i = 0; i < nobs; i++) {
    double* point = &(flat[M_i * E_actual(E)]);
    fill_in_point(pointNumToStartIndex[i], E, copredict, prediction, point, &target);

    // Erase this point if we don't want missing values in the resulting manifold
    if (skipMissing) {
      bool foundMissing = false;
      for (int j = 0; j < E_actual(E); j++) {
        if (point[j] == MISSING) {
          foundMissing = true;
          break;
        }
      }

      if (foundMissing) {
        continue;
      }
    }

    y.push_back(target);
    if (_panel_mode) {
      panelIDs.push_back(_panel_ids[pointNumToStartIndex[i]]);
    }

    M_i += 1;
  }

  nobs = M_i;

  return { flat, y, panelIDs, nobs, E, E_dt(E), E_extras(E), E * numExtrasLagged(), E_actual(E) };
}

void ManifoldGenerator::fill_in_point(int i, int E, bool copredict, bool prediction, double* point,
                                      double* target) const
{
  int panel = _panel_mode ? _panel_ids[i] : -1;
  bool use_co_x = copredict && prediction;
  const std::vector<double>& yTS = (use_co_x ? _co_x : (_xmap_mode ? _xmap : _x));

  std::vector<int> laggedIndices = get_lagged_indices(i, E, panel);

  auto lookup_vec = [&laggedIndices](const std::vector<double>& vec, int j) {
    if (laggedIndices[j] < 0) {
      return MISSING;
    } else {
      return vec[laggedIndices[j]];
    }
  };

  // What is the target of this point in the manifold?
  int targetIndex = i;

  if (_p != 0) {
    // At what time does the prediction occur?
    int targetObsNum = _observation_number[targetIndex] + _p;

    int direction = _p > 0 ? 1 : -1;

    if (find_observation_num(targetObsNum, targetIndex, direction, panel)) {
      *target = yTS[targetIndex];
    } else {
      targetIndex = -1;
      *target = MISSING;
    }
  } else {
    *target = yTS[targetIndex];
  }

  // Fill in the lagged embedding of x (or co_x) in the first columns
  for (int j = 0; j < E; j++) {
    if (use_co_x) {
      point[j] = lookup_vec(_co_x, j);
    } else {
      point[j] = lookup_vec(_x, j);
    }
  }

  // TODO: Need to add back in the _cumulative_dt option

  // Put the lagged embedding of dt in the next columns
  if (E_dt(E) > 0) {
    // The first dt value is a bit special as it is relative to the
    // time of the corresponding y prediction.
    if (_p == 0) {
      point[E + 0] = 0; // Special case for contemporaneous predictions.
    } else {
      double tNow = lookup_vec(_t, 0);
      if (tNow != MISSING && targetIndex >= 0) {
        double tPred = _t[targetIndex];
        point[E + 0] = _dtWeight * (tPred - tNow);
      } else {
        point[E + 0] = MISSING;
      }
    }

    for (int j = 1; j < E_dt(E); j++) {
      double tNext = lookup_vec(_t, j - 1);
      double tNow = lookup_vec(_t, j);
      if (tNext != MISSING && tNow != MISSING) {
        point[E + j] = _dtWeight * (tNext - tNow);
      } else {
        point[E + j] = MISSING;
      }
    }
  }

  // Finally put the extras in the last columns
  int offset = 0;
  for (int k = 0; k < _num_extras; k++) {
    int numLags = (k < _num_extras_lagged) ? E : 1;
    for (int j = 0; j < numLags; j++) {
      point[E + E_dt(E) + offset + j] = lookup_vec(_extras[k], j);
    }
    offset += numLags;
  }
}

std::vector<bool> ManifoldGenerator::generate_usable(const std::vector<bool>& touse, int maxE) const
{
  bool copredict = false; // TODO: Need to handle coprediction's usable
  bool prediction = false;

  // Generate the 'usable' variable
  std::vector<bool> usable(touse.size());

  auto point = std::make_unique<double[]>(E_actual(maxE));
  double target;

  for (int i = 0; i < _t.size(); i++) {
    if (!touse[i]) {
      usable[i] = false;
      continue;
    }

    fill_in_point(i, maxE, copredict, prediction, point.get(), &target);

    if (_allow_missing) {
      bool observedAny = false;
      for (int j = 0; j < E_actual(maxE); j++) {
        if (point[j] != MISSING) {
          observedAny = true;
          break;
        }
      }

      usable[i] = observedAny && target != MISSING;
    } else {
      bool foundMissing = false;
      for (int j = 0; j < E_actual(maxE); j++) {
        if (point[j] == MISSING) {
          foundMissing = true;
          break;
        }
      }

      usable[i] = !foundMissing && target != MISSING;
    }
  }

  return usable;
}

void to_json(json& j, const ManifoldGenerator& g)
{
  j = json{ { "_use_dt", g._use_dt },
            { "_add_dt0", g._add_dt0 },
            { "_cumulative_dt", g._cumulative_dt },
            { "_panel_mode", g._panel_mode },
            { "_tau", g._tau },
            { "_p", g._p },
            { "_num_extras", g._num_extras },
            { "_num_extras_lagged", g._num_extras_lagged },
            { "_dtWeight", g._dtWeight },
            { "_x", g._x },
            { "_xmap", g._xmap },
            { "_co_x", g._co_x },
            { "_t", g._t },
            { "_observation_number", g._observation_number },
            { "_extras", g._extras },
            { "_panel_ids", g._panel_ids } };
}

void from_json(const json& j, ManifoldGenerator& g)
{
  j.at("_use_dt").get_to(g._use_dt);
  j.at("_add_dt0").get_to(g._add_dt0);
  j.at("_cumulative_dt").get_to(g._cumulative_dt);
  j.at("_panel_mode").get_to(g._panel_mode);
  j.at("_tau").get_to(g._tau);
  j.at("_p").get_to(g._p);
  j.at("_num_extras").get_to(g._num_extras);
  j.at("_num_extras_lagged").get_to(g._num_extras_lagged);
  j.at("_dtWeight").get_to(g._dtWeight);
  j.at("_x").get_to(g._x);
  j.at("_xmap").get_to(g._xmap);
  j.at("_co_x").get_to(g._co_x);
  j.at("_t").get_to(g._t);
  j.at("_observation_number").get_to(g._observation_number);
  j.at("_extras").get_to(g._extras);
  j.at("_panel_ids").get_to(g._panel_ids);
}
