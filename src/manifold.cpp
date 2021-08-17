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

int ManifoldGenerator::get_observation_num(int i)
{
  return _observation_number[i];
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

bool ManifoldGenerator::find_observation_num(int target, int& k, int direction, int panel) const
{

  bool panelMode = _panel_ids.size() > 0;

  // Loop either forward or back until we find the right index or give up.
  while (k >= 0 && k < _observation_number.size()) {
    // If in panel mode, make sure we don't wander over a panel boundary.
    if (panelMode) {
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

std::vector<int> ManifoldGenerator::get_lagged_indices(int i, int startIndex, int E, int panel) const
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
    int target = pointStartObsNum - j * _tau;

    if (find_observation_num(target, k, -1, panel)) {
      laggedIndices[j] = k;
    }
  }

  return laggedIndices;
}

Manifold ManifoldGenerator::create_manifold(int E, const std::vector<bool>& filter, bool copredict, bool prediction,
                                            bool skipMissing) const
{
  bool panelMode = _panel_ids.size() > 0;

  std::vector<int> pointNumToStartIndex, panelIDs;
  std::vector<double> y;

  int nobs = 0;
  for (int i = 0; i < filter.size(); i++) {
    if (filter[i]) {
      pointNumToStartIndex.push_back(i);
      y.push_back(_y[i]);
      if (panelMode) {
        panelIDs.push_back(_panel_ids[i]);
      }
      nobs += 1;
    }
  }

  auto flat = std::make_unique<double[]>(nobs * E_actual(E));

  // Fill in the manifold row-by-row (point-by-point)
  int M_i = 0;

  for (int i = 0; i < nobs; i++) {
    int panel = panelMode ? panelIDs[i] : -1;

    std::vector<int> laggedIndices = get_lagged_indices(i, pointNumToStartIndex.at(i), E, panel);

    auto lookup_vec = [&laggedIndices](const std::vector<double>& vec, int j) {
      if (laggedIndices[j] < 0) {
        return MISSING;
      } else {
        return vec[laggedIndices[j]];
      }
    };

    // Fill in the lagged embedding of x (or co_x) in the first columns
    for (int j = 0; j < E; j++) {
      if (prediction && copredict) {
        flat[M_i * E_actual(E) + j] = lookup_vec(_co_x, j);
      } else {
        flat[M_i * E_actual(E) + j] = lookup_vec(_x, j);
      }
    }

    // Put the lagged embedding of dt in the next columns
    if (E_dt(E) > 0) {
      // The first dt value is a bit special as it is relative to the
      // time of the corresponding y prediction.
      if (_p == 0) {
        flat[M_i * E_actual(E) + E + 0] = 0; // Special case for contemporaneous predictions.
      } else {
        double tNow = lookup_vec(_t, 0);

        // At what time does the prediction occur?
        int k = pointNumToStartIndex[i];
        int target = _observation_number[k] + _p;
        int direction = _p > 0 ? 1 : -1;

        if (tNow != MISSING && find_observation_num(target, k, direction, panel)) {
          double tPred = _t[k];
          flat[M_i * E_actual(E) + E + 0] = _dtWeight * (tPred - tNow);

        } else {
          flat[M_i * E_actual(E) + E + 0] = MISSING;
        }
      }

      for (int j = 1; j < E_dt(E); j++) {
        double tNext = lookup_vec(_t, j - 1);
        double tNow = lookup_vec(_t, j);
        if (tNext != MISSING && tNow != MISSING) {
          flat[M_i * E_actual(E) + E + j] = _dtWeight * (tNext - tNow);
        } else {
          flat[M_i * E_actual(E) + E + j] = MISSING;
        }
      }
    }

    // Finally put the extras in the last columns
    int offset = 0;
    for (int k = 0; k < _num_extras; k++) {
      int numLags = (k < _num_extras_lagged) ? E : 1;
      for (int j = 0; j < numLags; j++) {
        flat[M_i * E_actual(E) + E + E_dt(E) + offset + j] = lookup_vec(_extras[k], j);
      }
      offset += numLags;
    }

    // Erase this point if we don't want missing values in the resulting manifold
    if (skipMissing) {
      bool foundMissing = false;
      for (int j = 0; j < E_actual(E); j++) {
        if (flat[M_i * E_actual(E) + j] == MISSING) {
          foundMissing = true;
          break;
        }
      }

      if (foundMissing) {
        y.erase(y.begin() + M_i);
        if (panelMode) {
          panelIDs.erase(panelIDs.begin() + M_i);
        }
        continue;
      }
    }

    M_i += 1;
  }

  nobs = M_i;

  return { flat, y, panelIDs, nobs, E, E_dt(E), E_extras(E), E * numExtrasLagged(), E_actual(E) };
}

double ManifoldGenerator::lagged(const std::vector<double>& vec, const std::vector<int>& pointNumToStartIndex, int i,
                                 int j) const
{
  int index = pointNumToStartIndex.at(i);
  int t0 = _observation_number[index];

  for (int k = 0; k < j * _tau; k++) {
    if (t0 - _observation_number[index] == j * _tau) {
      break;
    }

    index -= 1;
    if (index < 0) {
      return MISSING;
    }
  }

  if (_panel_mode) {
    if (_panel_ids[index] != _panel_ids[pointNumToStartIndex.at(i)]) {
      return MISSING;
    }
  }

  return vec[index];
}

double ManifoldGenerator::find_dt(const std::vector<int>& pointNumToStartIndex, int i, int j) const
{
  int ind1, ind2;
  if (_cumulative_dt) {
    ind1 = pointNumToStartIndex.at(i) + _tau;
    ind2 = ind1 - j * _tau;
  } else {
    ind1 = pointNumToStartIndex.at(i) + _add_dt0 * _tau - j * _tau;
    ind2 = ind1 - _tau;
  }

  if ((ind1 >= _t.size()) || (ind2 < 0) || (_t[ind1] == MISSING) || (_t[ind2] == MISSING) || (_t[ind1] < _t[ind2])) {
    return MISSING;
  }
  return _dtWeight * (_t[ind1] - _t[ind2]);
}

std::vector<bool> ManifoldGenerator::generate_usable(const std::vector<bool>& touse, int maxE) const
{
  // Make the largest manifold we'll need in order to find missing values for 'usable'
  std::vector<bool> allTrue(touse.size());
  for (int i = 0; i < allTrue.size(); i++) {
    allTrue[i] = true;
  }

  Manifold M = create_manifold(maxE, allTrue, false, false, false);

  // Generate the 'usable' variable
  std::vector<bool> usable(touse.size());
  for (int i = 0; i < usable.size(); i++) {
    if (_allow_missing) {
      usable[i] = touse[i] && M.any_not_missing(i) && M.y(i) != MISSING;
    } else {
      usable[i] = touse[i] && !M.any_missing(i) && M.y(i) != MISSING;
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
            { "_y", g._y },
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
  j.at("_y").get_to(g._y);
  j.at("_co_x").get_to(g._co_x);
  j.at("_t").get_to(g._t);
  j.at("_observation_number").get_to(g._observation_number);
  j.at("_extras").get_to(g._extras);
  j.at("_panel_ids").get_to(g._panel_ids);
}
