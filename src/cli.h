#pragma once

#include <format>
#include "edm.h"
#include <fstream>
#include <iomanip>

#ifdef JSON

#include <nlohmann/json.hpp>
using json = nlohmann::json;

struct Inputs
{
  Options opts;
  ManifoldGenerator generator;
  int E;
  std::vector<bool> libraryRows, predictionRows;
};

Inputs parse_lowlevel_inputs_file(const json& j);
Inputs read_lowlevel_inputs_file(std::string fName);
void append_to_dumpfile(std::string fName, const json& taskGroup);
std::vector<bool> int_to_bool(std::vector<int> iv);
json run_tests(json testInputs, int nthreads, IO* io = nullptr);

#endif