// Suppress Windows problems with sprintf etc. functions.
#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>

#include "common.h"
#include "edm.h"

#include <future>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "httplib.h"

#include "driver.h"

// To allow server to handle compressed requests
// #define CPPHTTPLIB_ZLIB_SUPPORT

#include <nlohmann/json.hpp>

using json = nlohmann::json;

class ServerIO : public IO
{
public:
  virtual void out(const char* s) const { std::cout << s; }
  virtual void error(const char* s) const { std::cerr << s; }
  virtual void flush() const
  { /* TODO */
  }
};

ServerIO io;

// Global state, needed to persist between multiple edm calls
std::queue<Prediction> predictions;
std::queue<std::future<void>> futures;

std::atomic<bool> breakButtonPressed = false;
std::atomic<bool> allTasksFinished = false;

json previousResults = {};

bool keep_going()
{
  return !breakButtonPressed;
}

void all_tasks_finished()
{
  allTasksFinished = true;
}

// In case we have some remnants of previous runs still
// in the system (e.g. after a 'break'), clear our past results.
void reset_global_state()
{
  io.get_and_clear_async_buffer();

  while (!futures.empty()) {
    futures.pop();
  }
  while (!predictions.empty()) {
    predictions.pop();
  }

  breakButtonPressed = false;
  allTasksFinished = false;

  previousResults = {};
}

void launch_edm_task(json j)
{
  Options taskOpts = j["taskOpts"];
  ManifoldGenerator generator = j["generator"];
  int E = j["E"];
  std::vector<bool> trainingRows = j["trainingRows"];
  std::vector<bool> predictionRows = j["predictionRows"];

  io.print(fmt::format("/launch_edm_task ({} of {})\n", taskOpts.taskNum + 1, taskOpts.numTasks));
  if (taskOpts.taskNum == 0) {
    reset_global_state();
  }
  if (io.verbosity > 1) {
    io.print(fmt::format("Task specification: {}\n", j.dump()));
  }

  predictions.push({});

  futures.push(edm_async(taskOpts, generator, E, trainingRows, predictionRows, &io, &(predictions.back()), keep_going,
                         all_tasks_finished));
}

json get_all_task_results()
{
  json j;
  while (predictions.size() > 0) {
    Prediction& pred = predictions.front();
    j[pred.stats.taskNum] = pred;

    predictions.pop();
    futures.pop();
  }

  return j;
}

int main(int argc, char* argv[])
{
  using namespace httplib;

  int port = 8123;
  if (argc > 1) {
    port = atoi(argv[1]);
  }

  io.verbosity = 1;
  std::cout << "Starting EDM server on port " << port << std::endl;

  Server svr;

  svr.Post("/launch_edm_task", [](const Request& req, Response& res) {
    json j = json::parse(req.body);
    launch_edm_task(j);
    res.set_content("Task launched", "text/plain");
  });

  svr.Get("/report_progress", [&](const Request& req, Response& res) {
    std::string progress = io.get_and_clear_async_buffer();
    bool finished = allTasksFinished;

    if (io.verbosity > 0) {
      io.print(progress);
    }

    json j;
    j["progress"] = progress;
    j["finished"] = finished;

    res.set_content(j.dump(), "application/json");
  });

  svr.Get("/collect_results", [](const Request& req, Response& res) {
    io.print("/collect_results\n");

    // In case of a failure in transmission, keep the previous results around
    // and resend them if needed.
    if (predictions.size() > 0) {
      previousResults = get_all_task_results();
    }
    if (io.verbosity > 1) {
      io.print(fmt::format("Sending back results {}\n", previousResults.dump()));
    }
    res.set_content(previousResults.dump(), "application/json");
  });

  svr.Get("/stop", [&](const Request& req, Response& res) {
    io.print("/stop\n");
    svr.stop();
  });

  svr.Get("/test", [](const Request& req, Response& res) {
    io.print("/test\n");
    res.set_content("EDM server running!", "text/plain");
  });

  svr.listen("localhost", port);

  std::cout << "Closing EDM server" << std::endl;
}
