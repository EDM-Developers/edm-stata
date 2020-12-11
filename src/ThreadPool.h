// Adapted from https://github.com/jhasse/ThreadPool/blob/master/ThreadPool.hpp
#pragma once

#include <functional>
#include <future>
#include <queue>
#include <vector>

#ifndef _MSC_VER
#ifndef __APPLE__
#include <pthread.h>

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>
#endif
#endif

class ThreadPool
{
private:
  // the task queue
  std::queue<std::packaged_task<void()>> tasks;

  // synchronization
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop = false;

  void kill_all_workers()
  {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers) {
      worker.join();
    }
    workers.clear();
  }

public:
  // need to keep track of threads so we can join them
  std::vector<std::thread> workers;

  ThreadPool() {}

  ThreadPool(size_t threads) { set_num_workers(threads); }

  template<class F, class... Args>
  decltype(auto) enqueue(F&& f, Args&&... args);
  ~ThreadPool();

  inline void set_num_workers(size_t numworkers)
  {
    size_t currentSize = workers.size();
    if (currentSize == numworkers) {
      return;
    }

    if (numworkers < currentSize) {
      kill_all_workers();
      stop = false;
      currentSize = 0;
    }

    for (size_t i = 0; i < numworkers - currentSize; ++i) {
      workers.emplace_back([this, i, numworkers] {

#ifndef _MSC_VER
#ifndef __APPLE__
        pthread_setname_np(pthread_self(),
                           fmt::format("edm worker {} of {} [id {}])", i, numworkers, pthread_self()).c_str());
#endif
#endif

        for (;;) {
          std::packaged_task<void()> task;

          // Don't restart the thread when the pool is running (stop == false) but there's no work to do
          // (tasks.empty()).
          {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
            // Stop the worker if the pool is stopped & there's no left-over tasks in the queue.
            if (this->stop && this->tasks.empty())
              return;
            task = std::move(this->tasks.front());
            this->tasks.pop();
          }

          task();
        }
      });
    }
  }
};

// add new work item to the pool
template<class F, class... Args>
decltype(auto) ThreadPool::enqueue(F&& f, Args&&... args)
{
  using return_type = std::invoke_result_t<F, Args...>;

  std::packaged_task<return_type()> task(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task.get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex);

    // don't allow enqueueing after stopping the pool
    if (stop)
      throw std::runtime_error("enqueue on stopped ThreadPool");

    tasks.emplace(std::move(task));
  }
  condition.notify_one();
  return res;
}
