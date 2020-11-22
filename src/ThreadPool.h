// Adapted from https://github.com/jhasse/ThreadPool/blob/master/ThreadPool.hpp
#pragma once

#include <boost/circular_buffer.hpp>
#include <functional>
#include <future>
#include <vector>

class MultiQueueThreadPool
{
public:
  explicit MultiQueueThreadPool(size_t, size_t, size_t);
  template<class F, class... Args>
  decltype(auto) enqueue(F&& f, Args&&... args);
  ~MultiQueueThreadPool();

  // need to keep track of threads so we can join them
  std::vector<std::thread> workers;

private:
  // the task queue
  std::vector<boost::circular_buffer<std::packaged_task<void()>>> queues;

  // synchronization
  std::vector<std::mutex> mutexes;
  std::vector<std::condition_variable> conditions;
  std::vector<bool> stop;

  // allocating jobs to threads in batches
  size_t _batchSize, currentBatch, workerNum = 0;
};

// the constructor just launches some amount of workers
inline MultiQueueThreadPool::MultiQueueThreadPool(size_t threads, size_t maxBacklog = 0, size_t batchSize = 1)
  : mutexes(threads)
  , conditions(threads)
  , stop(threads)
  , _batchSize(batchSize)
  , currentBatch(batchSize)
{
  for (size_t i = 0; i < threads; ++i) {
    queues.emplace_back(maxBacklog);
    stop[i] = false;

    workers.emplace_back([this, i] {
      for (;;) {
        std::packaged_task<void()> task;

        {
          std::unique_lock<std::mutex> lock(this->mutexes[i]);
          this->conditions[i].wait(lock, [this, i] { return this->stop[i] || !this->queues[i].empty(); });
          if (this->stop[i] && this->queues[i].empty())
            return;

          task = std::move(this->queues[i].front());
          this->queues[i].pop_front();
        }

        task();
      }
    });
  }
}

// add new work item to the pool
template<class F, class... Args>
decltype(auto) MultiQueueThreadPool::enqueue(F&& f, Args&&... args)
{
  using return_type = std::invoke_result_t<F, Args...>;

  std::packaged_task<return_type()> task(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task.get_future();

  // Choose which thread to allocate the task to
  if (currentBatch == 0) {
    currentBatch = _batchSize;
    workerNum = (workerNum + 1) % workers.size();
  }
  currentBatch -= 1;

  {
    std::unique_lock<std::mutex> lock(mutexes[workerNum]);

    // don't allow enqueueing after stopping the pool
    if (stop[workerNum])
      throw std::runtime_error("enqueue on stopped ThreadPool");

    if (queues[workerNum].full()) {
      queues[workerNum].set_capacity(2 * queues[workerNum].capacity());
      if (queues[workerNum].full()) {
        throw std::runtime_error("Couldn't add more items to the work queue.");
      }
    }
    queues[workerNum].push_back(std::move(task));
  }
  conditions[workerNum].notify_one();
  return res;
}