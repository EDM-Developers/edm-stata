// Adapted from https://github.com/jhasse/ThreadPool/blob/master/ThreadPool.hpp
#pragma once

#include <boost/circular_buffer.hpp>
#include <functional>
#include <future>
#include <vector>

class ThreadPool
{
public:
  explicit ThreadPool(size_t, size_t);
  template<class F, class... Args>
  decltype(auto) enqueue(F&& f, Args&&... args);
  ~ThreadPool();

  // need to keep track of threads so we can join them
  std::vector<std::thread> workers;

private:
  // the task queue
  boost::circular_buffer<std::packaged_task<void()>> tasks;

  // synchronization
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads, size_t numtasks)
  : stop(false)
{
  tasks.set_capacity(numtasks);

  for (size_t i = 0; i < threads; ++i)
    workers.emplace_back([this] {
      for (;;) {
        std::packaged_task<void()> task;

        {
          std::unique_lock<std::mutex> lock(this->queue_mutex);
          this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
          if (this->stop && this->tasks.empty())
            return;
          task = std::move(this->tasks.front());
          this->tasks.pop_front();
        }

        task();
      }
    });
}

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

    tasks.push_back(std::move(task));
  }
  condition.notify_one();
  return res;
}
