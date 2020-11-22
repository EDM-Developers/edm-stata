#include "ThreadPool.h"

// the destructor joins all threads
MultiQueueThreadPool::~MultiQueueThreadPool()
{
  for (size_t i = 0; i < workers.size(); i++) {
    {
      std::scoped_lock lock(mutexes[i]);
      stop[i] = true;
    }
    conditions[i].notify_all();
  }

  for (std::thread& worker : workers) {
    worker.join();
  }
}