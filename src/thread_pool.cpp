#include "thread_pool.h"

// the destructor joins all threads
ThreadPool::~ThreadPool()
{
  kill_all_workers();
}
