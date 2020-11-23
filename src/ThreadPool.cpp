#include "ThreadPool.h"

// the destructor joins all threads
ThreadPool::~ThreadPool()
{
  kill_all_workers();
}
