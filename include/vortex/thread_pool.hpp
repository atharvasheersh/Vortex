#pragma once
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace vortex {
class ThreadPool {
  public:
    explicit ThreadPool(std::size_t threads) {
        if (threads == 0 || threads > 256)
            throw std::invalid_argument("threads must be 1..256");
        try {
            for (std::size_t i = 0; i < threads; ++i)
                workers_.emplace_back([this] {
                    for (;;) {
                        std::function<void()> task;
                        {
                            std::unique_lock lock(mutex_);
                            ready_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                            if (stopping_ && tasks_.empty())
                                return;
                            task = std::move(tasks_.front());
                            tasks_.pop();
                        }
                        task();
                    }
                });
        } catch (...) {
            stop();
            throw;
        }
    }
    ~ThreadPool() { stop(); }
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;
    template <class F> auto submit(F task) {
        using R = std::invoke_result_t<F>;
        auto packaged = std::make_shared<std::packaged_task<R()>>(std::move(task));
        auto result = packaged->get_future();
        {
            std::lock_guard lock(mutex_);
            if (stopping_)
                throw std::runtime_error("pool is stopping");
            tasks_.emplace([packaged] { (*packaged)(); });
        }
        ready_.notify_one();
        return result;
    }

  private:
    void stop() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto &worker : workers_)
            if (worker.joinable())
                worker.join();
    }
    std::mutex mutex_;
    std::condition_variable ready_;
    bool stopping_ = false;
    std::queue<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
};
} // namespace vortex
