#pragma once
#include <vector>
#include <string>
#include <functional>
#include <mutex>
#include <atomic>
#include <thread>
#include <queue>
#include <condition_variable>

// HTTP download result
struct DownloadResult {
    bool success;
    int  status_code;
    std::vector<uint8_t> data;
    std::string error;
};

// Async download manager using WinHTTP
class Downloader {
public:
    Downloader(int maxConcurrent = 32);
    ~Downloader();

    // Synchronous HTTPS GET
    static DownloadResult httpGet(const std::string& host, const std::string& path,
                                   int port = 443, bool https = true);

    // Queue an async download
    using Callback = std::function<void(const std::string& id, DownloadResult result)>;
    void queueDownload(const std::string& id,
                       const std::string& host,
                       const std::string& path,
                       Callback callback);

    // Wait for all queued downloads to complete
    void waitAll();

    // Number of pending downloads
    int pending() const { return m_pending.load(); }

    // Shutdown workers and cancel queued-but-not-started downloads.
    void shutdown();

private:
    struct DownloadTask {
        std::string id;
        std::string host;
        std::string path;
        Callback callback;
    };

    void workerThread();

    std::vector<std::thread> m_workers;
    std::queue<DownloadTask> m_queue;
    std::mutex               m_mutex;
    std::condition_variable  m_cv;
    std::atomic<int>         m_pending{0};
    std::atomic<bool>        m_shutdown{false};
    std::condition_variable  m_doneCV;
};
