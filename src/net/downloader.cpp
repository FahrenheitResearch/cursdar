#include "downloader.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")
#else
#include <curl/curl.h>
#endif

#include <cstdio>
#include <exception>
#include <string>

// ── HTTP synchronous GET (WinHTTP on Windows, libcurl on Linux) ──

#ifndef _WIN32
// libcurl write callback: appends received data to the result vector
static size_t curlWriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t totalBytes = size * nmemb;
    auto* vec = static_cast<std::vector<uint8_t>*>(userp);
    const uint8_t* data = static_cast<const uint8_t*>(contents);
    vec->insert(vec->end(), data, data + totalBytes);
    return totalBytes;
}

struct CurlGlobalInit {
    CurlGlobalInit() { curl_global_init(CURL_GLOBAL_DEFAULT); }
    ~CurlGlobalInit() { curl_global_cleanup(); }
};

static const CurlGlobalInit kCurlGlobalInit;
#endif

DownloadResult Downloader::httpGet(const std::string& host, const std::string& path,
                                    int port, bool https) {
    DownloadResult result;
    result.success = false;
    result.status_code = 0;

#ifdef _WIN32
    // Convert to wide strings
    auto toWide = [](const std::string& s) -> std::wstring {
        if (s.empty()) return L"";
        int sz = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
        std::wstring ws(sz, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), ws.data(), sz);
        return ws;
    };

    std::wstring wHost = toWide(host);
    std::wstring wPath = toWide(path);

    HINTERNET hSession = WinHttpOpen(
        L"cursdar/1.0",
        WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
        WINHTTP_NO_PROXY_NAME,
        WINHTTP_NO_PROXY_BYPASS, 0);

    if (!hSession) {
        result.error = "WinHttpOpen failed";
        return result;
    }

    // Enable HTTP/2
    DWORD http2 = WINHTTP_PROTOCOL_FLAG_HTTP2;
    WinHttpSetOption(hSession, WINHTTP_OPTION_ENABLE_HTTP_PROTOCOL, &http2, sizeof(http2));

    // Set timeouts (connect=5s, send=10s, receive=30s)
    WinHttpSetTimeouts(hSession, 5000, 5000, 10000, 30000);

    HINTERNET hConnect = WinHttpConnect(hSession, wHost.c_str(),
                                         (INTERNET_PORT)port, 0);
    if (!hConnect) {
        result.error = "WinHttpConnect failed";
        WinHttpCloseHandle(hSession);
        return result;
    }

    DWORD flags = https ? WINHTTP_FLAG_SECURE : 0;
    HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"GET", wPath.c_str(),
                                             nullptr, WINHTTP_NO_REFERER,
                                             WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
    if (!hRequest) {
        result.error = "WinHttpOpenRequest failed";
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return result;
    }

    // Accept compressed responses
    WinHttpAddRequestHeaders(hRequest,
        L"Accept-Encoding: gzip, deflate",
        (DWORD)-1L, WINHTTP_ADDREQ_FLAG_ADD);

    // Enable auto-decompression
    DWORD decompression = WINHTTP_DECOMPRESSION_FLAG_ALL;
    WinHttpSetOption(hRequest, WINHTTP_OPTION_DECOMPRESSION, &decompression, sizeof(decompression));

    if (!WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                            WINHTTP_NO_REQUEST_DATA, 0, 0, 0)) {
        result.error = "WinHttpSendRequest failed: " + std::to_string(GetLastError());
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return result;
    }

    if (!WinHttpReceiveResponse(hRequest, nullptr)) {
        result.error = "WinHttpReceiveResponse failed: " + std::to_string(GetLastError());
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return result;
    }

    // Get status code
    DWORD statusCode = 0;
    DWORD statusSize = sizeof(statusCode);
    WinHttpQueryHeaders(hRequest,
                        WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                        WINHTTP_HEADER_NAME_BY_INDEX, &statusCode,
                        &statusSize, WINHTTP_NO_HEADER_INDEX);
    result.status_code = (int)statusCode;

    // Read body
    result.data.reserve(1024 * 1024); // 1MB initial
    DWORD bytesAvailable = 0;
    do {
        if (!WinHttpQueryDataAvailable(hRequest, &bytesAvailable)) break;
        if (bytesAvailable == 0) break;

        std::vector<uint8_t> buf(bytesAvailable);
        DWORD bytesRead = 0;
        if (WinHttpReadData(hRequest, buf.data(), bytesAvailable, &bytesRead)) {
            result.data.insert(result.data.end(), buf.begin(), buf.begin() + bytesRead);
        }
    } while (bytesAvailable > 0);

    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);

    result.success = (statusCode == 200);
    if (!result.success && result.error.empty()) {
        result.error = "HTTP " + std::to_string(statusCode);
    }

#else // Linux: libcurl implementation

    // Build full URL from host + path
    std::string scheme = https ? "https" : "http";
    std::string url = scheme + "://" + host;
    if ((https && port != 443) || (!https && port != 80)) {
        url += ":" + std::to_string(port);
    }
    url += path;

    CURL* curl = curl_easy_init();
    if (!curl) {
        result.error = "curl_easy_init failed";
        return result;
    }

    result.data.reserve(1024 * 1024); // 1MB initial

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "cursdar/1.0");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result.data);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, ""); // auto decompress
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, https ? 1L : 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, https ? 2L : 0L);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        result.error = std::string("curl_easy_perform failed: ") + curl_easy_strerror(res);
        curl_easy_cleanup(curl);
        return result;
    }

    long httpCode = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
    result.status_code = (int)httpCode;

    curl_easy_cleanup(curl);

    result.success = (httpCode == 200);
    if (!result.success && result.error.empty()) {
        result.error = "HTTP " + std::to_string(httpCode);
    }

#endif

    return result;
}

// ── Thread pool ─────────────────────────────────────────────

Downloader::Downloader(int maxConcurrent) {
    m_workers.reserve(maxConcurrent);
    for (int i = 0; i < maxConcurrent; i++) {
        m_workers.emplace_back(&Downloader::workerThread, this);
    }
}

Downloader::~Downloader() {
    shutdown();
}

void Downloader::queueDownload(const std::string& id,
                                const std::string& host,
                                const std::string& path,
                                Callback callback) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_shutdown.load()) return;
        m_queue.push({id, host, path, std::move(callback)});
        m_pending++;
    }
    m_cv.notify_one();
}

void Downloader::waitAll() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_doneCV.wait(lock, [this] { return m_pending.load() == 0; });
}

void Downloader::shutdown() {
    size_t dropped = 0;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_shutdown = true;
        dropped = m_queue.size();
        while (!m_queue.empty()) m_queue.pop();
    }
    if (dropped > 0) {
        m_pending.fetch_sub((int)dropped);
        m_doneCV.notify_all();
    }
    m_cv.notify_all();
    for (auto& t : m_workers) {
        if (t.joinable() && t.get_id() != std::this_thread::get_id())
            t.join();
    }
    m_workers.clear();
}

void Downloader::workerThread() {
    while (true) {
        DownloadTask task;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this] { return m_shutdown.load() || !m_queue.empty(); });
            if (m_shutdown && m_queue.empty()) return;
            task = std::move(m_queue.front());
            m_queue.pop();
        }

        DownloadResult result;
        try {
            result = httpGet(task.host, task.path);
        } catch (const std::exception& ex) {
            result.success = false;
            result.status_code = 0;
            result.error = std::string("httpGet threw: ") + ex.what();
        } catch (...) {
            result.success = false;
            result.status_code = 0;
            result.error = "httpGet threw unknown exception";
        }

        if (task.callback) {
            try {
                task.callback(task.id, std::move(result));
            } catch (const std::exception& ex) {
                std::fprintf(stderr, "Downloader callback threw: %s\n", ex.what());
            } catch (...) {
                std::fprintf(stderr, "Downloader callback threw unknown exception\n");
            }
        }

        m_pending--;
        m_doneCV.notify_all();
    }
}
