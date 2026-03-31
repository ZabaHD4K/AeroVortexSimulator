#pragma once
#include <iostream>
#include <string>

enum class LogLevel { DEBUG, INFO, WARN, ERROR };

inline LogLevel g_logLevel = LogLevel::INFO;

#define LOG_DEBUG(msg) do { if (g_logLevel <= LogLevel::DEBUG) std::cout << "[DEBUG] " << msg << std::endl; } while(0)
#define LOG_INFO(msg)  do { if (g_logLevel <= LogLevel::INFO)  std::cout << "[INFO]  " << msg << std::endl; } while(0)
#define LOG_WARN(msg)  do { if (g_logLevel <= LogLevel::WARN)  std::cerr << "[WARN]  " << msg << std::endl; } while(0)
#define LOG_ERROR(msg) do { if (g_logLevel <= LogLevel::ERROR) std::cerr << "[ERROR] " << msg << std::endl; } while(0)
