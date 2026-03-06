#pragma once

#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "frame.h"

cv::Mat resizeAndGrayscale(const cv::Mat &input, int width, int height);
Gray4Frame quantizeTo4Bit(const cv::Mat &gray8);
cv::Mat renderForDisplay(const Gray4Frame &frame, cv::Size outputSize, bool enhance, bool dither, int blockSize = 8);

void drawHud(cv::Mat &image,
             const std::vector<std::string> &lines,
             cv::Point origin = cv::Point(12, 24),
             int maxWidth = -1);

std::string makeRecordingFileName();
bool openRecordingFile(std::ofstream &output, const std::string &path);
bool appendRecordingPacket(std::ofstream &output, const std::vector<uint8_t> &packet);
bool loadRecordingFile(const std::string &path, std::vector<std::vector<uint8_t>> &packets);
