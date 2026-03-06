#pragma once

#include <cstdint>
#include <string>
#include <vector>

bool decodeMediaAudioToMonoF32(const std::string &path,
                               uint32_t targetSampleRate,
                               std::vector<float> &outPcm,
                               std::string &error);

