#pragma once

#include "types.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

// Look up the frame-0 detected bounding box for this video in the
// init-boxes JSON. Key is the video directory basename
// (e.g. "test1" from "Datasets/.../test/test1/infrared.mp4").
// Box is [x, y, w, h]; tracker only consumes w, h to seed
// shift_subtract on frame 0.
//
// Hand-rolled regex match instead of pulling in a JSON dependency
// (the schema is fixed and trivially regex-able):
//   "<key>": {"box": [<x>, <y>, <w>, <h>], ...}
inline std::array<float, 4> load_init_box(const std::string &json_path,
                                          const std::string &video_path) {
    std::filesystem::path p(video_path);
    std::string key = p.parent_path().filename().string();

    std::ifstream f(json_path);
    if (!f.is_open()) {
        throw std::runtime_error("init-boxes JSON not found: " + json_path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string content = ss.str();

    std::regex rgx("\"" + key +
                   "\"\\s*:\\s*\\{\\s*\"box\"\\s*:\\s*\\[\\s*"
                   "([0-9.eE+-]+)\\s*,\\s*([0-9.eE+-]+)\\s*,\\s*"
                   "([0-9.eE+-]+)\\s*,\\s*([0-9.eE+-]+)");
    std::smatch m;
    if (!std::regex_search(content, m, rgx)) {
        throw std::runtime_error("init box not found for key '" + key +
                                 "' in " + json_path);
    }
    return {std::stof(m[1].str()), std::stof(m[2].str()),
            std::stof(m[3].str()), std::stof(m[4].str())};
}
