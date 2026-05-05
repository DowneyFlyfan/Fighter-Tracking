#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "post_process/KalmanScoredTracker.h"
#include "utils/init_box_loader.h"
#include "utils/thread_affinity.h"
#include "utils/types.h"
#include "utils/video_io.h"

constexpr std::string_view DEFAULT_VIDEO_FILE =
    "./Datasets/Anti-UAV-RGBT/test/test1/infrared.mp4";
constexpr std::string_view INIT_BOXES_JSON = "./tools/init_boxes.json";

int main(int argc, char **argv) {
    const std::string VIDEO_FILE =
        argc > 1 ? argv[1] : std::string(DEFAULT_VIDEO_FILE);
    bind_thread_to_core(0);
    std::cout << std::fixed << std::setprecision(6);

    KalmanScoredTracker tracker;

    std::array<float, 4> init_box;
    try {
        init_box = load_init_box(std::string(INIT_BOXES_JSON), VIDEO_FILE);
        std::cout << "init_box (Grounding DINO Tiny): ["
                  << init_box[0] << ", " << init_box[1] << ", "
                  << init_box[2] << ", " << init_box[3] << "]\n";
    } catch (const std::exception &e) {
        std::cerr << "Fatal: " << e.what() << std::endl;
        return 1;
    }

    cv::VideoCapture cap(VIDEO_FILE, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cerr << "Fatal: failed to open " << VIDEO_FILE << std::endl;
        return 1;
    }
    cv::VideoWriter writer("output.mp4",
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           30.0,
                           cv::Size(static_cast<int>(IMG_WIDTH),
                                    static_cast<int>(IMG_HEIGHT)));
    if (!writer.isOpened()) {
        std::cerr << "Fatal: failed to open output.mp4" << std::endl;
        return 1;
    }

    cv::Mat frame;
    int frame_count = 0;
    std::array<float, 4> box{};
    while (read_masked_frame(cap, frame)) {
        if (frame_count == 0) {
            box = init_box;
            tracker.init(box);
            tracker.build_template(frame, init_box);
        } else {
            auto t0 = std::chrono::high_resolution_clock::now();
            box = tracker.update(frame);
            auto t1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> dt = t1 - t0;
            std::cout << "BOX:"
                      << static_cast<int>(box[0] + box[2] / 2) << ","
                      << static_cast<int>(box[1] + box[3] / 2)
                      << " W,H:" << static_cast<int>(box[2])
                      << "," << static_cast<int>(box[3]) << "\n"
                      << "RUN time: " << dt.count() << " ms\n\n";
        }
        ++frame_count;

        cv::Mat vis;
        cv::cvtColor(frame, vis, cv::COLOR_GRAY2BGR);
        cv::rectangle(vis,
                      cv::Point(static_cast<int>(box[0]),
                                static_cast<int>(box[1])),
                      cv::Point(static_cast<int>(box[0] + box[2]),
                                static_cast<int>(box[1] + box[3])),
                      cv::Scalar(0, 255, 0), 2);
        writer.write(vis);
    }

    cap.release();
    writer.release();
    return 0;
}
