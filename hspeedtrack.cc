#include <NvInfer.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <pthread.h>
#include <sched.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "post_process/CtrCorrect.h"
#include "post_process/FilterByBox.h"
#include "post_process/FilterKpts.h"
#include "post_process/MatchKptsCorrect.h"
#include "post_process/SmiTri.h"
#include "utils/box_size.h"
#include "utils/descriptor_match.h"
#include "utils/get_roi.h"
#include "utils/init_engine.h"
#include "utils/parallel_topk.h"
#include "utils/slice.h"
#include "utils/thresh.h"
#include "utils/types.h"
#include "utils/utils.h"

using namespace nvinfer1;

void bind_thread_to_core(int core_id) { // NOTE: Make sure that program doesn't
                                        // jump between different cores
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_t current_thread = pthread_self();
    pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

constexpr int NUM_THREADS = 4;
constexpr std::string_view ENGINE_FILE =
    "./engine_model/Norm_Grad_Response_Masked_Max_480.engine";
constexpr std::string_view DATA_FOLDER = "./Datasets/test_imgs";
constexpr cudaMemLocation loc = {.type = cudaMemLocationTypeHost};

int frame_count = 0;

int main() {
    bind_thread_to_core(0);
    omp_set_num_threads(NUM_THREADS);
    std::cout << std::fixed << std::setprecision(16);

    // Initialize TensorRT engine
    auto engine_opt = init_engine(std::string(ENGINE_FILE));
    if (!engine_opt) {
        std::cerr << "Fatal: failed to initialize engine" << std::endl;
        return 1;
    }
    INIT_engine &Grad_Response = *engine_opt;

    // Allocate CUDA unified memory for TRT I/O bindings
    constexpr int UNIT_SIZE = sizeof(float);
    float *img, *response, *x_max, *y_max;
    CHECK_CUDA(cudaMallocManaged(&img, ROI_SIZE * ROI_SIZE * UNIT_SIZE));
    CHECK_CUDA(cudaMallocManaged(&response, ROI_SIZE * ROI_SIZE * UNIT_SIZE));
    CHECK_CUDA(cudaMallocManaged(&x_max, ROI_SIZE * UNIT_SIZE));
    CHECK_CUDA(cudaMallocManaged(&y_max, ROI_SIZE * UNIT_SIZE));

    // Create CUDA stream
    cudaStream_t stream1;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    Grad_Response.context->setTensorAddress(
        Grad_Response.engine->getIOTensorName(0), img);
    Grad_Response.context->setTensorAddress(
        Grad_Response.engine->getIOTensorName(1), response);
    Grad_Response.context->setTensorAddress(
        Grad_Response.engine->getIOTensorName(2), x_max);
    Grad_Response.context->setTensorAddress(
        Grad_Response.engine->getIOTensorName(3), y_max);

    FilterKptsResult kpts_result;
    FilterByBoxResult boxfil_result;
    SmiTriCheck smitri_check;
    std::array<double, 2> new_ctr_pts;
    Point ctr_pt_last;
    Box corrected_xywh;
    Box frangi_xyxy;
    MatchKptsCorrectResult OrbMatch_result;

    auto sorted_entries = get_sorted_image_entries(std::string(DATA_FOLDER));

    std::array<float, 4> tgt_xywh_curr, tgt_xywh_last, tgt_xywh_refined_last;
    std::array<Point, 40> kpts_for_patches, kpts_curr, kpts_last,
        kpts_refined_last;
    std::array<Descriptor, 40> dscrp_curr, dscrp_last, dscrp_refined_last;

    std::array<std::array<float, 3>, 40> matches;
    std::array<std::array<float, 3>, 40> matches_refined;

    bool flame_signal_curr = true;
    bool flame_signal_last = false;

    // Only the very first invocation of the frame-0 branch reads from
    // BOXsz_dict. A later fallback (target lost → frame_count = 0) reuses
    // tgt_xywh_last[2,3], which still holds the last valid tracked size.
    bool first_init = true;

    uint8_t *roi = new uint8_t[ROI_SIZE * ROI_SIZE];
    float *flame_cover_mask = new float[ROI_SIZE * ROI_SIZE];
    float *orb_response = new float[ROI_SIZE * ROI_SIZE];

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat eroded_mask(ROI_SIZE, ROI_SIZE, CV_32F);
    int roi_tl_x, roi_tl_y;

    std::array<std::pair<float, int>, TOPK> topk;
    std::array<std::array<std::array<float, 25>, 25>, 40> patches;

    cv::VideoWriter writer(
        "output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
        cv::Size(static_cast<int>(IMG_WIDTH), static_cast<int>(IMG_HEIGHT)));

    {
        // Double-buffered image prefetch: while GPU runs frame N's TRT, CPU
        // reads frame N+1 from disk
        cv::Mat img_curr_np, img_next_np;
        std::string nimg, nimg_next;
        if (!sorted_entries.empty()) {
            img_curr_np = cv::imread(sorted_entries.front().path().string(),
                                     cv::IMREAD_GRAYSCALE);
            nimg = sorted_entries.front().path().stem().string();
        }
        for (auto it = sorted_entries.begin(); it != sorted_entries.end();
             ++it) {
            auto next_it = std::next(it);
            bool has_next = (next_it != sorted_entries.end());

            // Buffer swap: if previous iteration prefetched the next image,
            // promote it. Otherwise (first iter or post-continue), read sync.
            if (it != sorted_entries.begin()) {
                if (!img_next_np.empty()) {
                    img_curr_np = std::move(img_next_np);
                    nimg = std::move(nimg_next);
                    img_next_np.release();
                } else {
                    img_curr_np =
                        cv::imread(it->path().string(), cv::IMREAD_GRAYSCALE);
                    nimg = it->path().stem().string();
                }
            }
            if (frame_count == 0) {
                // === First frame processing (also used as the fallback
                // re-init path when the target is lost) ===
                // On the very first invocation we read the box size from
                // BOXsz_dict. On any later fallback we reuse the last
                // tracked size in tgt_xywh_last, so per-frame dictionary
                // lookups are not needed.
                float w, h;
                if (first_init) {
                    auto boxsz = BOXsz_dict.find(nimg);
                    w = boxsz->second[0];
                    h = boxsz->second[1];
                    first_init = false;
                } else {
                    w = tgt_xywh_last[2];
                    h = tgt_xywh_last[3];
                }

                frame_count += 1;

                // NOTE: Even need to adjust its height and width
                int w_resize = static_cast<int>(w * (ROI_SIZE / IMG_WIDTH));
                int h_resize = static_cast<int>(h * (ROI_SIZE / IMG_HEIGHT));

                cv::Mat resized_img;
                cv::resize(img_curr_np, resized_img,
                           cv::Size(ROI_SIZE, ROI_SIZE), 0, 0, cv::INTER_AREA);

                cv::Mat float_img;
                resized_img.convertTo(float_img, CV_32F);
                std::memcpy(img, float_img.ptr<float>(),
                            sizeof(float) * ROI_SIZE * ROI_SIZE);

                // TRT
                Grad_Response.context->enqueueV3(stream1);
                cudaMemPrefetchAsync(x_max, ROI_SIZE * sizeof(float), loc, 0,
                                     stream1);
                cudaMemPrefetchAsync(y_max, ROI_SIZE * sizeof(float), loc, 0,
                                     stream1);
                cudaStreamSynchronize(stream1);

                // Prefix sum on x_max and y_max
                for (int i = 1; i < ROI_SIZE; ++i) {
                    x_max[i] += x_max[i - 1];
                }
                for (int i = 1; i < ROI_SIZE; ++i) {
                    y_max[i] += y_max[i - 1];
                }

                // Shifted subtraction to find target location
                tgt_xywh_curr =
                    shift_subtract(x_max, y_max, w_resize, h_resize);

                // Scale back to original image coordinates
                tgt_xywh_curr[0] = static_cast<float>(
                    std::round(tgt_xywh_curr[0] * (IMG_WIDTH / ROI_SIZE)));
                tgt_xywh_curr[1] = static_cast<float>(
                    std::round(tgt_xywh_curr[1] * (IMG_HEIGHT / ROI_SIZE)));
                tgt_xywh_curr[2] = w;
                tgt_xywh_curr[3] = h;

                // Extract ROI (Region of Interest) around target
                std::tie(roi_tl_x, roi_tl_y) =
                    GetROI(roi, img_curr_np.ptr<uint8_t>(0), tgt_xywh_curr);

                cudaMemPrefetchAsync(img, ROI_SIZE * ROI_SIZE * sizeof(float),
                                     loc, 0, stream1);
                cudaStreamSynchronize(stream1);

                // Threshold to detect flame and convert to float
                threshold(roi, flame_cover_mask, img, flame_signal_curr);

                // Second TRT inference on ROI, then prefetch response to CPU
                Grad_Response.context->enqueueV3(stream1);
                cudaMemPrefetchAsync(response,
                                     ROI_SIZE * ROI_SIZE * sizeof(float), loc,
                                     0, stream1);
                // Overlap: read next frame from disk while GPU runs 2nd TRT
                if (has_next) {
                    img_next_np =
                        cv::imread(next_it->path().string(),
                                   cv::IMREAD_GRAYSCALE);
                    nimg_next = next_it->path().stem().string();
                }
                cudaStreamSynchronize(stream1);

                // Top-K keypoint extraction from response map
                topk = topk_sorted_parallel(response);

                // NOTE: Maybe can be improved here
                int i = 0;
                for (const auto &vi : topk) {
                    int index = vi.second;
                    int y = index / ROI_SIZE;
                    int x = index % ROI_SIZE;
                    kpts_for_patches[i] = {static_cast<float>(y),
                                           static_cast<float>(x)};
                    kpts_curr[i] = {static_cast<float>(x + roi_tl_x),
                                    static_cast<float>(y + roi_tl_y)};
                    ++i;
                }

                // Extract patches and compute ORB (Oriented FAST and Rotated
                // BRIEF) descriptors
                extract_all_patches(patches, img, kpts_for_patches);
                dscrp_curr = extract_descriptors(patches);

                // Pass state to next frame
                tgt_xywh_last = tgt_xywh_curr;
                kpts_last = kpts_curr;
                dscrp_last = dscrp_curr;
                flame_signal_last = flame_signal_curr;

                tgt_xywh_refined_last = tgt_xywh_curr;
                kpts_refined_last = kpts_curr;
                dscrp_refined_last = dscrp_curr;

                // Write annotated frame to video
                {
                    cv::Mat vis;
                    cv::cvtColor(img_curr_np, vis, cv::COLOR_GRAY2BGR);
                    cv::rectangle(vis,
                                  cv::Point(static_cast<int>(tgt_xywh_curr[0]),
                                            static_cast<int>(tgt_xywh_curr[1])),
                                  cv::Point(static_cast<int>(tgt_xywh_curr[0] +
                                                             tgt_xywh_curr[2]),
                                            static_cast<int>(tgt_xywh_curr[1] +
                                                             tgt_xywh_curr[3])),
                                  cv::Scalar(0, 255, 0), 2);
                    writer.write(vis);
                }

            } else {
                // === Subsequent frame processing ===
                frame_count += 1;
                auto start = std::chrono::high_resolution_clock::now();

                // Extract ROI using last frame's target location
                std::tie(roi_tl_x, roi_tl_y) =
                    GetROI(roi, img_curr_np.ptr<uint8_t>(0), tgt_xywh_last);

                // Prefetch img to CPU (GPU read during enqueueV3 changes page
                // state)
                cudaMemPrefetchAsync(img, ROI_SIZE * ROI_SIZE * sizeof(float),
                                     loc, 0, stream1);
                cudaStreamSynchronize(stream1);

                // Threshold and convert ROI
                threshold(roi, flame_cover_mask, img, flame_signal_curr);

                // If flame was present last frame but gone now, reset to
                // first-frame mode
                if (flame_signal_last && !flame_signal_curr) {
                    frame_count = 0;
                    flame_signal_last = flame_signal_curr;
                    continue;
                }

                // TRT inference on current ROI, then prefetch outputs to CPU
                // (all queued on stream1: inference → prefetch, executed in
                // order)
                Grad_Response.context->enqueueV3(stream1);
                cudaMemPrefetchAsync(response,
                                     ROI_SIZE * ROI_SIZE * sizeof(float), loc,
                                     0, stream1);
                cudaMemPrefetchAsync(x_max, ROI_SIZE * sizeof(float), loc, 0,
                                     stream1);
                cudaMemPrefetchAsync(y_max, ROI_SIZE * sizeof(float), loc, 0,
                                     stream1);

                // Erode flame mask on CPU while GPU inference + prefetch run
                cv::Mat flame_cover_mask_mat(ROI_SIZE, ROI_SIZE, CV_32F,
                                             flame_cover_mask);
                cv::erode(flame_cover_mask_mat, eroded_mask, kernel);
                if (has_next) {
                    img_next_np =
                        cv::imread(next_it->path().string(),
                                   cv::IMREAD_GRAYSCALE);
                    nimg_next = next_it->path().stem().string();
                }

                // Wait for inference + prefetch to complete
                cudaStreamSynchronize(stream1);

                // Apply mask to response (suppress flame and boundary regions)
                multiply(response, eroded_mask.ptr<float>(), orb_response);

                // Top-K keypoint extraction from masked response
                topk = topk_sorted_parallel(orb_response);

                int i = 0;
                for (const auto &vi : topk) {
                    int index = vi.second;
                    int y = index / ROI_SIZE;
                    int x = index % ROI_SIZE;
                    kpts_for_patches[i] = {static_cast<float>(y),
                                           static_cast<float>(x)};
                    kpts_curr[i] = {static_cast<float>(x + roi_tl_x),
                                    static_cast<float>(y + roi_tl_y)};
                    ++i;
                }

                // Extract patches and compute descriptors
                extract_all_patches(patches, img, kpts_for_patches);
                dscrp_curr = extract_descriptors(patches);

                // Match descriptors between consecutive frames
                matches = match_descriptors(dscrp_last, dscrp_curr);
                matches_refined =
                    match_descriptors(dscrp_refined_last, dscrp_curr);

                // Prefix sum for cumulative response
                for (int i = 1; i < ROI_SIZE; ++i) {
                    x_max[i] += x_max[i - 1];
                }
                for (int i = 1; i < ROI_SIZE; ++i) {
                    y_max[i] += y_max[i - 1];
                }

                // Shifted subtraction for target localization.
                // Use the running box size from tgt_xywh_last (updated by
                // post-processing) instead of a per-frame dictionary lookup.
                tgt_xywh_curr = shift_subtract(x_max, y_max, tgt_xywh_last[2],
                                               tgt_xywh_last[3]);

                // Post-processing: ORB-based correction path
                kpts_result =
                    FilterKptsMode(matches, kpts_last, kpts_curr, dscrp_curr);
                boxfil_result =
                    FilterByBox(kpts_result.src_pts, kpts_result.dst_pts,
                                kpts_result.dst_dscrp, tgt_xywh_last);
                OrbMatch_result = MatchKptsCorrect(
                    boxfil_result.kp1_boxfiltered,
                    boxfil_result.kp2_boxfiltered, tgt_xywh_last);

                // Post-processing: similar triangle correction path
                kpts_result = FilterKpts(matches_refined, kpts_refined_last,
                                         kpts_curr, dscrp_curr);
                boxfil_result =
                    FilterByBox(kpts_result.src_pts, kpts_result.dst_pts,
                                kpts_result.dst_dscrp, tgt_xywh_last);
                smitri_check = CheckSmiTri(boxfil_result.kp1_boxfiltered,
                                           boxfil_result.kp2_boxfiltered);

                if (smitri_check.apply) {
                    // Convert to double precision for similar triangle
                    // computation
                    std::array<std::array<double, 2>, 3> long_src_pts;
                    std::array<std::array<double, 2>, 3> long_dst_pts;

                    for (int i = 0; i < 3; ++i) {
                        long_src_pts[i] = {
                            static_cast<double>(smitri_check.src_points[i][0]),
                            static_cast<double>(smitri_check.src_points[i][1])};
                        long_dst_pts[i] = {
                            static_cast<double>(smitri_check.dst_points[i][0]),
                            static_cast<double>(smitri_check.dst_points[i][1])};
                    }
                    // Apply similar triangle (SmiTri) correction
                    frangi_xyxy[0] = tgt_xywh_curr[0] + roi_tl_x;
                    frangi_xyxy[1] = tgt_xywh_curr[1] + roi_tl_y;
                    frangi_xyxy[2] =
                        tgt_xywh_curr[0] + roi_tl_x + tgt_xywh_curr[2];
                    frangi_xyxy[3] =
                        tgt_xywh_curr[1] + roi_tl_y + tgt_xywh_curr[3];

                    ctr_pt_last = {tgt_xywh_refined_last[0] +
                                       tgt_xywh_refined_last[2] / 2,
                                   tgt_xywh_refined_last[1] +
                                       tgt_xywh_refined_last[3] / 2};
                    std::array<double, 2> long_ctr_pt_last = {
                        static_cast<double>(ctr_pt_last[0]),
                        static_cast<double>(ctr_pt_last[1])};

                    new_ctr_pts =
                        SmiTri(long_src_pts, long_dst_pts, long_ctr_pt_last);
                    std::array<float, 2> fp_new_ctr_pts = {
                        static_cast<float>(new_ctr_pts[0]),
                        static_cast<float>(new_ctr_pts[1])};
                    corrected_xywh = CtrCorrect(fp_new_ctr_pts, frangi_xyxy,
                                                tgt_xywh_refined_last,
                                                smitri_check.dst_points);

                    tgt_xywh_refined_last = corrected_xywh;
                    kpts_refined_last = kpts_curr;
                    dscrp_refined_last = dscrp_curr;

                    tgt_xywh_last = corrected_xywh;
                    kpts_last = kpts_curr;
                    dscrp_last = dscrp_curr;
                    flame_signal_last = flame_signal_curr;
                } else {
                    // Use ORB match result without similar triangle correction
                    tgt_xywh_last = OrbMatch_result.tgt_xywh_curr_orb;
                    kpts_last = kpts_curr;
                    dscrp_last = dscrp_curr;
                    flame_signal_last = flame_signal_curr;
                }

                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> whole_time =
                    end - start;
                std::cout << "RUN time: " << whole_time.count() << " ms\n\n";

                // Write annotated frame to video
                {
                    cv::Mat vis;
                    cv::cvtColor(img_curr_np, vis, cv::COLOR_GRAY2BGR);
                    cv::rectangle(vis,
                                  cv::Point(static_cast<int>(tgt_xywh_last[0]),
                                            static_cast<int>(tgt_xywh_last[1])),
                                  cv::Point(static_cast<int>(tgt_xywh_last[0] +
                                                             tgt_xywh_last[2]),
                                            static_cast<int>(tgt_xywh_last[1] +
                                                             tgt_xywh_last[3])),
                                  cv::Scalar(0, 255, 0), 2);
                    writer.write(vis);
                }
            }
        }
    }

    writer.release();

    // Cleanup
    delete[] flame_cover_mask;
    delete[] orb_response;

    cudaStreamDestroy(stream1);
    cudaFree(img);
    cudaFree(response);
    cudaFree(x_max);
    cudaFree(y_max);
}
