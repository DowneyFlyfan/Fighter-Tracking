#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <sched.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "opencv2/core/hal/interface.h"
#include "utils/types.h"

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
#include "utils/utils.h"

using namespace nvinfer1;

void bind_thread_to_core(int core_id) {
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
    std::cout << std::fixed << std::setprecision(6);

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
            // promote it to current. Otherwise (first iter or post-continue),
            // synchronously read.
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
            using ms = std::chrono::duration<double, std::milli>;
            auto now = std::chrono::high_resolution_clock::now;

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

                std::cout << nimg << std::endl;
                frame_count += 1;
                auto fstart = now();
                int w_resize = static_cast<int>(w * (ROI_SIZE / IMG_WIDTH));
                int h_resize = static_cast<int>(h * (ROI_SIZE / IMG_HEIGHT));

                auto t_after_resize_wh = now();
                cv::Mat resized_img;
                cv::resize(img_curr_np, resized_img,
                           cv::Size(ROI_SIZE, ROI_SIZE), 0, 0, cv::INTER_AREA);

                auto t_after_resize_img = now();
                cv::Mat float_img;
                resized_img.convertTo(float_img, CV_32F);
                auto t_after_convert = now();
                std::memcpy(img, float_img.ptr<float>(),
                            sizeof(float) * ROI_SIZE * ROI_SIZE);
                auto t_after_memcpy = now();

                // TRT
                Grad_Response.context->enqueueV3(stream1);
                cudaMemPrefetchAsync(x_max, ROI_SIZE * sizeof(float), loc, 0,
                                     stream1);
                cudaMemPrefetchAsync(y_max, ROI_SIZE * sizeof(float), loc, 0,
                                     stream1);
                cudaStreamSynchronize(stream1);
                auto t_after_1st_trt = now();

                // Prefix sum on x_max and y_max
                for (int i = 1; i < ROI_SIZE; ++i) {
                    x_max[i] += x_max[i - 1];
                }
                for (int i = 1; i < ROI_SIZE; ++i) {
                    y_max[i] += y_max[i - 1];
                }
                auto t_after_cumsum = now();

                // Shifted subtraction to find target location
                tgt_xywh_curr =
                    shift_subtract(x_max, y_max, w_resize, h_resize);
                auto t_after_shift_sub = now();

                // Scale back to original image coordinates
                tgt_xywh_curr[0] = static_cast<float>(
                    std::round(tgt_xywh_curr[0] * (IMG_WIDTH / ROI_SIZE)));
                tgt_xywh_curr[1] = static_cast<float>(
                    std::round(tgt_xywh_curr[1] * (IMG_HEIGHT / ROI_SIZE)));
                tgt_xywh_curr[2] = w;
                tgt_xywh_curr[3] = h;
                auto t_after_rescale = now();

                // Extract ROI (Region of Interest) around target
                std::tie(roi_tl_x, roi_tl_y) =
                    GetROI(roi, img_curr_np.ptr<uint8_t>(0), tgt_xywh_curr);
                auto t_after_get_roi = now();

                cudaMemPrefetchAsync(img, ROI_SIZE * ROI_SIZE * sizeof(float),
                                     loc, 0, stream1);
                cudaStreamSynchronize(stream1);

                // Threshold to detect flame and convert to float
                threshold(roi, flame_cover_mask, img, flame_signal_curr);

                auto t_after_thresh = now();

                // Second TRT inference on ROI, then prefetch response to CPU
                Grad_Response.context->enqueueV3(stream1);
                cudaMemPrefetchAsync(response,
                                     ROI_SIZE * ROI_SIZE * sizeof(float), loc,
                                     0, stream1);
                // Overlap: read next frame from disk while GPU runs 2nd TRT
                // (excluded from pipeline timing — measured separately)
                double imread_next_ms = 0.0;
                if (has_next) {
                    auto t_imread0 = now();
                    img_next_np =
                        cv::imread(next_it->path().string(),
                                   cv::IMREAD_GRAYSCALE);
                    nimg_next = next_it->path().stem().string();
                    imread_next_ms = ms(now() - t_imread0).count();
                }
                cudaStreamSynchronize(stream1);
                auto t_after_2nd_trt = now();

                // Top-K keypoint extraction from response map
                topk = topk_sorted_parallel(response);
                auto t_after_topk = now();

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
                auto t_after_kpts = now();

                // Extract patches and compute ORB (Oriented FAST and Rotated
                // BRIEF) descriptors
                extract_all_patches(patches, img, kpts_for_patches);
                dscrp_curr = extract_descriptors(patches);
                auto t_after_dscrp = now();

                // Pass state to next frame
                tgt_xywh_last = tgt_xywh_curr;
                kpts_last = kpts_curr;
                dscrp_last = dscrp_curr;
                flame_signal_last = flame_signal_curr;

                tgt_xywh_refined_last = tgt_xywh_curr;
                kpts_refined_last = kpts_curr;
                dscrp_refined_last = dscrp_curr;
                auto fend = now();

                // === Debug timing output for first frame ===
                // Subtract imread_next_ms: imread of next frame is overlapped
                // with GPU but should not count toward this frame
                ms fframe_time = fend - fstart;
                std::cout << "First frame total time  : "
                          << (fframe_time.count() - imread_next_ms)
                          << " ms (imread next: " << imread_next_ms << " ms)\n";
                std::cout << "  Resize w/h            : "
                          << ms(t_after_resize_wh - fstart).count() << " ms\n";
                std::cout << "  Resize image          : "
                          << ms(t_after_resize_img - t_after_resize_wh).count()
                          << " ms\n";
                std::cout << "  Convert to float      : "
                          << ms(t_after_convert - t_after_resize_img).count()
                          << " ms\n";
                std::cout << "  Copy to unified mem   : "
                          << ms(t_after_memcpy - t_after_convert).count()
                          << " ms\n";
                std::cout << "  1st TRT + sync        : "
                          << ms(t_after_1st_trt - t_after_memcpy).count()
                          << " ms\n";
                std::cout << "  Cumsum                : "
                          << ms(t_after_cumsum - t_after_1st_trt).count()
                          << " ms\n";
                std::cout << "  Shift subtraction     : "
                          << ms(t_after_shift_sub - t_after_cumsum).count()
                          << " ms\n";
                std::cout << "  Rescale to orig coords: "
                          << ms(t_after_rescale - t_after_shift_sub).count()
                          << " ms\n";
                std::cout << "  GetROI                : "
                          << ms(t_after_get_roi - t_after_rescale).count()
                          << " ms\n";
                std::cout << "  Threshold + flame chk : "
                          << ms(t_after_thresh - t_after_get_roi).count()
                          << " ms\n";
                std::cout << "  2nd TRT + sync        : "
                          << (ms(t_after_2nd_trt - t_after_thresh).count() -
                              imread_next_ms)
                          << " ms\n";
                std::cout << "  TopK                  : "
                          << ms(t_after_topk - t_after_2nd_trt).count()
                          << " ms\n";
                std::cout << "  Get keypoints         : "
                          << ms(t_after_kpts - t_after_topk).count() << " ms\n";
                std::cout << "  Patches + descriptors : "
                          << ms(t_after_dscrp - t_after_kpts).count()
                          << " ms\n";
                std::cout << "  State pass to next frm: "
                          << ms(fend - t_after_dscrp).count() << " ms\n";

                // Write annotated frame to video (not included in timing)
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
                std::cout << nimg << std::endl;
                frame_count += 1;
                auto start = now();
                std::tie(roi_tl_x, roi_tl_y) =
                    GetROI(roi, img_curr_np.ptr<uint8_t>(0), tgt_xywh_last);

                auto t_after_roi = now();

                // Threshold and convert ROI
                threshold(roi, flame_cover_mask, img, flame_signal_curr);
                auto t_after_thresh = now();

                // If flame was present last frame but gone now, reset to
                // first-frame mode
                if (flame_signal_last && !flame_signal_curr) {
                    frame_count = 0;
                    flame_signal_last = flame_signal_curr;
                    continue;
                }

                // TRT inference on current ROI, then prefetch outputs to CPU
                Grad_Response.context->enqueueV3(stream1);
                cudaMemPrefetchAsync(response,
                                     ROI_SIZE * ROI_SIZE * sizeof(float), loc,
                                     0, stream1);
                cudaMemPrefetchAsync(x_max, ROI_SIZE * sizeof(float), loc, 0,
                                     stream1);
                cudaMemPrefetchAsync(y_max, ROI_SIZE * sizeof(float), loc, 0,
                                     stream1);
                cudaMemPrefetchAsync(img, ROI_SIZE * ROI_SIZE * sizeof(float),
                                     loc, 0, stream1);

                auto t_after_enqueue = now();

                // Erode flame mask on CPU while GPU inference + prefetch run
                cv::erode(cv::Mat(ROI_SIZE, ROI_SIZE, CV_32F, flame_cover_mask),
                          eroded_mask, kernel);
                auto t_after_erode = now();

                // Read next frame from disk during GPU window (excluded from
                // pipeline timing — measured separately)
                double imread_next_ms = 0.0;
                if (has_next) {
                    auto t_imread0 = now();
                    img_next_np =
                        cv::imread(next_it->path().string(),
                                   cv::IMREAD_GRAYSCALE);
                    nimg_next = next_it->path().stem().string();
                    imread_next_ms = ms(now() - t_imread0).count();
                }

                // Wait for inference + prefetch to complete
                cudaStreamSynchronize(stream1);
                auto t_after_sync = now();

                // Apply mask to response (suppress flame and boundary regions)
                multiply(response, eroded_mask.ptr<float>(), orb_response);
                auto t_after_multiply = now();

                // Top-K keypoint extraction from masked response
                topk = topk_sorted_parallel(orb_response);
                auto t_after_topk = now();

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
                auto t_after_patches = now();
                dscrp_curr = extract_descriptors(patches);
                auto t_after_dscrp = now();

                // Match descriptors between consecutive frames
                matches = match_descriptors(dscrp_last, dscrp_curr);
                auto t_after_match1 = now();
                matches_refined =
                    match_descriptors(dscrp_refined_last, dscrp_curr);
                auto t_after_match2 = now();

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
                auto t_after_cumsum = now();

                // Post-processing: ORB-based correction path
                kpts_result =
                    FilterKptsMode(matches, kpts_last, kpts_curr, dscrp_curr);
                boxfil_result =
                    FilterByBox(kpts_result.src_pts, kpts_result.dst_pts,
                                kpts_result.dst_dscrp, tgt_xywh_last);
                OrbMatch_result = MatchKptsCorrect(
                    boxfil_result.kp1_boxfiltered,
                    boxfil_result.kp2_boxfiltered, tgt_xywh_last);
                auto t_after_orb = now();

                // Post-processing: similar triangle correction path
                kpts_result = FilterKpts(matches_refined, kpts_refined_last,
                                         kpts_curr, dscrp_curr);
                boxfil_result =
                    FilterByBox(kpts_result.src_pts, kpts_result.dst_pts,
                                kpts_result.dst_dscrp, tgt_xywh_last);
                smitri_check = CheckSmiTri(boxfil_result.kp1_boxfiltered,
                                           boxfil_result.kp2_boxfiltered);
                auto t_after_smitri_check = now();
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
                    std::cout << "Corrected" << std::endl;
                    std::cout << "Corrected box [" << corrected_xywh[0] << ", "
                              << corrected_xywh[1] << ", " << corrected_xywh[2]
                              << ", " << corrected_xywh[3] << "]\n";

                    tgt_xywh_refined_last = corrected_xywh;
                    kpts_refined_last = kpts_curr;
                    dscrp_refined_last = dscrp_curr;

                    tgt_xywh_last = corrected_xywh;
                    kpts_last = kpts_curr;
                    dscrp_last = dscrp_curr;
                    flame_signal_last = flame_signal_curr;
                } else {
                    // Use ORB match result without similar triangle correction
                    std::cout << "Not corrected" << std::endl;
                    std::cout << "[" << OrbMatch_result.tgt_xywh_curr_orb[0]
                              << ", " << OrbMatch_result.tgt_xywh_curr_orb[1]
                              << ", " << OrbMatch_result.tgt_xywh_curr_orb[2]
                              << ", " << OrbMatch_result.tgt_xywh_curr_orb[3]
                              << "]\n";

                    tgt_xywh_last = OrbMatch_result.tgt_xywh_curr_orb;
                    kpts_last = kpts_curr;
                    dscrp_last = dscrp_curr;
                    flame_signal_last = flame_signal_curr;
                }
                auto end = now();

                // === Debug timing output for subsequent frames ===
                // Subtract imread_next_ms: imread of next frame is overlapped
                // with GPU but should not count toward this frame's RUN time
                std::cout << "RUN time: "
                          << (ms(end - start).count() - imread_next_ms)
                          << " ms (imread next: " << imread_next_ms << " ms)\n";
                std::cout << "  GetROI                : "
                          << ms(t_after_roi - start).count() << " ms\n";
                std::cout << "  Threshold + flame chk : "
                          << ms(t_after_thresh - t_after_roi).count()
                          << " ms\n";
                std::cout << "  TRT enqueue + prefetch: "
                          << ms(t_after_enqueue - t_after_thresh).count()
                          << " ms\n";
                std::cout << "  Erode mask (CPU||GPU) : "
                          << ms(t_after_erode - t_after_enqueue).count()
                          << " ms\n";
                std::cout << "  CUDA sync             : "
                          << (ms(t_after_sync - t_after_erode).count() -
                              imread_next_ms)
                          << " ms\n";
                std::cout << "  Multiply resp * mask  : "
                          << ms(t_after_multiply - t_after_sync).count()
                          << " ms\n";
                std::cout << "  TopK                  : "
                          << ms(t_after_topk - t_after_multiply).count()
                          << " ms\n";
                std::cout << "  Extract patches       : "
                          << ms(t_after_patches - t_after_topk).count()
                          << " ms\n";
                std::cout << "  Extract descriptors   : "
                          << ms(t_after_dscrp - t_after_patches).count()
                          << " ms\n";
                std::cout << "  Match descriptors 1   : "
                          << ms(t_after_match1 - t_after_dscrp).count()
                          << " ms\n";
                std::cout << "  Match descriptors 2   : "
                          << ms(t_after_match2 - t_after_match1).count()
                          << " ms\n";
                std::cout << "  Cumsum + shift sub    : "
                          << ms(t_after_cumsum - t_after_match2).count()
                          << " ms\n";
                std::cout << "  ORB post-process      : "
                          << ms(t_after_orb - t_after_cumsum).count()
                          << " ms\n";
                std::cout << "  SmiTri check          : "
                          << ms(t_after_smitri_check - t_after_orb).count()
                          << " ms\n";
                std::cout << "  SmiTri apply + output : "
                          << ms(end - t_after_smitri_check).count() << " ms\n";
                std::cout << "\n";

                // Write annotated frame to video (not included in timing)
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
