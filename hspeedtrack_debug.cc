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

#include "types.h"

#include "init_engine.h"
#include "post_process/CtrCorrect.h"
#include "post_process/FilterByBox.h"
#include "post_process/FilterKpts.h"
#include "post_process/MatchKptsCorrect.h"
#include "post_process/SmiTri.h"
#include "post_process/ifSmiTri.h"
#include "utils/parallel_topk.h"
#include "utils/box_size.h"
#include "utils/descriptor_match.h"
#include "utils/get_roi.h"
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
int frame_count = 0;

int main() {
    bind_thread_to_core(0);
    omp_set_num_threads(NUM_THREADS);
    std::cout << std::fixed << std::setprecision(16);

    // Initialize TensorRT engine
    const std::string engine_path =
        "./engine_model/Norm_Grad_Response_Masked_Max_480.engine";
    auto engine_opt = init_engine(engine_path);
    if (!engine_opt) {
        std::cerr << "Fatal: failed to initialize engine" << std::endl;
        return 1;
    }
    INIT_engine& Grad_Response = *engine_opt;

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
    Grad_Response.context->setTensorAddress(Grad_Response.engine->getIOTensorName(0), img);
    Grad_Response.context->setTensorAddress(Grad_Response.engine->getIOTensorName(1), response);
    Grad_Response.context->setTensorAddress(Grad_Response.engine->getIOTensorName(2), x_max);
    Grad_Response.context->setTensorAddress(Grad_Response.engine->getIOTensorName(3), y_max);

    FilterKptsResult kpts_result;
    FilterByBoxResult boxfil_result;
    if_SmiTri_result IfSmiTri_result;
    std::array<double, 2> new_ctr_pts;
    Point ctr_pt_last;
    Box corrected_xywh;
    Box frangi_xyxy;
    MatchKptsCorrectResult OrbMatch_result;

    auto sorted_entries = get_sorted_image_entries("./test_imgs/");

    std::array<float, 4> tgt_xywh_curr, tgt_xywh_last, tgt_xywh_refined_last;
    std::array<Point, 40> kpts_for_patches, kpts_curr, kpts_last, kpts_refined_last;
    std::array<Descriptor, 40> dscrp_curr, dscrp_last, dscrp_refined_last;

    std::array<std::array<float, 3>, 40> matches;
    std::array<std::array<float, 3>, 40> matches_refined;

    bool new_box_signal = true;
    bool flame_signal_curr = true;
    bool flame_signal_last = false;

    uint8_t* roi = new uint8_t[ROI_SIZE * ROI_SIZE];
    float* flame_cover_mask = new float[ROI_SIZE * ROI_SIZE];
    float* orb_response = new float[ROI_SIZE * ROI_SIZE];

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat gpu_mask_inv(ROI_SIZE, ROI_SIZE, CV_32F);
    int roi_tl_x, roi_tl_y;

    std::array<std::pair<float, int>, TOPK> topk;
    std::array<std::array<std::array<float, 25>, 25>, 40> patches;

    cv::VideoWriter writer("output.mp4",
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
        cv::Size(static_cast<int>(IMG_WIDTH), static_cast<int>(IMG_HEIGHT)));

    {
        for (const auto& entry : sorted_entries) {
            std::string nimg = entry.path().stem().string();
            std::string img_path = entry.path().string();
            cv::Mat img_curr_np = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
            auto boxsz = BOXsz_dict.find(nimg);

            float w = boxsz->second[0];
            float h = boxsz->second[1];

            // Process only when new_box_signal is set
            if (!new_box_signal) {
                continue;
            }

            if (frame_count == 0) {
                // === First frame processing ===
                std::cout << nimg << std::endl;
                frame_count += 1;
                auto fstart = std::chrono::high_resolution_clock::now();
                int w_resize = static_cast<int>(w * (ROI_SIZE / IMG_WIDTH));
                int h_resize = static_cast<int>(h * (ROI_SIZE / IMG_HEIGHT));

                auto ftime1 = std::chrono::high_resolution_clock::now();
                cv::Mat resized_img;
                cv::resize(img_curr_np, resized_img, cv::Size(ROI_SIZE, ROI_SIZE), 0, 0, cv::INTER_AREA);

                auto ftime2 = std::chrono::high_resolution_clock::now();
                cv::Mat float_img;
                resized_img.convertTo(float_img, CV_32F);
                auto ftime3 = std::chrono::high_resolution_clock::now();
                std::memcpy(img, float_img.ptr<float>(), sizeof(float) * ROI_SIZE * ROI_SIZE);
                auto ftime4 = std::chrono::high_resolution_clock::now();

                // First TRT (TensorRT) inference
                Grad_Response.context->enqueueV3(stream1);
                cudaDeviceSynchronize();
                auto ftime5 = std::chrono::high_resolution_clock::now();

                // Prefix sum on x_max and y_max
                for (int i = 1; i < ROI_SIZE; ++i) {
                    x_max[i] += x_max[i - 1];
                }
                for (int i = 1; i < ROI_SIZE; ++i) {
                    y_max[i] += y_max[i - 1];
                }
                auto ftime6 = std::chrono::high_resolution_clock::now();

                // Shifted subtraction to find target location
                tgt_xywh_curr = shift_subtract(x_max, y_max, w_resize, h_resize);
                auto ftime7 = std::chrono::high_resolution_clock::now();

                // Scale back to original image coordinates
                tgt_xywh_curr[0] = static_cast<float>(std::round(tgt_xywh_curr[0] * (IMG_WIDTH / ROI_SIZE)));
                tgt_xywh_curr[1] = static_cast<float>(std::round(tgt_xywh_curr[1] * (IMG_HEIGHT / ROI_SIZE)));
                tgt_xywh_curr[2] = w;
                tgt_xywh_curr[3] = h;
                auto ftime8 = std::chrono::high_resolution_clock::now();

                // Extract ROI (Region of Interest) around target
                std::tie(roi_tl_x, roi_tl_y) = GetROI(roi, img_curr_np.ptr<uint8_t>(0), tgt_xywh_curr);
                auto ftime9 = std::chrono::high_resolution_clock::now();

                // Threshold to detect flame and convert to float
                threshold(roi, flame_cover_mask, img, flame_signal_curr);

                auto ftime10 = std::chrono::high_resolution_clock::now();

                // Second TRT inference on ROI
                Grad_Response.context->enqueueV3(stream1);
                cudaStreamSynchronize(stream1);
                auto ftime11 = std::chrono::high_resolution_clock::now();

                // Top-K keypoint extraction from response map
                topk = topk_sorted_parallel(response);
                auto ftime12 = std::chrono::high_resolution_clock::now();

                int i = 0;
                for (const auto& vi : topk) {
                    int index = vi.second;
                    int y = index / ROI_SIZE;
                    int x = index % ROI_SIZE;
                    kpts_for_patches[i] = {static_cast<float>(y), static_cast<float>(x)};
                    kpts_curr[i] = {static_cast<float>(y + roi_tl_y), static_cast<float>(x + roi_tl_x)};
                    ++i;
                }
                auto ftime13 = std::chrono::high_resolution_clock::now();

                // Extract patches and compute ORB (Oriented FAST and Rotated BRIEF) descriptors
                extract_all_patches(patches, img, kpts_for_patches);
                dscrp_curr = extract_descriptors(patches);
                auto ftime14 = std::chrono::high_resolution_clock::now();

                // Pass state to next frame
                tgt_xywh_last = tgt_xywh_curr;
                kpts_last = kpts_curr;
                dscrp_last = dscrp_curr;
                flame_signal_last = flame_signal_curr;

                tgt_xywh_refined_last = tgt_xywh_curr;
                kpts_refined_last = kpts_curr;
                dscrp_refined_last = dscrp_curr;
                auto fend = std::chrono::high_resolution_clock::now();

                // === Debug timing output for first frame ===
                std::chrono::duration<double, std::milli> fframe_time = fend - fstart;
                std::chrono::duration<double, std::milli> ftime1_s = ftime1 - fstart;
                std::chrono::duration<double, std::milli> ftime2_1 = ftime2 - ftime1;
                std::chrono::duration<double, std::milli> ftime3_2 = ftime3 - ftime2;
                std::chrono::duration<double, std::milli> ftime4_3 = ftime4 - ftime3;
                std::chrono::duration<double, std::milli> ftime5_4 = ftime5 - ftime4;
                std::chrono::duration<double, std::milli> ftime6_5 = ftime6 - ftime5;
                std::chrono::duration<double, std::milli> ftime7_6 = ftime7 - ftime6;
                std::chrono::duration<double, std::milli> ftime8_7 = ftime8 - ftime7;
                std::chrono::duration<double, std::milli> ftime9_8 = ftime9 - ftime8;
                std::chrono::duration<double, std::milli> ftime10_9 = ftime10 - ftime9;
                std::chrono::duration<double, std::milli> ftime11_10 = ftime11 - ftime10;
                std::chrono::duration<double, std::milli> ftime12_11 = ftime12 - ftime11;
                std::chrono::duration<double, std::milli> ftime13_12 = ftime13 - ftime12;
                std::chrono::duration<double, std::milli> ftime14_13 = ftime14 - ftime13;
                std::chrono::duration<double, std::milli> ftimeend_14 = fend - ftime14;

                std::cout << "First frame total time  : " << fframe_time.count() << " ms\n";
                std::cout << "Resize w/h              : " << ftime1_s.count() << " ms\n";
                std::cout << "Resize image (Mat)      : " << ftime2_1.count() << " ms\n";
                std::cout << "Convert to float (Mat)  : " << ftime3_2.count() << " ms\n";
                std::cout << "Copy to unified mem(Mat): " << ftime4_3.count() << " ms\n";
                std::cout << "1st resp TRT            : " << ftime5_4.count() << " ms\n";
                std::cout << "CUMSUM                  : " << ftime6_5.count() << " ms\n";
                std::cout << "Shifted subtraction     : " << ftime7_6.count() << " ms\n";
                std::cout << "Restore box to orig img : " << ftime8_7.count() << " ms\n";
                std::cout << "Get ROI                 : " << ftime9_8.count() << " ms\n";
                std::cout << "ROI operations          : " << ftime10_9.count() << " ms\n";
                std::cout << "2nd resp TRT            : " << ftime11_10.count() << " ms\n";
                std::cout << "TopK                    : " << ftime12_11.count() << " ms\n";
                std::cout << "Get keypoints           : " << ftime13_12.count() << " ms\n";
                std::cout << "Patches/descriptors     : " << ftime14_13.count() << " ms\n";
                std::cout << "Inter-frame param pass  : " << ftimeend_14.count() << " ms\n";

                // Write annotated frame to video (not included in timing)
                {
                    cv::Mat vis;
                    cv::cvtColor(img_curr_np, vis, cv::COLOR_GRAY2BGR);
                    cv::rectangle(vis,
                        cv::Point(static_cast<int>(tgt_xywh_curr[0]), static_cast<int>(tgt_xywh_curr[1])),
                        cv::Point(static_cast<int>(tgt_xywh_curr[0] + tgt_xywh_curr[2]),
                                  static_cast<int>(tgt_xywh_curr[1] + tgt_xywh_curr[3])),
                        cv::Scalar(0, 255, 0), 2);
                    writer.write(vis);
                }

            } else {
                // === Subsequent frame processing ===
                std::cout << nimg << std::endl;
                frame_count += 1;
                auto start = std::chrono::high_resolution_clock::now();

                // Extract ROI using last frame's target location
                std::tie(roi_tl_x, roi_tl_y) = GetROI(roi, img_curr_np.ptr<uint8_t>(0), tgt_xywh_last);
                auto time0_1 = std::chrono::high_resolution_clock::now();

                // Threshold and convert ROI
                threshold(roi, flame_cover_mask, img, flame_signal_curr);
                auto time0_2 = std::chrono::high_resolution_clock::now();

                auto time0_3 = std::chrono::high_resolution_clock::now();

                // If flame was present last frame but gone now, reset to first-frame mode
                if (flame_signal_last && !flame_signal_curr) {
                    frame_count = 0;
                    flame_signal_last = flame_signal_curr;
                    continue;
                }

                auto time0 = std::chrono::high_resolution_clock::now();
                auto time1_0 = std::chrono::high_resolution_clock::now();
                auto time1 = std::chrono::high_resolution_clock::now();

                // TRT inference on current ROI
                Grad_Response.context->enqueueV3(stream1);
                auto time2_0 = std::chrono::high_resolution_clock::now();
                auto time2_1 = std::chrono::high_resolution_clock::now();

                // Erode flame mask to suppress boundary noise
                cv::Mat flame_cover_mask_mat(ROI_SIZE, ROI_SIZE, CV_32F, flame_cover_mask);
                cv::erode(flame_cover_mask_mat, gpu_mask_inv, kernel);
                auto time2_2 = std::chrono::high_resolution_clock::now();
                auto time2_3 = std::chrono::high_resolution_clock::now();

                cudaStreamSynchronize(stream1);
                auto time2 = std::chrono::high_resolution_clock::now();

                auto time3_0 = std::chrono::high_resolution_clock::now();
                auto time3_1 = std::chrono::high_resolution_clock::now();

                // Apply mask to response (suppress flame and boundary regions)
                multiply(response, gpu_mask_inv.ptr<float>(), orb_response);

                auto time3 = std::chrono::high_resolution_clock::now();
                auto time4_0 = std::chrono::high_resolution_clock::now();

                // Top-K keypoint extraction from masked response
                topk = topk_sorted_parallel(orb_response);
                auto time4_1 = std::chrono::high_resolution_clock::now();

                int i = 0;
                for (const auto& vi : topk) {
                    int index = vi.second;
                    int y = index / ROI_SIZE;
                    int x = index % ROI_SIZE;
                    kpts_for_patches[i] = {static_cast<float>(y), static_cast<float>(x)};
                    kpts_curr[i] = {static_cast<float>(y + roi_tl_y), static_cast<float>(x + roi_tl_x)};
                    ++i;
                }
                auto time4_2 = std::chrono::high_resolution_clock::now();

                // Extract patches and compute descriptors
                extract_all_patches(patches, img, kpts_for_patches);
                auto time4_3 = std::chrono::high_resolution_clock::now();
                dscrp_curr = extract_descriptors(patches);
                auto time4_4 = std::chrono::high_resolution_clock::now();

                // Match descriptors between consecutive frames
                matches = match_descriptors(dscrp_last, dscrp_curr);
                auto time4_5 = std::chrono::high_resolution_clock::now();
                matches_refined = match_descriptors(dscrp_refined_last, dscrp_curr);
                auto time4_6 = std::chrono::high_resolution_clock::now();

                // Prefix sum for cumulative response
                for (int i = 1; i < ROI_SIZE; ++i) {
                    x_max[i] += x_max[i - 1];
                }
                for (int i = 1; i < ROI_SIZE; ++i) {
                    y_max[i] += y_max[i - 1];
                }

                auto time4 = std::chrono::high_resolution_clock::now();

                // Shifted subtraction for target localization
                tgt_xywh_curr = shift_subtract(x_max, y_max, w, h);

                // Post-processing: ORB-based correction path
                auto time5 = std::chrono::high_resolution_clock::now();
                kpts_result = FilterKptsMode(matches, kpts_last, kpts_curr, dscrp_curr);
                boxfil_result = FilterByBox(kpts_result.src_pts, kpts_result.dst_pts, kpts_result.dst_dscrp, tgt_xywh_last);
                OrbMatch_result = MatchKptsCorrect(boxfil_result.kp1_boxfiltered, boxfil_result.kp2_boxfiltered, tgt_xywh_last);

                // Post-processing: similar triangle correction path
                auto time6 = std::chrono::high_resolution_clock::now();
                kpts_result = FilterKpts(matches_refined, kpts_refined_last, kpts_curr, dscrp_curr);
                boxfil_result = FilterByBox(kpts_result.src_pts, kpts_result.dst_pts, kpts_result.dst_dscrp, tgt_xywh_last);
                IfSmiTri_result = IfSmiTri(boxfil_result.kp1_boxfiltered, boxfil_result.kp2_boxfiltered);

                // Convert to double precision for similar triangle computation
                auto time7 = std::chrono::high_resolution_clock::now();
                std::array<std::array<double, 2>, 3> long_src_pts;
                std::array<std::array<double, 2>, 3> long_dst_pts;

                for (int i = 0; i < 3; ++i) {
                    long_src_pts[i] = {
                        static_cast<double>(IfSmiTri_result.src_points[i][0]),
                        static_cast<double>(IfSmiTri_result.src_points[i][1])
                    };
                    long_dst_pts[i] = {
                        static_cast<double>(IfSmiTri_result.dst_points[i][0]),
                        static_cast<double>(IfSmiTri_result.dst_points[i][1])
                    };
                }

                auto time8 = std::chrono::high_resolution_clock::now();
                if (IfSmiTri_result.choose && new_box_signal) {
                    // Apply similar triangle (SmiTri) correction
                    frangi_xyxy[0] = tgt_xywh_curr[0] + roi_tl_x;
                    frangi_xyxy[1] = tgt_xywh_curr[1] + roi_tl_y;
                    frangi_xyxy[2] = tgt_xywh_curr[0] + roi_tl_x + tgt_xywh_curr[2];
                    frangi_xyxy[3] = tgt_xywh_curr[1] + roi_tl_y + tgt_xywh_curr[3];

                    ctr_pt_last = {tgt_xywh_refined_last[0] + tgt_xywh_refined_last[2] / 2,
                                   tgt_xywh_refined_last[1] + tgt_xywh_refined_last[3] / 2};
                    std::array<double, 2> long_ctr_pt_last = {
                        static_cast<double>(ctr_pt_last[0]),
                        static_cast<double>(ctr_pt_last[1])};

                    new_ctr_pts = SmiTri(long_src_pts, long_dst_pts, long_ctr_pt_last);
                    std::array<float, 2> fp_new_ctr_pts = {
                        static_cast<float>(new_ctr_pts[0]),
                        static_cast<float>(new_ctr_pts[1])};
                    corrected_xywh = CtrCorrect(fp_new_ctr_pts, frangi_xyxy, tgt_xywh_refined_last, IfSmiTri_result.dst_points);
                    std::cout << "Corrected" << std::endl;
                    std::cout << "Corrected box [" << corrected_xywh[0] << ", " << corrected_xywh[1] << ", " << corrected_xywh[2] << ", " << corrected_xywh[3] << "]\n";

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
                    std::cout << "[" << OrbMatch_result.tgt_xywh_curr_orb[0] << ", " << OrbMatch_result.tgt_xywh_curr_orb[1] << ", " << OrbMatch_result.tgt_xywh_curr_orb[2] << ", " << OrbMatch_result.tgt_xywh_curr_orb[3] << "]\n";

                    tgt_xywh_last = OrbMatch_result.tgt_xywh_curr_orb;
                    kpts_last = kpts_curr;
                    dscrp_last = dscrp_curr;
                    flame_signal_last = flame_signal_curr;
                }
                auto end = std::chrono::high_resolution_clock::now();

                // === Debug timing output for subsequent frames ===
                std::chrono::duration<double, std::milli> whole_time = end - start;
                std::chrono::duration<double, std::milli> time_0start = time0 - start;
                std::chrono::duration<double, std::milli> time_01_start = time0_1 - start;
                std::chrono::duration<double, std::milli> time_02_01 = time0_2 - time0_1;
                std::chrono::duration<double, std::milli> time_03_02 = time0_3 - time0_2;
                std::chrono::duration<double, std::milli> time_0_03 = time0 - time0_3;

                std::chrono::duration<double, std::milli> time_10 = time1 - time0;
                std::chrono::duration<double, std::milli> time_10_0 = time1_0 - time0;
                std::chrono::duration<double, std::milli> time_1_10 = time1 - time1_0;

                std::chrono::duration<double, std::milli> time_21 = time2 - time1;
                std::chrono::duration<double, std::milli> time_20_1 = time2_0 - time1;
                std::chrono::duration<double, std::milli> time_21_20 = time2_1 - time2_0;
                std::chrono::duration<double, std::milli> time_22_21 = time2_2 - time2_1;
                std::chrono::duration<double, std::milli> time_23_22 = time2_3 - time2_2;
                std::chrono::duration<double, std::milli> time_2_23 = time2 - time2_3;

                std::chrono::duration<double, std::milli> time_32 = time3 - time2;
                std::chrono::duration<double, std::milli> time_30_2 = time3_0 - time2;
                std::chrono::duration<double, std::milli> time_31_30 = time3_1 - time3_0;
                std::chrono::duration<double, std::milli> time_3_31 = time3 - time3_1;

                std::chrono::duration<double, std::milli> time_43 = time4 - time3;
                std::chrono::duration<double, std::milli> time_40_3 = time4_0 - time3;
                std::chrono::duration<double, std::milli> time_41_40 = time4_1 - time4_0;
                std::chrono::duration<double, std::milli> time_42_41 = time4_2 - time4_1;
                std::chrono::duration<double, std::milli> time_43_42 = time4_3 - time4_2;
                std::chrono::duration<double, std::milli> time_44_43 = time4_4 - time4_3;
                std::chrono::duration<double, std::milli> time_45_44 = time4_5 - time4_4;
                std::chrono::duration<double, std::milli> time_46_45 = time4_6 - time4_5;
                std::chrono::duration<double, std::milli> time_4_46 = time4 - time4_6;

                std::chrono::duration<double, std::milli> time_54 = time5 - time4;
                std::chrono::duration<double, std::milli> time_65 = time6 - time5;
                std::chrono::duration<double, std::milli> time_76 = time7 - time6;
                std::chrono::duration<double, std::milli> time_87 = time8 - time7;
                std::chrono::duration<double, std::milli> time_end8 = end - time8;

                std::cout << "RUN time: " << whole_time.count() << " ms\n";
                std::cout << "Get ROI -> lost check   : " << time_0start.count() << " ms\n";
                std::cout << "      Get ROI           : " << time_01_start.count() << " ms\n";
                std::cout << "      Threshold binarize: " << time_02_01.count() << " ms\n";
                std::cout << "      Check for flame   : " << time_03_02.count() << " ms\n";
                std::cout << "      Branch statement  : " << time_0_03.count() << " ms\n";

                std::cout << "Normalize -> TRT input  : " << time_10.count() << " ms\n";
                std::cout << "      Normalize         : " << time_10_0.count() << " ms\n";
                std::cout << "      Memory copy       : " << time_1_10.count() << " ms\n";

                std::cout << "Resp TRT || erode mask  : " << time_21.count() << " ms\n";
                std::cout << "      Resp TRT run      : " << time_20_1.count() << " ms\n";
                std::cout << "      Mask to fp        : " << time_21_20.count() << " ms\n";
                std::cout << "      Erode mask        : " << time_22_21.count() << " ms\n";
                std::cout << "      Invert mask       : " << time_23_22.count() << " ms\n";
                std::cout << "      CUDA sync         : " << time_2_23.count() << " ms\n";

                std::cout << "Get TRT output->mask    : " << time_32.count() << " ms\n";
                std::cout << "      Ptr get TRT output: " << time_30_2.count() << " ms\n";
                std::cout << "      Data to mat       : " << time_31_30.count() << " ms\n";
                std::cout << "      Mat * mask        : " << time_3_31.count() << " ms\n";

                std::cout << "Cum TRT || TPDM         : " << time_43.count() << " ms\n";
                std::cout << "      TopK              : " << time_41_40.count() << " ms\n";
                std::cout << "      Descriptor        : " << time_44_43.count() << " ms\n";
                std::cout << "      Match1            : " << time_45_44.count() << " ms\n";
                std::cout << "      Match2            : " << time_46_45.count() << " ms\n";
                std::cout << "      Cumsum            : " << time_4_46.count() << " ms\n";

                std::cout << "Shifted subtraction     : " << time_54.count() << " ms\n";
                std::cout << "ORB post-process        : " << time_65.count() << " ms\n";
                std::cout << "SmiTri decision         : " << time_76.count() << " ms\n";
                std::cout << "Prepare double precision: " << time_87.count() << " ms\n";
                std::cout << "SmiTri select output    : " << time_end8.count() << " ms\n";
                std::cout << "\n";

                // Write annotated frame to video (not included in timing)
                {
                    cv::Mat vis;
                    cv::cvtColor(img_curr_np, vis, cv::COLOR_GRAY2BGR);
                    cv::rectangle(vis,
                        cv::Point(static_cast<int>(tgt_xywh_last[0]), static_cast<int>(tgt_xywh_last[1])),
                        cv::Point(static_cast<int>(tgt_xywh_last[0] + tgt_xywh_last[2]),
                                  static_cast<int>(tgt_xywh_last[1] + tgt_xywh_last[3])),
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
