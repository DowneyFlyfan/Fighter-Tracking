#pragma once

#include "../utils/mosse_dcf.h"
#include "../utils/types.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <optional>
#include <vector>

// Motion + DoH hybrid tracker. Brightness only as side helper.
//
// PRIMARY DETECTOR  = inter-frame motion mask (camera-motion-
// compensated frame diff). drone moves -> mask fires; sun / static
// distractors -> mask 0 after warp. Brightness-independent.
//
// FALLBACK DETECTOR = multi-scale DoH (used when motion mask is
// empty, e.g. drone hovering or first frames before history fills).
//
// SCORING per blob = size_match * Mahalanobis gating (no brightness).
//
// Position measurement: chosen blob's CENTROID (geometric center,
// motion-mask-derived = full moving silhouette, NOT bright spot).
// Size measurement: brightness Otsu on a 60x60 window around centroid
// (only run AFTER motion picks the right place; its job is just to
// refine the bbox extent).
//
// 4-state Kalman: (cx, cy, vx, vy). Size lives in init_w_/init_h_
// EMA-updated externally - keeps Kalman simple, decouples noisy
// per-frame size measurement from position smoothing.
class KalmanScoredTracker {
    // TODO: Reduce Engineering Numbers
  public:
    static constexpr int SEARCH_RADIUS = 160;
    static constexpr float GATING_CHI2 = 25.0f;
    static constexpr int MIN_BLOB_AREA = 4;
    static constexpr int MAX_BLOB_AREA = 4000;
    static constexpr float SIZE_SIGMA_FRAC = 0.6f;
    // Lower alpha = smoother size; was 0.15. 0.05 reduces frame-to-
    // frame size jitter (output box size changes ~5% per detected
    // full-drone frame instead of 15%).
    static constexpr float SIZE_EMA_ALPHA = 0.05f;

    // DoH (multi-scale Gaussian Hessian). DOH_FRAC * doh_anchor_
    // = mask threshold. anchor auto-calibrated per video (drone's
    // own DoH peak in frame 0). 0.4 keeps drone signal while
    // suppressing flat-region floating-point noise (DoH = 0
    // exactly there but small positive due to convolution rounding).
    // Multi-scale DoH for Lindeberg automatic scale selection.
    // sigmas {1, 2, 4, 8, 16} -> drone diameter range ~3..45 px (covers
    // Anti-UAV / fighter scale span). Per-pixel argmax over scales gives
    // current blob radius via Lindeberg's relation r = sqrt(2)*sigma_best.
    static constexpr int N_SCALES = 3;
    static constexpr float DOH_FRAC = 0.4f;

    // Motion (primary detector).
    static constexpr int STACK_GAP = 3;           // diff frames apart
    static constexpr int MOTION_DIFF_THRESH = 12; // per-pixel uint8

    // Camera ego-motion compensation. Centered phaseCorrelate window;
    // shifts above EGO_MAX_SHIFT (per-frame) are rejected as outliers
    // (likely caused by moving distractors hijacking phase peak).
    // 4-corner ego-motion: each corner is EGO_WIN x EGO_WIN; clamped
    // to half the smaller frame dim so the four corners don't overlap.
    static constexpr int   EGO_WIN       = 192;
    static constexpr float EGO_RESP_MIN  = 0.05f;
    static constexpr float EGO_MAX_SHIFT = 30.f;

    // delta t = 1 Frame
    KalmanScoredTracker() : kf_(4, 2, 0) {
        // 4-state, 2-measurement: state = (cx, cy, vx, vy);
        // measurement = (cx, cy). Size lives in init_w_/init_h_
        // EMA-updated externally (decouples noisy per-frame size from
        // position smoothing).
        // F: constant-velocity transition.
        kf_.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1,
                                0, 0, 1, 0, 0, 0, 0, 1);
        // H: pick cx, cy from state.
        kf_.measurementMatrix =
            (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
        // Q (process noise): position 5, velocity 50 (looser for
        // fast camera motion).
        kf_.processNoiseCov = (cv::Mat_<float>(4, 4) << 5.f, 0, 0, 0, 0, 5.f, 0,
                               0, 0, 0, 50.f, 0, 0, 0, 0, 50.f);
        // R (measurement noise) = 1. Trusts the measurement strongly
        // -> high responsiveness to fast / oscillating drone motion
        // (interference6 has direction reversals every ~20 frames;
        // R=4 was too sluggish and let the tracker lock distractors).
        kf_.measurementNoiseCov = (cv::Mat_<float>(2, 2) << 1.f, 0, 0, 1.f);
        cv::setIdentity(kf_.errorCovPost, cv::Scalar(10.0));

        // CLAHE (Contrast Limited Adaptive Histogram Equalisation):
        // local-tile histogram equalisation that boosts dark / low-
        // contrast targets (drone) without saturating already-bright
        // distractors. Runs once per frame, ~0.3 ms on 480x640.
        clahe_ = cv::createCLAHE(2.0, cv::Size(8, 8));

        // sigmas form an octave doubling sequence; ksize ~ 6*sigma+1.
        const double all_sigmas[3] = {1.0, 2.0, 4.0};
        const int    all_ksizes[3] = {7,   13,  25};
        for (int s = 0; s < N_SCALES; ++s) {
            sigmas_[s] = static_cast<float>(all_sigmas[s]);
            sigma4_norm_[s] = static_cast<float>(std::pow(all_sigmas[s], 4));
            // Separable 1D kernels (row = func of x, col = func of y).
            // Replaces 2D filter2D with sepFilter2D: 25x25 -> 25+25,
            // ~30x fewer multiplies. Mathematically identical.
            make_sep_kernels_(all_sigmas[s], all_ksizes[s],
                              gxx_x_[s], gxx_y_[s],
                              gyy_x_[s], gyy_y_[s],
                              gxy_x_[s], gxy_y_[s]);
        }
    }

    // Build the frame-0 drone templates:
    //  - HOG-144d (gradient orientation, brightness-shift invariant)
    //  - MOSSE DCF (correlation filter, frozen + online updated).
    void build_template(const cv::Mat &frame0, const Box &init_xywh) {
        cv::Mat frame0_eq;
        clahe_->apply(frame0, frame0_eq);     // consistent with update()
        const int x =
            std::clamp(static_cast<int>(init_xywh[0]), 0, frame0_eq.cols - 1);
        const int y =
            std::clamp(static_cast<int>(init_xywh[1]), 0, frame0_eq.rows - 1);
        const int w =
            std::clamp(static_cast<int>(init_xywh[2]), 1, frame0_eq.cols - x);
        const int h =
            std::clamp(static_cast<int>(init_xywh[3]), 1, frame0_eq.rows - y);
        const cv::Mat patch = frame0_eq(cv::Rect(x, y, w, h));
        hog_tmpl_ = compute_hog(patch);
        has_hog_tmpl_ = true;
        const cv::Point2f c(init_xywh[0] + init_xywh[2] / 2.f,
                            init_xywh[1] + init_xywh[3] / 2.f);
        dcf_.init(frame0_eq, c);
    }

    void init(const Box &init_xywh) {
        const float cx = init_xywh[0] + init_xywh[2] / 2.f;
        const float cy = init_xywh[1] + init_xywh[3] / 2.f;
        // 4-state init: (cx, cy, vx=0, vy=0).
        kf_.statePost = (cv::Mat_<float>(4, 1) << cx, cy, 0, 0);
        init_w_ = init_xywh[2];
        init_h_ = init_xywh[3];
        ref_area_ = std::max(init_w_ * init_h_, 4.f);
        init_box_ = init_xywh;
        doh_anchor_ = -1.f;
        frame_history_.clear();
        lost_frames_ = 0;
    }

    Box update(const cv::Mat &gray_frame_in) {
        // CLAHE enhance: lifts dark / low-contrast targets so DoH /
        // DCF / HOG see drone on equal footing with bright distractors.
        cv::Mat gray_frame;
        clahe_->apply(gray_frame_in, gray_frame);
        // 0. Camera ego-motion compensation. phaseCorrelate(prev, cur)
        //    on a centered window estimates global frame-to-frame
        //    translation (background dominates -> measures camera pan).
        //    Apply to Kalman state so predict() advances by drone's
        //    world-frame velocity, not image-frame velocity (which
        //    would mix drone + camera motion and cause Kalman to over-
        //    shoot whenever camera motion changes direction).
        cv::Point2f cam_shift(0.f, 0.f);
        if (!frame_history_.empty() &&
            frame_history_.back().size() == gray_frame.size()) {
            // 4-corner phaseCorrelate + median. Each EGO_WIN x EGO_WIN
            // patch sits in a corner -> avoids both the drone (centred)
            // and most moving distractors (which cluster around drone
            // in interference scenes). Median of accepted estimates
            // is robust to a single bad corner (e.g. distractor
            // crossing it). Captures pure background camera pan.
            const int ew = std::min(EGO_WIN,
                                     std::min(gray_frame.cols,
                                              gray_frame.rows) / 2);
            const int W = gray_frame.cols, H = gray_frame.rows;
            const int cx0 = (W - ew) / 2, cy0 = (H - ew) / 2;
            const cv::Rect rois[5] = {
                cv::Rect(0,           0,           ew, ew), // TL
                cv::Rect(W - ew,      0,           ew, ew), // TR
                cv::Rect(0,           H - ew,      ew, ew), // BL
                cv::Rect(W - ew,      H - ew,      ew, ew), // BR
                cv::Rect(cx0,         cy0,         ew, ew)  // C
            };
            const cv::Mat &prev = frame_history_.back();
            std::vector<float> sxs, sys;
            sxs.reserve(4); sys.reserve(4);
            for (const auto &r : rois) {
                cv::Mat cur_f, prev_f;
                gray_frame(r).convertTo(cur_f, CV_32F);
                prev(r).convertTo(prev_f, CV_32F);
                double resp = 0.0;
                cv::Point2d s = cv::phaseCorrelate(prev_f, cur_f,
                                                    cv::noArray(), &resp);
                const float sx = static_cast<float>(s.x);
                const float sy = static_cast<float>(s.y);
                if (resp >= EGO_RESP_MIN &&
                    std::abs(sx) <= EGO_MAX_SHIFT &&
                    std::abs(sy) <= EGO_MAX_SHIFT) {
                    sxs.push_back(sx);
                    sys.push_back(sy);
                }
            }
            if (sxs.size() >= 2) {
                // Median (sxs.size() in {2, 3, 4}).
                std::nth_element(sxs.begin(),
                                  sxs.begin() + sxs.size() / 2, sxs.end());
                std::nth_element(sys.begin(),
                                  sys.begin() + sys.size() / 2, sys.end());
                cam_shift = cv::Point2f(sxs[sxs.size() / 2],
                                         sys[sys.size() / 2]);
            }
        }
        // Pre-correct state position by camera shift (drone's IMAGE
        // position follows world content). Velocity stays untouched
        // -> represents drone's WORLD-frame motion only.
        kf_.statePost.at<float>(0) += cam_shift.x;
        kf_.statePost.at<float>(1) += cam_shift.y;

        // 1. Kalman predict.
        cv::Mat pred = kf_.predict();
        const float px = pred.at<float>(0);
        const float py = pred.at<float>(1);

        // 2. Search window.
        const int sw = 2 * SEARCH_RADIUS;
        const int wx = std::clamp(static_cast<int>(px) - SEARCH_RADIUS, 0,
                                  gray_frame.cols - sw);
        const int wy = std::clamp(static_cast<int>(py) - SEARCH_RADIUS, 0,
                                  gray_frame.rows - sw);
        cv::Mat window = gray_frame(cv::Rect(wx, wy, sw, sw));

        // 4. Multi-scale DoH + scale argmax (Lindeberg).
        cv::Mat blurred;
        window.convertTo(blurred, CV_32F);
        cv::Mat doh_max(window.size(), CV_32F, cv::Scalar(0));
        cv::Mat scale_idx(window.size(), CV_8U, cv::Scalar(0));
        for (int s = 0; s < N_SCALES; ++s) {
            cv::Mat dxx, dyy, dxy, doh_s;
            cv::sepFilter2D(blurred, dxx, CV_32F, gxx_x_[s], gxx_y_[s]);
            cv::sepFilter2D(blurred, dyy, CV_32F, gyy_x_[s], gyy_y_[s]);
            cv::sepFilter2D(blurred, dxy, CV_32F, gxy_x_[s], gxy_y_[s]);
            doh_s = dxx.mul(dyy) - dxy.mul(dxy);
            cv::max(doh_s, 0.0, doh_s);
            doh_s *= sigma4_norm_[s];
            cv::Mat won = doh_s > doh_max;
            doh_s.copyTo(doh_max, won);
            scale_idx.setTo(s, won);
        }
        // doh_anchor_ = drone's own DoH peak in init box, calibrated
        // ONCE per video. mask = DoH > DOH_FRAC * anchor.
        if (doh_anchor_ < 0.f) {
            const int ix =
                std::clamp(static_cast<int>(init_box_[0]) - wx, 0, sw - 1);
            const int iy =
                std::clamp(static_cast<int>(init_box_[1]) - wy, 0, sw - 1);
            const int iw_ =
                std::clamp(static_cast<int>(init_box_[2]), 1, sw - ix);
            const int ih_ =
                std::clamp(static_cast<int>(init_box_[3]), 1, sw - iy);
            cv::Mat init_doh = doh_max(cv::Rect(ix, iy, iw_, ih_));
            double pk;
            cv::minMaxLoc(init_doh, nullptr, &pk);
            doh_anchor_ = std::max(static_cast<float>(pk), 1.0f);
        }
        cv::Mat doh_mask_f, doh_mask;
        cv::threshold(doh_max, doh_mask_f, DOH_FRAC * doh_anchor_, 255,
                      cv::THRESH_BINARY);
        doh_mask_f.convertTo(doh_mask, CV_8U);
        cv::Mat kern =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(doh_mask, doh_mask, cv::MORPH_CLOSE, kern);

        // 5. Use DoH for LOCATING (precise, no positional bias).
        //    Motion only as a HARD GATE: a candidate blob must have
        //    SOME motion pixels overlapping its bbox to be accepted.
        //    This rejects static distractors (sun, hot ground) without
        //    dragging the position estimate toward the motion blob's
        //    centroid (which is biased toward midpoint between drone
        //    old + new position).
        cv::Mat S = kf_.errorCovPre(cv::Rect(0, 0, 2, 2)) +
                    kf_.measurementNoiseCov(cv::Rect(0, 0, 2, 2));
        cv::Mat S_inv = S.inv();

        // MOSSE response map: 1 FFT around predicted center, sample
        // per candidate (free lookup). Discriminates drone from
        // distractors by learned correlation peak.
        if (dcf_.initialized()) {
            dcf_.compute_response_map(gray_frame, cv::Point2f(px, py));
        }

        cv::Mat labels, stats, centroids;
        const int n = cv::connectedComponentsWithStats(doh_mask, labels, stats,
                                                       centroids, 8, CV_32S);
        float best_score = -1.f;
        float best_cx = 0.f, best_cy = 0.f;
        const float sigma_a = std::max(ref_area_ * SIZE_SIGMA_FRAC, 1.f);

        for (int i = 1; i < n; ++i) {
            const int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area < MIN_BLOB_AREA || area > MAX_BLOB_AREA)
                continue;
            const float blob_cx =
                wx + static_cast<float>(centroids.at<double>(i, 0));
            const float blob_cy =
                wy + static_cast<float>(centroids.at<double>(i, 1));

            cv::Mat innov =
                (cv::Mat_<float>(2, 1) << blob_cx - px, blob_cy - py);
            const float maha2 = static_cast<float>(
                cv::Mat(innov.t() * S_inv * innov).at<float>(0));
            if (maha2 > GATING_CHI2)
                continue;
            const float gating = std::exp(-maha2 / 2.f);

            const float da = static_cast<float>(area) - ref_area_;
            const float size_match =
                std::exp(-(da * da) / (2.f * sigma_a * sigma_a));

            // HOG shape score (hand-crafted, ~10 us per candidate).
            // Compute candidate's HOG, dot product with frame-0 drone
            // HOG template (both L2-normalised -> cosine sim in [-1,1]).
            // Rescale to [0, 1] for score multiplication.
            float hog_score = 1.f;
            if (has_hog_tmpl_) {
                const int blob_x_loc = stats.at<int>(i, cv::CC_STAT_LEFT);
                const int blob_y_loc = stats.at<int>(i, cv::CC_STAT_TOP);
                const int blob_ww = stats.at<int>(i, cv::CC_STAT_WIDTH);
                const int blob_hh = stats.at<int>(i, cv::CC_STAT_HEIGHT);
                // Pad 50% context around the blob.
                const int pad_w = blob_ww / 2;
                const int pad_h = blob_hh / 2;
                const int gx = wx + blob_x_loc - pad_w;
                const int gy = wy + blob_y_loc - pad_h;
                const int gw = blob_ww + 2 * pad_w;
                const int gh = blob_hh + 2 * pad_h;
                const int cx0 = std::clamp(gx, 0, gray_frame.cols - 2);
                const int cy0 = std::clamp(gy, 0, gray_frame.rows - 2);
                const int cw = std::min(gw, gray_frame.cols - cx0);
                const int ch = std::min(gh, gray_frame.rows - cy0);
                if (cw > 0 && ch > 0) {
                    cv::Mat patch = gray_frame(cv::Rect(cx0, cy0, cw, ch));
                    const auto h = compute_hog(patch);
                    float cos = 0.f;
                    for (int j = 0; j < HOG_DIM; ++j)
                        cos += h[j] * hog_tmpl_[j];
                    hog_score = std::max(0.f, (cos + 1.f) * 0.5f);
                }
            }

            // MOSSE DCF score (sampled from cached response map).
            float dcf_score = 1.f;
            if (dcf_.initialized()) {
                dcf_score = dcf_.sample_at(cv::Point2f(blob_cx, blob_cy));
                // Floor at 0.05 to avoid killing all candidates when
                // the response map is uniformly weak (e.g. drone
                // appearance momentarily occluded).
                dcf_score = std::max(dcf_score, 0.05f);
            }

            const float score = size_match * gating * hog_score * dcf_score;
            if (score > best_score) {
                best_score = score;
                best_cx = blob_cx;
                best_cy = blob_cy;
            }
        }

        // 8. SHAPE detection from DoH (NO brightness anywhere). Two
        //    thresholds on the same DoH map:
        //
        //      HIGH (DOH_FRAC * anchor): used for detection
        //         (already done above) - tight, picks drone core.
        //      LOW  (0.05 * anchor):     used for SIZING / centroid
        //         - looser, captures the entire drone shape outline
        //         (DoH stays positive throughout drone interior +
        //         decays to zero at edges, so a low threshold wraps
        //         the full silhouette).
        //
        //    The connected component containing the DoH peak in the
        //    LOW-threshold mask = drone full shape. Its centroid =
        //    drone GEOMETRIC center. Its bbox = drone bbox.
        // SIZING uses LOWER DoH threshold (0.05 * anchor): captures
        // entire drone shape (DoH stays positive in drone interior,
        // decays to 0 at edges). CC containing DoH peak = full drone
        // silhouette; centroid = drone GEOMETRIC center.
        if (best_score >= 0.f) {
            cv::Mat doh_lo_f, doh_lo;
            cv::threshold(doh_max, doh_lo_f, 0.05f * doh_anchor_, 255,
                          cv::THRESH_BINARY);
            doh_lo_f.convertTo(doh_lo, CV_8U);
            cv::morphologyEx(doh_lo, doh_lo, cv::MORPH_CLOSE, kern);
            cv::Mat l_labels, l_stats, l_centroids;
            const int ln = cv::connectedComponentsWithStats(
                doh_lo, l_labels, l_stats, l_centroids, 8, CV_32S);
            const int cy_l =
                std::clamp(static_cast<int>(best_cy) - wy, 0, sw - 1);
            const int cx_l =
                std::clamp(static_cast<int>(best_cx) - wx, 0, sw - 1);
            const int target_label = l_labels.at<int>(cy_l, cx_l);
            float meas_cx = best_cx, meas_cy = best_cy;
            bool accepted = false;
            if (target_label > 0 && target_label < ln) {
                const float blob_w = static_cast<float>(
                    l_stats.at<int>(target_label, cv::CC_STAT_WIDTH));
                const float blob_h = static_cast<float>(
                    l_stats.at<int>(target_label, cv::CC_STAT_HEIGHT));
                // Lindeberg automatic scale selection (kept for the
                // corner-branch size estimate when CC can't be trusted).
                const int s_best = scale_idx.at<uchar>(cy_l, cx_l);
                const float sigma_best = sigmas_[s_best];
                const float lin_diam =
                    2.f * std::sqrt(2.f) * sigma_best;
                // full_drone: CC bbox close to current init_w/init_h
                // (drone still). EMA-tracked via prior frames; smooth
                // and stable on each video. corner branch uses
                // Lindeberg diameter only as a fallback when init has
                // diverged or never been set (start of video).
                const float w_ratio = blob_w / std::max(init_w_, 1.f);
                const float h_ratio = blob_h / std::max(init_h_, 1.f);
                const bool full_drone =
                    (w_ratio >= FULL_DRONE_FRAC_MIN &&
                     w_ratio <= FULL_DRONE_FRAC_MAX &&
                     h_ratio >= FULL_DRONE_FRAC_MIN &&
                     h_ratio <= FULL_DRONE_FRAC_MAX);
                if (w_ratio > 0.3f && w_ratio < 4.f &&
                    h_ratio > 0.3f && h_ratio < 4.f) {

                    // SHAPE TEMPLATE CHECK: compute Hu Moments of
                    // this CC's binary mask and compare against the
                    // frozen frame-0 template. If shape distance is
                    // too large, this is a "wrong shape pattern"
                    // (e.g. only half the drone, motion blur splice,
                    // adjacent distractor) - reject the measurement,
                    // fall through to predict-only.
                    cv::Mat tgt_mask = (l_labels == target_label);
                    const auto cur_hu = compute_hu_log(tgt_mask);
                    bool shape_ok = true;
                    if (has_tmpl_) {
                        double d = 0.0;
                        for (int j = 0; j < 7; ++j) {
                            const double diff = cur_hu[j] - tmpl_hu_[j];
                            d += diff * diff;
                        }
                        const float shape_score = std::exp(static_cast<float>(
                            -d / (2.0 * HU_SHAPE_SIGMA * HU_SHAPE_SIGMA)));
                        // Reject only if shape is wildly different.
                        if (shape_score < SHAPE_REJECT_BELOW)
                            shape_ok = false;
                    }

                    if (shape_ok) {
                        // Build/freeze template over first
                        // TMPL_BUILD_FRAMES detections. Average
                        // (NOT EMA) so template = stable mean of
                        // early-frame drone Hu Moments.
                        if (tmpl_build_n_ < TMPL_BUILD_FRAMES) {
                            if (!has_tmpl_) {
                                tmpl_hu_ = cur_hu;
                                has_tmpl_ = true;
                            } else {
                                const double w = 1.0 / (tmpl_build_n_ + 1);
                                for (int j = 0; j < 7; ++j) {
                                    tmpl_hu_[j] =
                                        (1.0 - w) * tmpl_hu_[j] + w * cur_hu[j];
                                }
                            }
                            ++tmpl_build_n_;
                        }

                        if (full_drone) {
                            // CC = full drone: EMA size on CC bbox.
                            init_w_ = (1.f - SIZE_EMA_ALPHA) * init_w_ +
                                      SIZE_EMA_ALPHA * blob_w;
                            init_h_ = (1.f - SIZE_EMA_ALPHA) * init_h_ +
                                      SIZE_EMA_ALPHA * blob_h;
                            ref_area_ = 0.9f * ref_area_ +
                                        0.1f * (blob_w * blob_h);
                        }
                        // Note: corner branch holds init_w/h; if CC
                        // ratio diverges far from current init for many
                        // consecutive frames, the user can re-trigger
                        // Lindeberg fallback by checking lin_diam.
                        (void)lin_diam;
                        if (full_drone) {
                            meas_cx = wx + static_cast<float>(
                                              l_centroids.at<double>(
                                                  target_label, 0));
                            meas_cy = wy + static_cast<float>(
                                              l_centroids.at<double>(
                                                  target_label, 1));
                        } else {
                            meas_cx = best_cx;
                            meas_cy = best_cy;
                        }
                        accepted = true;
                    }
                }
            }
            // PSR (Peak-to-Sidelobe Ratio) drift gate. Only gate the
            // ONLINE FILTER UPDATE; always trust measurement (DoH +
            // Kalman gating already ensure plausibility). High PSR =
            // sharp single peak = drone clearly there = safe to learn
            // current appearance into A_t/B_t. Low PSR = ambiguous /
            // multi-modal response = could be distractor; freeze
            // filter (no update) so it doesn't drift.
            const float psr =
                dcf_.initialized() ? dcf_.last_psr() : 0.f;

            if (accepted) {
                cv::Mat measurement =
                    (cv::Mat_<float>(2, 1) << meas_cx, meas_cy);
                kf_.correct(measurement);
                if (dcf_.initialized() && psr >= PSR_UPDATE) {
                    dcf_.update(gray_frame,
                                cv::Point2f(meas_cx, meas_cy));
                }
                // Confident frame -> reset lost counter.
                if (psr >= PSR_UPDATE) lost_frames_ = 0;
                else                    ++lost_frames_;
            } else {
                kf_.statePost = kf_.statePre.clone();
                kf_.errorCovPost = kf_.errorCovPre.clone();
                ++lost_frames_;
            }
        } else {
            kf_.statePost = kf_.statePre.clone();
            kf_.errorCovPost = kf_.errorCovPre.clone();
            ++lost_frames_;
        }

        // Drift recovery: too many low-confidence frames in a row =>
        // running filter A_t/B_t likely contaminated. Pull halfway
        // back to frozen frame-0 anchor; reset counter so we don't
        // pull every frame.
        if (dcf_.initialized() && lost_frames_ >= LOST_TRIGGER) {
            dcf_.pullback(0.5f);
            lost_frames_ = 0;
        }

        // 9. Push current frame for future motion diff.
        frame_history_.push_back(gray_frame.clone());
        while (static_cast<int>(frame_history_.size()) > STACK_GAP) {
            frame_history_.pop_front();
        }

        // Output: Kalman-smoothed (cx, cy) + EMA-tracked (w, h).
        const float ex = kf_.statePost.at<float>(0);
        const float ey = kf_.statePost.at<float>(1);
        const float ew = std::max(init_w_, 4.f);
        const float eh = std::max(init_h_, 4.f);
        return {ex - ew / 2.f, ey - eh / 2.f, ew, eh};
    }

  private:
    // Build separable 1D row+col kernels for Gxx, Gyy, Gxy at given
    // sigma. Outer-product equals the original 2D kernel: Gxx(x,y) =
    // [(x^2 - s^2)/s^5/sqrt(2pi) * exp(-x^2/2s^2)]
    //  * [1/(s sqrt(2pi)) * exp(-y^2/2s^2)]
    // (and analogously for Gyy, Gxy).  sepFilter2D is ~ksize/2x faster
    // than the equivalent filter2D.
    static void make_sep_kernels_(double sigma, int ksize,
                                   cv::Mat &gxx_x, cv::Mat &gxx_y,
                                   cv::Mat &gyy_x, cv::Mat &gyy_y,
                                   cv::Mat &gxy_x, cv::Mat &gxy_y) {
        const int c = ksize / 2;
        const double s2 = sigma * sigma;
        const double s3 = s2 * sigma;
        const double s5 = s2 * s3;
        const double inv_sqrt_2pi = 1.0 / std::sqrt(2.0 * CV_PI);
        gxx_x.create(1, ksize, CV_32F);
        gxx_y.create(ksize, 1, CV_32F);
        gyy_x.create(1, ksize, CV_32F);
        gyy_y.create(ksize, 1, CV_32F);
        gxy_x.create(1, ksize, CV_32F);
        gxy_y.create(ksize, 1, CV_32F);
        for (int t = 0; t < ksize; ++t) {
            const double tt = t - c;
            const double e = std::exp(-tt * tt / (2.0 * s2));
            const float g0 =
                static_cast<float>(inv_sqrt_2pi / sigma * e);          // Gaussian
            const float g1 =
                static_cast<float>(inv_sqrt_2pi * tt / s3 * e);        // 1st deriv
            const float g2 = static_cast<float>(
                inv_sqrt_2pi * (tt * tt - s2) / s5 * e);                // 2nd deriv
            gxx_x.at<float>(0, t) = g2; gxx_y.at<float>(t, 0) = g0;
            gyy_x.at<float>(0, t) = g0; gyy_y.at<float>(t, 0) = g2;
            gxy_x.at<float>(0, t) = g1; gxy_y.at<float>(t, 0) = g1;
        }
    }
    // Hu Moments on a binary mask, log-magnitude (sign-preserving).
    // Pure shape descriptor: rotation/scale/translation invariant.
    static std::vector<double> compute_hu_log(const cv::Mat &binary_mask) {
        cv::Moments m = cv::moments(binary_mask, true);
        double hu[7];
        cv::HuMoments(m, hu);
        std::vector<double> r(7);
        for (int i = 0; i < 7; ++i) {
            const double s = (hu[i] >= 0 ? 1.0 : -1.0);
            r[i] = -s * std::log10(std::abs(hu[i]) + 1e-12);
        }
        return r;
    }

    // HOG (Histogram of Oriented Gradients) on a grayscale patch.
    // 16x16 patch -> 4x4 cells of 4x4 px -> 9 unsigned bins per cell
    // -> 144-d feature, L2-normalised. Brightness-shift invariant
    // because uses gradients; block normalisation handles scale.
    static constexpr int HOG_PATCH = 16;
    static constexpr int HOG_CELLS = 4;
    static constexpr int HOG_BINS = 9;
    static constexpr int HOG_DIM = HOG_CELLS * HOG_CELLS * HOG_BINS;
    static std::vector<float> compute_hog(const cv::Mat &gray_patch) {
        cv::Mat p;
        if (gray_patch.cols != HOG_PATCH || gray_patch.rows != HOG_PATCH) {
            cv::resize(gray_patch, p, cv::Size(HOG_PATCH, HOG_PATCH));
        } else {
            p = gray_patch;
        }
        cv::Mat gx, gy, mag, ang;
        cv::Sobel(p, gx, CV_32F, 1, 0, 3);
        cv::Sobel(p, gy, CV_32F, 0, 1, 3);
        cv::cartToPolar(gx, gy, mag, ang, true);
        std::vector<float> h(HOG_DIM, 0.f);
        const int cs = HOG_PATCH / HOG_CELLS;
        const float bin_w = 180.f / HOG_BINS;
        for (int cy = 0; cy < HOG_CELLS; ++cy) {
            for (int cx = 0; cx < HOG_CELLS; ++cx) {
                for (int dy = 0; dy < cs; ++dy) {
                    for (int dx = 0; dx < cs; ++dx) {
                        int y = cy * cs + dy, x = cx * cs + dx;
                        float a = ang.at<float>(y, x);
                        if (a >= 180.f)
                            a -= 180.f;
                        const int b =
                            std::min(HOG_BINS - 1, static_cast<int>(a / bin_w));
                        h[(cy * HOG_CELLS + cx) * HOG_BINS + b] +=
                            mag.at<float>(y, x);
                    }
                }
            }
        }
        float n = 0.f;
        for (float v : h)
            n += v * v;
        n = std::sqrt(n) + 1e-6f;
        for (float &v : h)
            v /= n;
        return h;
    }

    cv::KalmanFilter kf_;
    cv::Ptr<cv::CLAHE> clahe_;
    cv::Mat gxx_x_[N_SCALES], gxx_y_[N_SCALES];
    cv::Mat gyy_x_[N_SCALES], gyy_y_[N_SCALES];
    cv::Mat gxy_x_[N_SCALES], gxy_y_[N_SCALES];
    float   sigmas_[N_SCALES];
    float sigma4_norm_[N_SCALES];
    std::deque<cv::Mat> frame_history_;
    Box init_box_ = {0, 0, 0, 0};
    float doh_anchor_ = -1.f;
    float init_w_ = 0.f;
    float init_h_ = 0.f;
    float ref_area_ = 0.f;
    // Frozen shape template: Hu Moments of the drone's DoH-low mask
    // built from the first few stable frames. Compared against each
    // candidate CC's Hu Moments to detect "wrong shape pattern" (e.g.
    // half the drone) during fast camera motion.
    std::vector<double> tmpl_hu_;
    bool has_tmpl_ = false;
    int tmpl_build_n_ = 0;
    // Hand-crafted HOG template (frame 0, 144-d L2-normalised).
    std::vector<float> hog_tmpl_;
    bool has_hog_tmpl_ = false;
    // MOSSE correlation filter (frame 0 init, online updated).
    MosseDcf dcf_;
    // Drift recovery: count consecutive low-confidence frames; when
    // >= LOST_TRIGGER, pull running filter back toward frame-0 anchor.
    int lost_frames_ = 0;
    static constexpr int TMPL_BUILD_FRAMES = 10;
    static constexpr float HU_SHAPE_SIGMA = 5.0f;
    static constexpr float SHAPE_REJECT_BELOW = 0.001f; // very lenient
    // PSR thresholds (Bolme 2010 nominal: 7-20). PSR_UPDATE = high
    // confidence -> safe to update filter. PSR_KEEP = below this
    // reject measurement (likely distractor lock).
    static constexpr float PSR_UPDATE  = 10.0f;
    static constexpr float PSR_KEEP    = 3.0f;
    // After this many consecutive low-confidence frames, pull MOSSE
    // running A_t/B_t halfway toward frozen frame-0 anchor.
    static constexpr int   LOST_TRIGGER = 10;
    // Full-drone CC vs current EMA-tracked init_w/init_h. Wider than
    // Lindeberg-based bound because EMA captures the actual recent
    // drone scale (more reliable than Lindeberg's per-frame sigma_best,
    // which can pick a boundary sigma on small drones with sigma=16).
    static constexpr float FULL_DRONE_FRAC_MIN = 0.65f;
    static constexpr float FULL_DRONE_FRAC_MAX = 1.55f;
};
