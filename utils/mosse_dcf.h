#pragma once

#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

// MOSSE (Minimum Output Sum of Squared Error) correlation filter.
// Bolme et al. 2010. Pure FFT, no NN. Trains a filter that maps
// drone patch -> Gaussian peak; new-frame correlation peak = drone
// location.
//
// Filter update online via running sum:
//   A_t = (1 - eta) A_{t-1} + eta * G * conj(F_t)     (numerator)
//   B_t = (1 - eta) B_{t-1} + eta * F_t * conj(F_t)   (denominator)
//   H = A / B (complex division, regularized by lambda)
//   response(z) = IFFT(FFT(z) * conj(H))
//
// All ops on PATCH_SIZE x PATCH_SIZE centered window.
class MosseDcf {
  public:
    static constexpr int   PATCH_SIZE = 64;   // FFT window (must be power of 2)
    static constexpr float PEAK_SIGMA = 3.0f; // Gaussian label std-dev
    static constexpr float ETA        = 0.125f; // online learning rate
    static constexpr float LAMBDA     = 0.01f;  // regularization

    bool initialized() const { return initialized_; }

    void init(const cv::Mat &gray, cv::Point2f center) {
        build_label_();
        build_window_();
        const cv::Mat F = patch_fft_(gray, center);
        cv::mulSpectrums(G_, F, A_, 0, true);   // G * conj(F)
        cv::mulSpectrums(F, F, B_, 0, true);    // F * conj(F)
        // Freeze frame-0 anchor (never updated) for drift recovery.
        A_0_ = A_.clone();
        B_0_ = B_.clone();
        initialized_ = true;
    }

    // Blend running filter toward frozen frame-0 anchor. alpha in [0,1]
    // (0 = no pullback, 1 = full reset to anchor). Use when drift is
    // detected (e.g. consecutive low-PSR or rejected frames).
    void pullback(float alpha) {
        if (A_0_.empty()) return;
        cv::addWeighted(A_, 1.0 - alpha, A_0_, alpha, 0.0, A_);
        cv::addWeighted(B_, 1.0 - alpha, B_0_, alpha, 0.0, B_);
    }

    // Compute correlation response over the window; return peak value
    // and offset (peak_xy - center). Higher peak = better match.
    std::pair<float, cv::Point2f>
    response(const cv::Mat &gray, cv::Point2f center) const {
        const cv::Mat F = patch_fft_(gray, center);
        cv::Mat AF; cv::mulSpectrums(F, A_, AF, 0, false);   // F * A
        cv::Mat resp_freq; complex_div_(AF, B_, resp_freq);
        cv::Mat resp;
        cv::idft(resp_freq, resp,
                 cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
        double mn, mx; cv::Point mnl, mxl;
        cv::minMaxLoc(resp, &mn, &mx, &mnl, &mxl);
        // PSR (peak-to-sidelobe ratio) = (peak - mean_off_peak) / std_off_peak.
        // Mask out 11x11 around peak, compute mean/std elsewhere.
        cv::Mat mask = cv::Mat::ones(resp.size(), CV_8U) * 255;
        cv::Rect peak_rect(mxl.x - 5, mxl.y - 5, 11, 11);
        peak_rect &= cv::Rect(0, 0, resp.cols, resp.rows);
        mask(peak_rect) = 0;
        cv::Scalar mean, stddev;
        cv::meanStdDev(resp, mean, stddev, mask);
        const float psr = (mx - mean[0]) / (stddev[0] + 1e-6);
        return {psr, cv::Point2f(mxl.x - PATCH_SIZE / 2.f,
                                  mxl.y - PATCH_SIZE / 2.f)};
    }

    // Compute & cache the response map once (cheap = 1 FFT + 1 IFFT).
    // Also computes PSR (Peak-to-Sidelobe Ratio):
    //   PSR = (peak - mean_offpeak) / std_offpeak
    // High PSR = sharp peak vs flat background = confident detection.
    // Low PSR = mediocre / multi-modal response = uncertain (likely
    // occluded or distractor-confused). Gate online filter updates
    // by PSR to break the drift cascade.
    float compute_response_map(const cv::Mat &gray, cv::Point2f center) {
        last_center_ = center;
        const cv::Mat F = patch_fft_(gray, center);
        cv::Mat AF; cv::mulSpectrums(F, A_, AF, 0, false);
        cv::Mat resp_freq; complex_div_(AF, B_, resp_freq);
        cv::idft(resp_freq, last_resp_,
                 cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
        double mn, mx; cv::Point mnl, mxl;
        cv::minMaxLoc(last_resp_, &mn, &mx, &mnl, &mxl);
        last_peak_ = static_cast<float>(mx);
        // PSR over off-peak region (mask out 11x11 around peak).
        cv::Mat mask(last_resp_.size(), CV_8U, cv::Scalar(255));
        cv::Rect peak_rect(mxl.x - 5, mxl.y - 5, 11, 11);
        peak_rect &= cv::Rect(0, 0, last_resp_.cols, last_resp_.rows);
        mask(peak_rect) = 0;
        cv::Scalar mu, sigma;
        cv::meanStdDev(last_resp_, mu, sigma, mask);
        last_psr_ = static_cast<float>(
            (mx - mu[0]) / (sigma[0] + 1e-6));
        return last_peak_;
    }

    float last_psr() const { return last_psr_; }

    // Sample cached response map at absolute pixel pos. Returns
    // [0, 1] (raw / peak). Outside window -> 0.
    float sample_at(cv::Point2f abs_pos) const {
        if (last_resp_.empty() || last_peak_ <= 0.f) return 0.f;
        const int dx = static_cast<int>(abs_pos.x - last_center_.x +
                                         PATCH_SIZE / 2.f);
        const int dy = static_cast<int>(abs_pos.y - last_center_.y +
                                         PATCH_SIZE / 2.f);
        if (dx < 0 || dx >= PATCH_SIZE || dy < 0 || dy >= PATCH_SIZE)
            return 0.f;
        const float v = std::max(0.f, last_resp_.at<float>(dy, dx));
        return std::min(1.f, v / last_peak_);
    }

    // Online filter update with new accepted measurement.
    void update(const cv::Mat &gray, cv::Point2f center) {
        if (!initialized_) { init(gray, center); return; }
        const cv::Mat F = patch_fft_(gray, center);
        cv::Mat A_new, B_new;
        cv::mulSpectrums(G_, F, A_new, 0, true);
        cv::mulSpectrums(F, F, B_new, 0, true);
        cv::addWeighted(A_, 1.0 - ETA, A_new, ETA, 0.0, A_);
        cv::addWeighted(B_, 1.0 - ETA, B_new, ETA, 0.0, B_);
    }

  private:
    void build_label_() {
        cv::Mat g(PATCH_SIZE, PATCH_SIZE, CV_32F);
        const float c = PATCH_SIZE / 2.f;
        for (int y = 0; y < PATCH_SIZE; ++y)
            for (int x = 0; x < PATCH_SIZE; ++x) {
                const float dx = x - c, dy = y - c;
                g.at<float>(y, x) =
                    std::exp(-(dx * dx + dy * dy) /
                              (2.f * PEAK_SIGMA * PEAK_SIGMA));
            }
        cv::dft(g, G_, cv::DFT_COMPLEX_OUTPUT);
    }

    void build_window_() {
        cv::createHanningWindow(cos_window_,
                                cv::Size(PATCH_SIZE, PATCH_SIZE),
                                CV_32F);
    }

    cv::Mat patch_fft_(const cv::Mat &gray, cv::Point2f center) const {
        cv::Mat patch;
        cv::getRectSubPix(gray, cv::Size(PATCH_SIZE, PATCH_SIZE),
                          center, patch);
        cv::Mat f; patch.convertTo(f, CV_32F);
        // log + per-patch normalize (illumination invariance).
        cv::log(f + 1.f, f);
        cv::Scalar mean, stddev;
        cv::meanStdDev(f, mean, stddev);
        f = (f - mean[0]) / (stddev[0] + 1e-6);
        f = f.mul(cos_window_);
        cv::Mat F;
        cv::dft(f, F, cv::DFT_COMPLEX_OUTPUT);
        return F;
    }

    static void complex_div_(const cv::Mat &A, const cv::Mat &B,
                              cv::Mat &out) {
        std::vector<cv::Mat> a(2), b(2);
        cv::split(A, a);
        cv::split(B, b);
        cv::Mat den = b[0].mul(b[0]) + b[1].mul(b[1]) + LAMBDA;
        cv::Mat re = (a[0].mul(b[0]) + a[1].mul(b[1])) / den;
        cv::Mat im = (a[1].mul(b[0]) - a[0].mul(b[1])) / den;
        std::vector<cv::Mat> o = {re, im};
        cv::merge(o, out);
    }

    cv::Mat G_;         // FFT of Gaussian label (frozen)
    cv::Mat A_, B_;     // running sums (numerator / denominator)
    cv::Mat A_0_, B_0_; // frame-0 frozen anchor (drift recovery)
    cv::Mat cos_window_;
    cv::Mat last_resp_; // cached response map (sample_at)
    cv::Point2f last_center_{0.f, 0.f};
    float   last_peak_   = 0.f;
    float   last_psr_    = 0.f;
    bool    initialized_ = false;
};
