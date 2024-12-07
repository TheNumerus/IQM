cmake_minimum_required(VERSION 3.29)
project(IQM-FSIM)

add_library(IQM-FSIM STATIC fsim.cpp
        steps/fsim_lowpass_filter.cpp
        steps/fsim_lowpass_filter.h
        steps/fsim_log_gabor.cpp
        steps/fsim_log_gabor.h
        steps/fsim_angular_filter.cpp
        steps/fsim_angular_filter.h
        steps/fsim_filter_combinations.cpp
        steps/fsim_filter_combinations.h
        steps/fsim_final_multiply.cpp
        steps/fsim_final_multiply.h
        steps/fsim_sum_filter_responses.cpp
        steps/fsim_sum_filter_responses.h
        steps/fsim_noise_power.cpp
        steps/fsim_noise_power.h
        steps/fsim_phase_congruency.cpp
        steps/fsim_phase_congruency.h
        steps/fsim_estimate_energy.cpp
        steps/fsim_estimate_energy.h
)

target_include_directories(IQM-FSIM
        PUBLIC "${PROJECT_SOURCE_DIR}"
        PUBLIC ${LIBRARY_ROOT}/VkFFT
)