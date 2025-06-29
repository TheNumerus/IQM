add_executable(IQM IQM/main.cpp IQM/args.cpp IQM/file_matcher.cpp IQM/vulkan_instance.cpp shared/methods.cpp shared/vulkan_res.cpp)
add_executable(IQM::IQM ALIAS IQM)

target_include_directories(IQM PUBLIC ../include)
target_include_directories(IQM SYSTEM PUBLIC ../lib/stb)

add_executable(IQM-profile IQM-profile/profile.cpp IQM-profile/args.cpp IQM-profile/vulkan_instance.cpp shared/methods.cpp shared/vulkan_res.cpp)
add_executable(IQM::profile ALIAS IQM-profile)

target_include_directories(IQM-profile PUBLIC ../include)
target_include_directories(IQM-profile SYSTEM PUBLIC ../lib/stb)

target_compile_options(IQM PRIVATE "-Wall;-Wextra")
target_compile_options(IQM-profile PRIVATE "-Wall;-Wextra")

find_package(glfw3)
target_link_libraries(IQM-profile glfw)

target_link_libraries(IQM
        SPIRV-Tools
        SPIRV-Tools-opt
)

target_link_libraries(IQM-profile
        SPIRV-Tools
        SPIRV-Tools-opt
)

if (ENABLE_RENDERDOC)
    target_compile_definitions(IQM PRIVATE -DENABLE_RENDERDOC)
    target_compile_definitions(IQM-profile PRIVATE -DENABLE_RENDERDOC)
endif ()

if (SSIM)
    target_compile_definitions(IQM PRIVATE -DCOMPILE_SSIM)
    target_compile_definitions(IQM-profile PRIVATE -DCOMPILE_SSIM)
    target_sources(IQM PRIVATE shared/wrappers/ssim.cpp)
    target_sources(IQM-profile PRIVATE shared/wrappers/ssim.cpp)
    target_link_libraries(IQM IQM::SSIM)
    target_link_libraries(IQM-profile IQM::SSIM)
endif ()

if (SVD)
    target_compile_definitions(IQM PRIVATE -DCOMPILE_SVD)
    target_compile_definitions(IQM-profile PRIVATE -DCOMPILE_SVD)
    target_sources(IQM PRIVATE shared/wrappers/svd.cpp)
    target_sources(IQM-profile PRIVATE shared/wrappers/svd.cpp)
    target_link_libraries(IQM IQM::SVD)
    target_link_libraries(IQM-profile IQM::SVD)
endif ()

if (FSIM)
    target_compile_definitions(IQM PRIVATE -DCOMPILE_FSIM)
    target_compile_definitions(IQM-profile PRIVATE -DCOMPILE_FSIM)
    target_sources(IQM PRIVATE shared/wrappers/fsim.cpp)
    target_sources(IQM-profile PRIVATE shared/wrappers/fsim.cpp)
    target_link_libraries(IQM IQM::FSIM)
    target_link_libraries(IQM-profile IQM::FSIM)
endif ()

if (FLIP)
    target_compile_definitions(IQM PRIVATE -DCOMPILE_FLIP)
    target_compile_definitions(IQM-profile PRIVATE -DCOMPILE_FLIP)
    target_sources(IQM PRIVATE shared/wrappers/flip.cpp)
    target_sources(IQM-profile PRIVATE shared/wrappers/flip.cpp)
    target_link_libraries(IQM IQM::FLIP)
    target_link_libraries(IQM-profile IQM::FLIP)
endif ()

if (PSNR)
    target_compile_definitions(IQM PRIVATE -DCOMPILE_PSNR)
    target_compile_definitions(IQM-profile PRIVATE -DCOMPILE_PSNR)
    target_sources(IQM PRIVATE shared/wrappers/psnr.cpp)
    target_sources(IQM-profile PRIVATE shared/wrappers/psnr.cpp)
    target_link_libraries(IQM IQM::PSNR)
    target_link_libraries(IQM-profile IQM::PSNR)
endif ()

if (LPIPS)
    target_compile_definitions(IQM PRIVATE -DCOMPILE_LPIPS)
    target_compile_definitions(IQM-profile PRIVATE -DCOMPILE_LPIPS)
    target_sources(IQM PRIVATE shared/wrappers/lpips.cpp)
    target_sources(IQM-profile PRIVATE shared/wrappers/lpips.cpp)
    target_link_libraries(IQM IQM::LPIPS)
    target_link_libraries(IQM-profile IQM::LPIPS)
endif ()
