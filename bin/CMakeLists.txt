add_executable(IQM IQM/main.cpp IQM/args.cpp IQM/file_matcher.cpp IQM/vulkan_instance.cpp shared/methods.cpp shared/vulkan_res.cpp)
add_executable(IQM::IQM ALIAS IQM)

target_include_directories(IQM PUBLIC ../include)
target_include_directories(IQM PUBLIC ../lib/stb)

add_executable(IQM-profile IQM-profile/profile.cpp IQM-profile/args.cpp shared/methods.cpp)
add_executable(IQM::profile ALIAS IQM-profile)

target_include_directories(IQM-profile PUBLIC ../include)
target_include_directories(IQM-profile PUBLIC ../lib/stb)

target_compile_definitions(IQM-profile PUBLIC -DPROFILE)
find_package(glfw3)
target_link_libraries(IQM-profile glfw)

if (ENABLE_RENDERDOC)
    target_compile_definitions(IQM PRIVATE -DENABLE_RENDERDOC)
    target_compile_definitions(IQM-profile PRIVATE -DENABLE_RENDERDOC)
endif ()

if (SSIM)
    target_compile_definitions(IQM PRIVATE -DCOMPILE_SSIM)
    target_compile_definitions(IQM-profile PRIVATE -DCOMPILE_SSIM)
    target_sources(IQM PRIVATE IQM/wrappers/ssim.cpp)
    target_link_libraries(IQM IQM::SSIM)
    target_link_libraries(IQM-profile IQM::SSIM)
endif ()

if (SVD)
    target_compile_definitions(IQM PRIVATE -DCOMPILE_SVD)
    target_compile_definitions(IQM-profile PRIVATE -DCOMPILE_SVD)
    target_link_libraries(IQM IQM::SVD)
    target_link_libraries(IQM-profile IQM::SVD)
endif ()

if (FSIM)
    target_compile_definitions(IQM PRIVATE -DCOMPILE_FSIM)
    target_compile_definitions(IQM-profile PRIVATE -DCOMPILE_FSIM)
    target_link_libraries(IQM IQM::FSIM)
    target_link_libraries(IQM-profile IQM::FSIM)
endif ()

if (FLIP)
    target_compile_definitions(IQM PRIVATE -DCOMPILE_FLIP)
    target_compile_definitions(IQM-profile PRIVATE -DCOMPILE_FLIP)
    target_link_libraries(IQM IQM::FLIP)
    target_link_libraries(IQM-profile IQM::FLIP)
endif ()