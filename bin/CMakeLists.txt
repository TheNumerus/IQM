add_executable(IQM IQM/main.cpp shared/args.cpp shared/methods.cpp)
add_executable(IQM::IQM ALIAS IQM)

target_include_directories(IQM PUBLIC ../include)
target_include_directories(IQM PUBLIC ../lib/stb)

add_executable(IQM-profile IQM-profile/profile.cpp shared/args.cpp shared/methods.cpp)
add_executable(IQM::profile ALIAS IQM-profile)

target_include_directories(IQM-profile PUBLIC ../include)
target_include_directories(IQM-profile PUBLIC ../lib/stb)

target_compile_definitions(IQM-profile PUBLIC -DPROFILE)
find_package(glfw3)
target_link_libraries(IQM-profile glfw)

if (SSIM)
    target_compile_definitions(IQM PRIVATE -DCOMPILE_SSIM)
    target_compile_definitions(IQM-profile PRIVATE -DCOMPILE_SSIM)
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