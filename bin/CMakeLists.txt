add_executable(IQM IQM/main.cpp shared/args.cpp shared/methods.cpp)
add_executable(IQM::IQM ALIAS IQM)

target_include_directories(IQM PUBLIC ../include)
target_include_directories(IQM PUBLIC ../lib/stb)

add_executable(IQM-profile IQM-profile/profile.cpp shared/args.cpp shared/methods.cpp)
add_executable(IQM::profile ALIAS IQM-profile)

target_include_directories(IQM-profile PUBLIC ../include)
target_include_directories(IQM-profile PUBLIC ../lib/stb)

if (SSIM)
    target_link_libraries(IQM IQM::SSIM)
    target_link_libraries(IQM-profile IQM::SSIM)
endif ()

if (SVD)
    target_link_libraries(IQM IQM::SVD)
    target_link_libraries(IQM-profile IQM::SVD)
endif ()

if (FSIM)
    target_link_libraries(IQM IQM::FSIM)
    target_link_libraries(IQM-profile IQM::FSIM)
endif ()

if (FLIP)
    target_link_libraries(IQM IQM::FLIP)
    target_link_libraries(IQM-profile IQM::FLIP)
endif ()