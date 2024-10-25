#include "args.h"

IQM::Args::Args(const unsigned argc, const char *argv[]) {
    this->outputPath = std::nullopt;

    bool parsedMethod = false;
    bool parsedInput = false;
    bool parsedReference = false;

    for (unsigned i = 0; i < argc; i++) {
        if ((i + 1) < argc) {
            if (strcmp(argv[i], "--method") == 0) {
                if (strcmp(argv[i + 1], "SSIM_CPU") == 0) {
                    this->method = Method::SSIM_CPU;
                    parsedMethod = true;
                } else if (strcmp(argv[i + 1], "SSIM") == 0) {
                    this->method = Method::SSIM;
                    parsedMethod = true;
                }
            } else if (strcmp(argv[i], "--input") == 0) {
                this->inputPath = std::string(argv[i + 1]);
                parsedInput = true;
            } else if (strcmp(argv[i], "--ref") == 0) {
                this->refPath = std::string(argv[i + 1]);
                parsedReference = true;
            } else if (strcmp(argv[i], "--output") == 0) {
                this->outputPath = std::string(argv[i + 1]);
            }
        }
    }

    if (!parsedMethod) {
        throw std::runtime_error("missing method");
    }
    if (!parsedInput) {
        throw std::runtime_error("missing input");
    }
    if (!parsedReference) {
        throw std::runtime_error("missing reference");
    }
}

