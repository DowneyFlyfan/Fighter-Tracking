#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

struct INIT_engine {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
};

// Load and initialize a TensorRT (TRT) engine from a serialized engine file.
// Returns an INIT_engine struct containing the runtime, engine, and execution
// context, or std::nullopt on failure.
inline std::optional<INIT_engine> init_engine(const std::string &engine_path) {
    static Logger trtLogger;
    std::unique_ptr<nvinfer1::IRuntime> runtime(
        nvinfer1::createInferRuntime(trtLogger));

    // Load the engine file
    std::ifstream engineFile(engine_path, std::ios::binary);
    if (!engineFile.good()) {
        std::cerr << "Error: Could not open engine file: " << engine_path
                  << std::endl;
        return std::nullopt;
    }
    engineFile.seekg(0, std::ios::end);
    size_t fileSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    std::vector<char> engineData(fileSize);
    engineFile.read(engineData.data(), fileSize);

    // Deserialize the engine
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!engine) {
        std::cerr << "Error: Failed to deserialize engine!" << std::endl;
        return std::nullopt;
    }

    // Create execution context
    std::unique_ptr<nvinfer1::IExecutionContext> context(
        engine->createExecutionContext());
    if (!context) {
        std::cerr << "Error: Failed to create execution context!" << std::endl;
        return std::nullopt;
    }

    INIT_engine inited_engine;
    inited_engine.runtime = std::move(runtime);
    inited_engine.engine = std::move(engine);
    inited_engine.context = std::move(context);

    return inited_engine;
}
