// Offline PGO workload: public training JPEGs, half-size planar YUV, nine workers.
#include <dlfcn.h>
#include <atomic>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4 || std::string(argv[3]) != "yuv") return 2;
    void* lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) throw std::runtime_error(dlerror());
    auto init = (void*(*)())dlsym(lib, "tjInitDecompress");
    auto destroy = (int(*)(void*))dlsym(lib, "tjDestroy");
    auto header = (int(*)(void*, const unsigned char*, unsigned long, int*, int*, int*, int*))
        dlsym(lib, "tjDecompressHeader3");
    auto yuv = (int(*)(void*, const unsigned char*, unsigned long, unsigned char*, int, int, int, int))
        dlsym(lib, "tjDecompressToYUV2");
    if (!init || !destroy || !header || !yuv) return 3;
    std::ifstream manifest(argv[2]);
    std::string path;
    std::vector<std::vector<unsigned char>> images;
    while (std::getline(manifest, path)) {
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error(path);
        images.emplace_back(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }
    if (images.empty()) return 4;
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    for (int t = 0; t < 9; ++t) workers.emplace_back([&] {
        void* handle = init();
        if (!handle) { failed = true; return; }
        std::vector<unsigned char> output;
        while (true) {
            auto index = next.fetch_add(1);
            if (index >= images.size()) break;
            auto& data = images[index];
            int w, h, sub, color;
            if (header(handle, data.data(), data.size(), &w, &h, &sub, &color)) {
                failed = true; break;
            }
            w = (w + 1) / 2; h = (h + 1) / 2;
            output.resize((size_t)(w + 8) * (h + 8) * 3);
            if (yuv(handle, data.data(), data.size(), output.data(), w, 1, h, 0)) {
                failed = true; break;
            }
        }
        destroy(handle);
    });
    for (auto& thread : workers) thread.join();
    // Unloading flushes the instrumented library's profile files.
    dlclose(lib);
    if (failed) return 5;
    std::cout << "PGO decoded " << images.size() << " training images\n";
}
