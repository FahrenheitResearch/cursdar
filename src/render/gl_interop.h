#pragma once
#include <cstdint>

// CUDA-OpenGL interop manager
// Manages a GL texture that CUDA writes to and ImGui displays
class GlCudaTexture {
public:
    GlCudaTexture() = default;
    ~GlCudaTexture();

    // Create GL texture and register with CUDA
    bool init(int width, int height);

    // Resize (recreate if needed)
    bool resize(int width, int height);

    // Copy device buffer to GL texture via CUDA interop
    void updateFromDevice(const uint32_t* d_buffer, int width, int height);

    // Get GL texture ID for ImGui::Image
    unsigned int textureId() const { return m_texId; }
    int width() const { return m_width; }
    int height() const { return m_height; }

    // Release resources
    void destroy();

private:
    unsigned int m_texId = 0;
    int  m_width = 0;
    int  m_height = 0;
    void* m_cudaResource = nullptr; // cudaGraphicsResource_t
    bool m_registered = false;
};
