#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <GL/gl.h>

#include "gl_interop.h"
#include "../cuda/cuda_common.cuh"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstdio>
#include <cstring>

// GL constants that may be missing from Windows GL/gl.h (only GL 1.1)
#ifndef GL_RGBA8
#define GL_RGBA8 0x8058
#endif
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

GlCudaTexture::~GlCudaTexture() {
    destroy();
}

bool GlCudaTexture::init(int width, int height) {
    m_width = width;
    m_height = height;

    // Create GL texture
    glGenTextures(1, &m_texId);
    glBindTexture(GL_TEXTURE_2D, m_texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Register with CUDA
    cudaError_t err = cudaGraphicsGLRegisterImage(
        (cudaGraphicsResource_t*)&m_cudaResource,
        m_texId, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard);

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsGLRegisterImage failed: %s\n",
                cudaGetErrorString(err));
        glDeleteTextures(1, &m_texId);
        m_texId = 0;
        return false;
    }

    m_registered = true;
    printf("GL-CUDA texture created: %dx%d (tex=%u)\n", width, height, m_texId);
    return true;
}

bool GlCudaTexture::resize(int width, int height) {
    if (width == m_width && height == m_height && m_texId != 0) return true;
    destroy();
    return init(width, height);
}

void GlCudaTexture::updateFromDevice(const uint32_t* d_buffer, int width, int height) {
    if (!m_registered || !d_buffer) return;

    cudaGraphicsResource_t resource = (cudaGraphicsResource_t)m_cudaResource;

    // Map GL texture for CUDA access
    CUDA_CHECK(cudaGraphicsMapResources(1, &resource, 0));

    // Get CUDA array from the mapped texture
    cudaArray_t array;
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));

    // Copy device buffer to the array
    CUDA_CHECK(cudaMemcpy2DToArray(
        array, 0, 0,
        d_buffer, width * sizeof(uint32_t),
        width * sizeof(uint32_t), height,
        cudaMemcpyDeviceToDevice));

    // Unmap
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &resource, 0));
}

void GlCudaTexture::destroy() {
    if (m_registered) {
        cudaGraphicsUnregisterResource((cudaGraphicsResource_t)m_cudaResource);
        m_registered = false;
        m_cudaResource = nullptr;
    }
    if (m_texId) {
        glDeleteTextures(1, &m_texId);
        m_texId = 0;
    }
    m_width = 0;
    m_height = 0;
}
