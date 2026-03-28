#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "app.h"
#include "ui/ui.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <cstdio>
#include <chrono>

// ── Globals for callbacks ───────────────────────────────────
static App* g_app = nullptr;
static bool g_mouseDown = false;
static bool g_rightMouseDown = false;
static bool g_middleMouseDown = false;
static double g_lastMouseX = 0, g_lastMouseY = 0;

static void scrollCallback(GLFWwindow* window, double xoff, double yoff) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (g_app) g_app->onScroll(xoff, yoff);
}

static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    if (g_app && width > 0 && height > 0) {
        g_app->onResize(width, height);
    }
}

int main(int argc, char** argv) {
    printf("=== CURSDAR - CUDA Radar Engine ===\n");
    printf("Loading all NEXRAD stations with custom CUDA kernels\n\n");

    // ── GLFW init ───────────────────────────────────────────
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    int winW = mode->width;
    int winH = mode->height;

    GLFWwindow* window = glfwCreateWindow(winW, winH, "CURSDAR - CUDA Radar", nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // No vsync - maximum frame rate
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    // Get actual framebuffer size
    glfwGetFramebufferSize(window, &winW, &winH);

    // ── Dear ImGui init ─────────────────────────────────────
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");

    ui::init();

    // ── App init ────────────────────────────────────────────
    App app;
    g_app = &app;

    if (!app.init(winW, winH)) {
        fprintf(stderr, "Failed to initialize app\n");
        return 1;
    }

    printf("Starting main loop...\n");

    // ── Main loop ───────────────────────────────────────────
    auto lastFrame = std::chrono::steady_clock::now();
    int frameCount = 0;
    float fpsTimer = 0;
    float fps = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Calculate delta time
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - lastFrame).count();
        lastFrame = now;

        // FPS counter
        frameCount++;
        fpsTimer += dt;
        if (fpsTimer >= 1.0f) {
            fps = frameCount / fpsTimer;
            frameCount = 0;
            fpsTimer = 0;
            char title[128];
            snprintf(title, sizeof(title),
                     "CURSDAR - %s | %s | Tilt %.1f | %d stations | %.0f FPS",
                     app.activeStationName(),
                     PRODUCT_INFO[app.activeProduct()].name,
                     app.activeTiltAngle(),
                     app.stationsLoaded(), fps);
            glfwSetWindowTitle(window, title);
        }

        // Mouse tracking
        double mx, my;
        glfwGetCursorPos(window, &mx, &my);

        // Station tracking (lock station in cross-section/3D mode)
        if (!ImGui::GetIO().WantCaptureMouse && !app.crossSection() && !app.mode3D()) {
            app.onMouseMove(mx, my);
        }

        // Left drag to pan, right drag to orbit (3D mode)
        if (!ImGui::GetIO().WantCaptureMouse) {
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                if (g_mouseDown)
                    app.onMouseDrag(mx - g_lastMouseX, my - g_lastMouseY);
                g_mouseDown = true;
            } else {
                g_mouseDown = false;
            }
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
                if (g_rightMouseDown)
                    app.onRightDrag(mx - g_lastMouseX, my - g_lastMouseY);
                else if (app.crossSection())
                    app.onMiddleClick(mx, my); // start cross-section line
                g_rightMouseDown = true;
            } else {
                g_rightMouseDown = false;
            }
            // Right-drag also moves cross-section endpoint
            if (g_rightMouseDown && app.crossSection()) {
                app.onMiddleDrag(mx, my);
            }
        }
        g_lastMouseX = mx;
        g_lastMouseY = my;

        // Update app state
        app.update(dt);

        // Render radar to GPU texture
        app.render();

        // ── ImGui frame ─────────────────────────────────────
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ui::render(app);

        // FPS overlay
        ImGui::SetNextWindowPos(ImVec2((float)app.viewport().width - 100, (float)app.viewport().height - 30));
        ImGui::Begin("##fps", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoBackground);
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 0.8f), "%.0f FPS", fps);
        ImGui::End();

        ImGui::Render();

        glViewport(0, 0, app.viewport().width, app.viewport().height);
        glClearColor(0.05f, 0.05f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // ── Cleanup ─────────────────────────────────────────────
    g_app = nullptr;
    // App destructor handles GPU cleanup

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    printf("CURSDAR shutdown complete.\n");
    return 0;
}
