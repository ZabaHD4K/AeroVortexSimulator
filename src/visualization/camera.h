#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
public:
    float distance = 3.0f;
    float yaw = 45.0f;    // degrees
    float pitch = 30.0f;  // degrees
    glm::vec3 target{0.0f};

    float fov = 45.0f;
    float nearPlane = 0.01f;
    float farPlane = 100.0f;

    glm::vec3 getPosition() const {
        float yr = glm::radians(yaw);
        float pr = glm::radians(pitch);
        return target + glm::vec3(
            distance * cos(pr) * cos(yr),
            distance * sin(pr),
            distance * cos(pr) * sin(yr)
        );
    }

    glm::mat4 getViewMatrix() const {
        return glm::lookAt(getPosition(), target, glm::vec3(0, 1, 0));
    }

    glm::mat4 getProjectionMatrix(float aspect) const {
        return glm::perspective(glm::radians(fov), aspect, nearPlane, farPlane);
    }

    void orbit(float dx, float dy) {
        yaw   += dx * 0.3f;
        pitch += dy * 0.3f;
        pitch = glm::clamp(pitch, -89.0f, 89.0f);
    }

    void pan(float dx, float dy) {
        float yr = glm::radians(yaw);
        glm::vec3 right = glm::vec3(-sin(yr), 0, cos(yr));
        glm::vec3 up    = glm::vec3(0, 1, 0);
        float scale = distance * 0.002f;
        target += (-right * dx + up * dy) * scale;
    }

    void zoom(float delta) {
        distance *= (1.0f - delta * 0.1f);
        distance = glm::clamp(distance, 0.1f, 50.0f);
    }

    void reset() {
        distance = 3.0f;
        yaw = 45.0f;
        pitch = 30.0f;
        target = glm::vec3(0.0f);
    }
};
