/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#define E 2.71828182846
#define PI 3.141592653589

float gaussWeight(ivec2 offset, float sigma) {
    float dist = (offset.x * offset.x) + (offset.y * offset.y);
    return pow(E, -(dist / (2.0 * pow(sigma, 2.0))));
}
