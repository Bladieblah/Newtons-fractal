#include <math.h>

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>

#include "bezier3.hpp"

using namespace std;

Bezier3::Bezier3(float _P[8]) {
    for (int i=0; i<8; i++) {
        nodes[i] = _P[i];
    }
}

void Bezier3::eval(float t, float result[2]) {
    float weights[4];
    float dummy = 1 - t;
    
    weights[0] = pow(dummy, 3);
    weights[1] = 3. * pow(dummy, 2) * t;
    weights[2] = 3. * dummy * t * t;
    weights[3] = pow(t, 3);
    
    for (int i=0; i<2; i++) {
        result[i] = 0;
        for (int j=0; j<4; j++) {
            result[i] += weights[j] * nodes[2 * j + i];
        }
    }
}

void Bezier3::draw(int N) {
    float X[2];
    float dt = 1./(N + 1);
    
    glLineWidth(3);
    
    glBegin(GL_LINE_STRIP);
    for (float t=0.; t<=1; t+=dt) {
        eval(t, X);
        glVertex2f(X[0], X[1]);
    }
    glEnd();
    
    glPointSize(7);
    glEnable(GL_POINT_SMOOTH);
    
    glBegin(GL_POINTS);
    for (int i=0; i<4; i++) {
        glVertex2f(nodes[2 * i], nodes[2 * i + 1]);
    }
    glEnd();
}

int Bezier3::closest(float point[2], float threshold) {
    int i, minLoc;
    float dist;
    float minDist = 1000;
    
    for (i=0; i<4; i++) {
        dist = sqrt(pow(point[0] - nodes[2 * i], 2) + pow(point[1] - nodes[2 * i + 1], 2));
        
        if (dist < minDist) {
            minDist = dist;
            minLoc = i;
        }
    }
    
    if (minDist <= threshold) {
        return minLoc;
    }
    
    return -1;
}
