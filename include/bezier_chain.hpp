#include <math.h>

#include "bezier3.hpp"

class BezierChain {
public:
    BezierChain(int N);
    
    void eval(float t, float result[2]);
    void draw(int N);
    void closest(float point[2], int result[2], float threshold=0.01);
    void update(int loc[2], float point[2]);
    void updateLocked(int loc[2], float point[2]);
    float evalX(float x);
    
    Bezier3 *beziers;
    int nBeziers;
    bool locked = false;
    
    float ymin = -0.5;
    float ymax =  0.5;
    float xmin = -0.8;
    float xmax =  0.8;
};
