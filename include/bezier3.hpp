#include <math.h>
#include <vector>

class Bezier3 {
public:
    Bezier3(float _P[8]);
    
    void eval(float t, float result[2]);
    void draw(int N);
    int closest(float point[2], float threshold=0.02);
    
    float nodes[8];
};
