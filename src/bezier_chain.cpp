#include <math.h>

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>

#include "bezier_chain.hpp"

using namespace std;

//BezierChain::BezierChain(std::vector<Bezier3> _beziers) {
//    nBeziers = (int)_beziers.size();
//    beziers = (Bezier3 *)malloc(nBeziers * sizeof(Bezier3));
//    
//    for (int i=0; i<(int)_beziers.size(); i++) {
//        beziers[i] = _beziers[i];
//    }
//}
//
//BezierChain::BezierChain(vector<float> _P) {
//    int i, j, j0;
//    nBeziers = (int)(_P.size() / 3);
//    beziers = (Bezier3 *)malloc(nBeziers * sizeof(Bezier3));
//    
//    if ((int)_P.size() % 6 != 2) {
//        fprintf(stderr, "Wrong number of bezier nodes or dimension\n");
//        exit(1);
//    }
//    
//    j0 = 0;
//    for (i=0; i<nBeziers; i++) {
//        vector<float> temp;
//        for (j=0; j<8; j++) {
//            temp.push_back(_P[j0+j]);
//        }
//        
//        beziers[i] = Bezier3(temp);
//        
//        j0 += 6;
//    }
//}

BezierChain::BezierChain(int N) {
    int i;
    float x, y;
    float dx = 1.6 / (float)N;
    float temp[8];
    nBeziers = N;
    beziers = (Bezier3 *)malloc(nBeziers * sizeof(Bezier3));
    
    x = -0.8;
    y = 0.5;
    
    for (i=0; i<N; i++) {
        
        temp[0] = x; temp[1] = 0;
        temp[2] = x; temp[3] = y;
        
        x += dx;
        
        temp[4] = x; temp[5] = y;
        temp[6] = x; temp[7] = 0;
        
        y *= -1;
        beziers[i] = Bezier3(temp);
    }
}

void BezierChain::eval(float t, float result[2]) {
    t *= nBeziers;
    t = fmax(0, t - 1e-3);
    int loc = (int)t;
    t -= loc;
    
    beziers[loc].eval(t, result);
}

void BezierChain::draw(int N) {
    for (int i=0; i<nBeziers; i++) {
        beziers[i].draw(N);
    }
}

void BezierChain::closest(float point[2], int result[2], float threshold) {
    int i, j;
    float dist;
    float minDist = 1000;
    
    for (j=0; j<nBeziers; j++) {
        for (i=0; i<4; i++) {
            dist = sqrt(pow(point[0] - beziers[j].nodes[2 * i], 2) + pow(point[1] - beziers[j].nodes[2 * i + 1], 2));
        
            if (dist < minDist) {
                minDist = dist;
                result[0] = j;
                result[1] = i;
            }
        }
    }
    
    if (minDist > threshold) {
        result[0] = -1;
        result[1] = -1;
    }
//     else {
//         fprintf(stderr, "Mindist = %.2f, thr = %.2f\n", minDist, threshold);
//     }
}

void BezierChain::update(int loc[2], float point[2]) {
    if (locked) {
        updateLocked(loc, point);
    }
    
    if (loc[1] == 0) {
        for (int i=0; i<2; i++) {
            beziers[loc[0]].nodes[2 + i] += point[i] - beziers[loc[0]].nodes[i];
            
            if (loc[0] > 0) {
                beziers[loc[0] - 1].nodes[6 + i] = point[i];
                beziers[loc[0] - 1].nodes[4 + i] += point[i] - beziers[loc[0]].nodes[i];
            }
            else if (locked && i == 1) {
                beziers[nBeziers - 1].nodes[6 + i] = point[i];
                beziers[nBeziers - 1].nodes[4 + i] += point[i] - beziers[loc[0]].nodes[i];
            }
            
            beziers[loc[0]].nodes[i] = point[i];
        }
    }
    else if (loc[1] == 3) {
        for (int i=0; i<2; i++) {
            beziers[loc[0]].nodes[4 + i] += point[i] - beziers[loc[0]].nodes[6 + i];
            
            if (loc[0] < nBeziers - 1) {
                beziers[loc[0] + 1].nodes[2 + i] += point[i] - beziers[loc[0]].nodes[6 + i];
                beziers[loc[0] + 1].nodes[i] = point[i];
            }
            else if (locked && i == 1) {
                beziers[0].nodes[2 + i] += point[i] - beziers[loc[0]].nodes[6 + i];
                beziers[0].nodes[i] = point[i];
            }
            
            beziers[loc[0]].nodes[6 + i] = point[i];
        }
        
        beziers[loc[0]].nodes[6] = point[0];
        beziers[loc[0]].nodes[7] = point[1];
        
        if (loc[0] < nBeziers - 1) {
            beziers[loc[0] + 1].nodes[0] = point[0];
            beziers[loc[0] + 1].nodes[1] = point[1];
        }
    }
    else if (loc[1] == 1) {
        beziers[loc[0]].nodes[2] = point[0];
        beziers[loc[0]].nodes[3] = point[1];
        
        if (loc[0] > 0) {
            for (int i=0; i<2; i++) {
                beziers[loc[0] - 1].nodes[4 + i] = 2 * beziers[loc[0]].nodes[i] - beziers[loc[0]].nodes[2 + i];
            }
        }
        else if (locked) {
            beziers[nBeziers - 1].nodes[4] = beziers[loc[0]].nodes[0] - beziers[loc[0]].nodes[2] + beziers[nBeziers - 1].nodes[6];
            beziers[nBeziers - 1].nodes[5] = 2 * beziers[loc[0]].nodes[1] - beziers[loc[0]].nodes[3];
        }
    }
    else if (loc[1] == 2) {
        beziers[loc[0]].nodes[4] = point[0];
        beziers[loc[0]].nodes[5] = point[1];
        
        if (loc[0] < nBeziers - 1){
            for (int i=0; i<2; i++) {
                beziers[loc[0] + 1].nodes[2 + i] = 2 * beziers[loc[0]].nodes[6 + i] - beziers[loc[0]].nodes[4 + i];
            }
        }
        else if (locked) {
            beziers[0].nodes[2] = beziers[loc[0]].nodes[6] - beziers[loc[0]].nodes[4] + beziers[0].nodes[0];
            beziers[0].nodes[3] = 2 * beziers[loc[0]].nodes[7] - beziers[loc[0]].nodes[5];
        }
    }
}

void BezierChain::updateLocked(int loc[2], float point[2]) {
    float dxMin, dxMax;
    float dyMin, dyMax;
    
    if (loc[0] == 0 && loc[1] < 2) {
        if (loc[1] == 0) {
            dxMin = 0; //xmin - beziers[0].nodes[0];
            dxMax = 0; //beziers[0].nodes[4] - beziers[0].nodes[2];
            
            dyMin = ymin - beziers[0].nodes[1];
            dyMax = ymax - beziers[0].nodes[1];
//             fprintf(stderr, "Case A, ");
        }
        else {
            dxMin = beziers[0].nodes[0] - beziers[0].nodes[2];
            dxMax = beziers[0].nodes[4] - beziers[0].nodes[2];
            
            dyMin = ymin - beziers[0].nodes[3];
            dyMax = ymax - beziers[0].nodes[3];
//             fprintf(stderr, "Case B, ");
        }
    }
    
    else if (loc[0] == nBeziers - 1 && loc[1] > 1) {
        if (loc[1] == 3) {
            dxMin = 0; //beziers[loc[0]].nodes[2] - beziers[loc[0]].nodes[4];
            dxMax = 0; //xmax - beziers[loc[0]].nodes[6];
            
            dyMin = ymin - beziers[loc[0]].nodes[7];
            dyMax = ymax - beziers[loc[0]].nodes[7];
//             fprintf(stderr, "Case C, ");
        }
        else {
            dxMin = beziers[loc[0]].nodes[2] - beziers[loc[0]].nodes[4];
            dxMax = beziers[loc[0]].nodes[6] - beziers[loc[0]].nodes[4];
            
            dyMin = ymin - beziers[loc[0]].nodes[5];
            dyMax = ymax - beziers[loc[0]].nodes[5];
//             fprintf(stderr, "Case D, ");
        }
    }
    
    else {
        if (loc[1] == 0) {
            dxMin = beziers[loc[0] - 1].nodes[2] - beziers[loc[0] - 1].nodes[4];
            dxMax = beziers[loc[0]].nodes[4] - beziers[loc[0]].nodes[2];
            
            dyMin = ymin - fmin(beziers[loc[0] - 1].nodes[5], beziers[loc[0]].nodes[3]);
            dyMax = ymax - fmax(beziers[loc[0] - 1].nodes[5], beziers[loc[0]].nodes[3]);
//             fprintf(stderr, "Case E, ");
        }
        else if (loc[1] == 3) {
            dxMin = beziers[loc[0]].nodes[2] - beziers[loc[0]].nodes[4];
            dxMax = beziers[loc[0] + 1].nodes[4] - beziers[loc[0] + 1].nodes[2];
            
            dyMin = ymin - fmin(beziers[loc[0]].nodes[5], beziers[loc[0] + 1].nodes[3]);
            dyMax = ymax - fmax(beziers[loc[0]].nodes[5], beziers[loc[0] + 1].nodes[3]);
//             fprintf(stderr, "Case F, ");
        }
        else if (loc[1] == 1) {
            dxMax = fmin(
                beziers[loc[0] - 1].nodes[4] - beziers[loc[0] - 1].nodes[2],
                beziers[loc[0]].nodes[4] - beziers[loc[0]].nodes[2]
            );
            dxMin = beziers[loc[0]].nodes[0] - beziers[loc[0]].nodes[2];
            
            dyMin = fmax(
                ymin - beziers[loc[0]].nodes[3],
                beziers[loc[0] - 1].nodes[5] - ymax
            );
            dyMax = fmin(
                ymax - beziers[loc[0]].nodes[3],
                beziers[loc[0] - 1].nodes[5] - ymin
            );
//             fprintf(stderr, "Case G, ");
        }
        else {
            dxMin = fmax(
                beziers[loc[0] + 1].nodes[2] - beziers[loc[0] + 1].nodes[4],
                beziers[loc[0]].nodes[2] - beziers[loc[0]].nodes[4]
            );
            dxMax = beziers[loc[0]].nodes[6] - beziers[loc[0]].nodes[4];
            
            dyMax = fmin(
                ymax - beziers[loc[0]].nodes[5],
                beziers[loc[0] + 1].nodes[3] - ymin
            );
            dyMin = fmax(
                ymin - beziers[loc[0]].nodes[5],
                beziers[loc[0] + 1].nodes[3] - ymax
            );
//             fprintf(stderr, "Case H, ");
        }
    }
    
//     fprintf(stderr, "dx in (%.2f, %.2f), dy in (%.2f, %.2f)\n", dxMin, dxMax, dyMin, dyMax);
    
    point[0] = beziers[loc[0]].nodes[2 * loc[1]] + fmin(dxMax, fmax(dxMin, point[0] - beziers[loc[0]].nodes[2 * loc[1]]));
    point[1] = beziers[loc[0]].nodes[2 * loc[1] + 1] + fmin(dyMax, fmax(dyMin, point[1] - beziers[loc[0]].nodes[2 * loc[1] + 1]));
}

float BezierChain::evalX(float x) {
    float tmin = 0;
    float tmax = 1;
    float tnew = 0.5;
    
    float precision = 1e-3;
    
    float pnew[2];
    eval(tnew, pnew);
    
    while (fabs(pnew[0] - x) > precision) {
        if (pnew[0] > x) {
            tmax = tnew;
        }
        else {
            tmin = tnew;
        }
        
        tnew = (tmax + tmin) * 0.5;
        eval(tnew, pnew);
    }
    
    return pnew[1];
}














