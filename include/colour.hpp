#include <vector>

class Colour {
public:
    Colour(std::vector<float> _x, std::vector< std::vector<float> > _y, int _size);
    
    void apply(float *colourMap);
    std::vector<float> get(float p);
    
    std::vector< std::vector<float> > map;
    
    int size;
};