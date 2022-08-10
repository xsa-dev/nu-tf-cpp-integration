#include "src/model.h"
#include <iostream>
using keras2cpp::Model;
using keras2cpp::Tensor;

int main(int args, char* argv[]) {
    // Initialize model.
    auto model = Model::load("example.model");
    float a, b, c;
    a = atof(argv[1]);
    b = atof(argv[2]);
    c = atof(argv[3]);
    // Create a 1D Tensor on length 10 for input data.
    std::cout << "Team 1 points: " << a << std::endl;
    std::cout << "Team 2 points: " << b << std::endl;
    std::cout << "Time: " << c << std::endl;
    Tensor in{3};
    in.data_ = {a, b, c};

    // Run prediction.
    Tensor out = model(in);
    std::cout << "Match result: " << std::endl;
    out.print();
    return 0;
}
