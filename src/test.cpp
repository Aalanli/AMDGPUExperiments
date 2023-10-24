#include "test.hpp"

void Test::foo(int a, int b) {
    this->a = a;
    this->b = b;
}

void do_something() {
    std::cout << "Doing something" << std::endl;
}