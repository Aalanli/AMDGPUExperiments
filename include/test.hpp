#pragma once

#include <iostream>

struct Test {
    int a;
    int b;

    void foo(int a, int b);
};

struct Test1: Test {
    float c;
};

extern "C" void do_something();