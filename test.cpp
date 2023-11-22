#include <stdio.h>

struct A {
    virtual void foo() {
        printf("a");
    }

    void dispatch() {
        foo();
    }
};

struct B : A {
    void foo() override {
        printf("b");
    }
};

int main() {
    auto a = A();
    auto b = B();
    a.dispatch();
    b.dispatch();
}