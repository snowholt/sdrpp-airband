#pragma once
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <functional>

// Minimal test registry. The repository has no test infrastructure and no test
// library is installed, so this file is the substrate the rest of the harness
// is built on. It is deliberately small enough to read in one sitting.
namespace tf {
    struct Case {
        const char* name;
        std::function<void()> fn;
    };

    inline std::vector<Case>& registry() {
        static std::vector<Case> r;
        return r;
    }

    inline int& failures() {
        static int f = 0;
        return f;
    }

    struct Reg {
        Reg(const char* n, std::function<void()> f) { registry().push_back({ n, f }); }
    };

    inline void fail(const char* file, int line, const std::string& msg) {
        std::fprintf(stderr, "    FAIL %s:%d  %s\n", file, line, msg.c_str());
        failures()++;
    }

    inline int runAll() {
        int failedCases = 0;
        for (auto& c : registry()) {
            int before = failures();
            std::printf("[ RUN  ] %s\n", c.name);
            c.fn();
            if (failures() > before) {
                std::printf("[ FAIL ] %s\n", c.name);
                failedCases++;
            }
            else {
                std::printf("[  OK  ] %s\n", c.name);
            }
        }
        std::printf("\n%d test(s), %d failed, %d assertion failure(s)\n",
                    (int)registry().size(), failedCases, failures());
        return failedCases ? 1 : 0;
    }
}

#define TEST(name)                                  \
    static void name();                             \
    static tf::Reg tf_reg_##name(#name, name);      \
    static void name()

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) { tf::fail(__FILE__, __LINE__, "CHECK(" #cond ")"); } \
    } while (0)

#define CHECK_EQ(a, b)                                                  \
    do {                                                                \
        auto tf_a = (a); auto tf_b = (b);                               \
        if (!(tf_a == tf_b)) {                                          \
            tf::fail(__FILE__, __LINE__, "CHECK_EQ(" #a ", " #b ") got " \
                     + std::to_string(tf_a) + " vs " + std::to_string(tf_b)); \
        }                                                               \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                           \
    do {                                                                \
        double tf_a = (a), tf_b = (b), tf_t = (tol);                    \
        if (!(std::fabs(tf_a - tf_b) <= tf_t)) {                        \
            tf::fail(__FILE__, __LINE__, "CHECK_NEAR(" #a ", " #b ") got " \
                     + std::to_string(tf_a) + " vs " + std::to_string(tf_b)); \
        }                                                               \
    } while (0)
