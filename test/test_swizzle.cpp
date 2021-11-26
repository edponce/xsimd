/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "test_utils.hpp"

#if !XSIMD_WITH_NEON && !XSIMD_WITH_NEON64 && !(XSIMD_WITH_AVX && !XSIMD_WITH_AVX2)
namespace xsimd
{
    template <typename T, std::size_t N>
    struct init_swizzle_base
    {
        using swizzle_vector_type = std::array<T, N>;
        swizzle_vector_type lhs_in, exped;

        template <int... Indices>
        std::vector<swizzle_vector_type> create_swizzle_vectors()
        {
            std::vector<swizzle_vector_type> vects;

            /* Generate input data */
            for (std::size_t i = 0; i < N; ++i)
            {
                lhs_in[i] = 2 * i + 1;
            }
            vects.push_back(std::move(lhs_in));

            /* Expected reversed data */
            for (std::size_t i = 0; i < N; ++i)
            {
                exped[i] = lhs_in[N - 1 - i];
            }
            vects.push_back(std::move(exped));

            return vects;
        }
    };
}

struct Reversor
{
    static constexpr unsigned get(unsigned i, unsigned n)
    {
        return n - 1 - i;
    }
};

template <class B>
class swizzle_test : public testing::Test
{
protected:
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;

    swizzle_test()
    {
        std::cout << "swizzle tests" << std::endl;
    }

    void swizzle_reverse()
    {
        xsimd::init_swizzle_base<value_type, size> swizzle_base;
        auto swizzle_vecs = swizzle_base.create_swizzle_vectors();
        auto v_lhs = swizzle_vecs[0];
        auto v_exped = swizzle_vecs[1];

        B b_lhs = B::load_unaligned(v_lhs.data());
        B b_exped = B::load_unaligned(v_exped.data());

        B b_res = xsimd::swizzle(b_lhs, xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<batch_type>, Reversor>());
        EXPECT_BATCH_EQ(b_res, b_exped) << print_function_name("swizzle test");
    }
};

TYPED_TEST_SUITE(swizzle_test, batch_float_types, simd_test_names);

TYPED_TEST(swizzle_test, swizzle_reverse)
{
    this->swizzle_reverse();
}
#endif
