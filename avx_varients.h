
static inline void * _rep_movsb(void *d, const void *s, size_t n)
{
    asm volatile("rep movsb"
                 : "=D"(d), "=S"(s), "=c"(n)
                 : "0"(d), "1"(s), "2"(n)
                 : "memory");
}

static inline void * _avx_cpy(void *d, const void *s, size_t n)
{
    // d, s -> 32 byte aligned
    // n -> multiple of 32

    auto *dVec = reinterpret_cast<__m256i *>(d);
    const auto *sVec = reinterpret_cast<const __m256i *>(s);
    size_t nVec = n / sizeof(__m256i);
    for (; nVec > 0; nVec--, sVec++, dVec++)
    {
        const __m256i temp = _mm256_load_si256(sVec);
        _mm256_store_si256(dVec, temp);
    }
}

static inline void * _avx_async_cpy(void *d, const void *s, size_t n)
{
    // d, s -> 32 byte aligned
    // n -> multiple of 32

    auto *dVec = reinterpret_cast<__m256i *>(d);
    const auto *sVec = reinterpret_cast<const __m256i *>(s);
    size_t nVec = n / sizeof(__m256i);
    for (; nVec > 0; nVec--, sVec++, dVec++)
    {
        const __m256i temp = _mm256_stream_load_si256(sVec);
        _mm256_stream_si256(dVec, temp);
    }
    _mm_sfence();
}

static inline void * _avx_async_pf_cpy(void *d, const void *s, size_t n)
{
    // d, s -> 64 byte aligned
    // n -> multiple of 64

    auto *dVec = reinterpret_cast<__m256i *>(d);
    const auto *sVec = reinterpret_cast<const __m256i *>(s);
    size_t nVec = n / sizeof(__m256i);
    for (; nVec > 2; nVec -= 2, sVec += 2, dVec += 2)
    {
        // prefetch the next iteration's data
        // by default _mm_prefetch moves the entire cache-lint (64b)
        _mm_prefetch(sVec + 2, _MM_HINT_T0);

        _mm256_stream_si256(dVec, _mm256_load_si256(sVec));
        _mm256_stream_si256(dVec + 1, _mm256_load_si256(sVec + 1));
    }
    _mm256_stream_si256(dVec, _mm256_load_si256(sVec));
    _mm256_stream_si256(dVec + 1, _mm256_load_si256(sVec + 1));
    _mm_sfence();
}

static inline void * _avx_cpy_unroll(void *d, const void *s, size_t n)
{
    // d, s -> 128 byte aligned
    // n -> multiple of 128

    auto *dVec = reinterpret_cast<__m256i *>(d);
    const auto *sVec = reinterpret_cast<const __m256i *>(s);
    size_t nVec = n / sizeof(__m256i);
    for (; nVec > 0; nVec -= 4, sVec += 4, dVec += 4)
    {
        _mm256_store_si256(dVec, _mm256_load_si256(sVec));
        _mm256_store_si256(dVec + 1, _mm256_load_si256(sVec + 1));
        _mm256_store_si256(dVec + 2, _mm256_load_si256(sVec + 2));
        _mm256_store_si256(dVec + 3, _mm256_load_si256(sVec + 3));
    }
}

static inline void * _avx_async_cpy_unroll(void *d, const void *s, size_t n)
{
    // d, s -> 128 byte aligned
    // n -> multiple of 128

    auto *dVec = reinterpret_cast<__m256i *>(d);
    const auto *sVec = reinterpret_cast<const __m256i *>(s);
    size_t nVec = n / sizeof(__m256i);
    for (; nVec > 0; nVec -= 4, sVec += 4, dVec += 4)
    {
        _mm256_stream_si256(dVec, _mm256_stream_load_si256(sVec));
        _mm256_stream_si256(dVec + 1, _mm256_stream_load_si256(sVec + 1));
        _mm256_stream_si256(dVec + 2, _mm256_stream_load_si256(sVec + 2));
        _mm256_stream_si256(dVec + 3, _mm256_stream_load_si256(sVec + 3));
    }
    _mm_sfence();
}

static inline void * _avx_async_pf_cpy_unroll(void *d, const void *s, size_t n)
{
    // d, s -> 128 byte aligned
    // n -> multiple of 128

    auto *dVec = reinterpret_cast<__m256i *>(d);
    const auto *sVec = reinterpret_cast<const __m256i *>(s);
    size_t nVec = n / sizeof(__m256i);
    for (; nVec > 4; nVec -= 4, sVec += 4, dVec += 4)
    {
        // prefetch data for next iteration
        _mm_prefetch(sVec + 4, _MM_HINT_T0);
        _mm_prefetch(sVec + 6, _MM_HINT_T0);
        _mm256_stream_si256(dVec, _mm256_load_si256(sVec));
        _mm256_stream_si256(dVec + 1, _mm256_load_si256(sVec + 1));
        _mm256_stream_si256(dVec + 2, _mm256_load_si256(sVec + 2));
        _mm256_stream_si256(dVec + 3, _mm256_load_si256(sVec + 3));
    }
    _mm256_stream_si256(dVec, _mm256_load_si256(sVec));
    _mm256_stream_si256(dVec + 1, _mm256_load_si256(sVec + 1));
    _mm256_stream_si256(dVec + 2, _mm256_load_si256(sVec + 2));
    _mm256_stream_si256(dVec + 3, _mm256_load_si256(sVec + 3));
    _mm_sfence();
}