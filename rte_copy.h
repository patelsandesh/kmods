/**
 * Copy bytes from one location to another. The locations must not overlap.
 *
 * @note This is implemented as a macro, so it's address should not be taken
 * and care is needed as parameter expressions may be evaluated multiple times.
 *
 * @param dst
 *   Pointer to the destination of the data.
 * @param src
 *   Pointer to the source data.
 * @param n
 *   Number of bytes to copy.
 * @return
 *   Pointer to the destination data.
 */
static inline void *
rte_memcpy(void *dst, const void *src, size_t n);

static inline void *
rte_mov15_or_less(void *dst, const void *src, size_t n)
{
    /**
     * Use the following structs to avoid violating C standard
     * alignment requirements and to avoid strict aliasing bugs
     */
    memcpy(dst, src, n);
    return dst;
}
/**
 * Copy 16 bytes from one location to another,
 * locations should not overlap.
 */
static inline void
rte_mov16(uint8_t *dst, const uint8_t *src)
{
    memcpy(dst, src, 16);
}

/**
 * Copy 32 bytes from one location to another,
 * locations should not overlap.
 */
static inline void
rte_mov32(uint8_t *dst, const uint8_t *src)
{
    asm volatile("vmovdqu32 %1,%%zmm1\n\t"
                 "vmovdqu32 %%zmm1,%0\n\t"
                 "sfence"
                 : "=m"(dst)
                 : "m"(*src));
}

/**
 * Copy 64 bytes from one location to another,
 * locations should not overlap.
 */
static inline void
rte_mov64(uint8_t *dst, const uint8_t *src)
{
    // asm volatile("vmovdqu64 %1,%%zmm1\n\t"
    //              "vmovdqu64 %%zmm1,%0\n\t"

    //              : "=m"(dst)
    //              : "m"(*src));

    memcpy(dst, src, 64);
    // check why assembly gives segfault?
}

/**
 * Copy 128 bytes from one location to another,
 * locations should not overlap.
 */
static inline void
rte_mov128(uint8_t *dst, const uint8_t *src)
{
    rte_mov64(dst + 0 * 64, src + 0 * 64);
    rte_mov64(dst + 1 * 64, src + 1 * 64);
}

/**
 * Copy 256 bytes from one location to another,
 * locations should not overlap.
 */
static inline void
rte_mov256(uint8_t *dst, const uint8_t *src)
{
    rte_mov64(dst + 0 * 64, src + 0 * 64);
    rte_mov64(dst + 1 * 64, src + 1 * 64);
    rte_mov64(dst + 2 * 64, src + 2 * 64);
    rte_mov64(dst + 3 * 64, src + 3 * 64);
}

/**
 * Copy 128-byte blocks from one location to another,
 * locations should not overlap.
 */
static inline void
rte_mov128blocks(uint8_t *dst, const uint8_t *src, size_t n)
{
    while (n >= 128 + 128 * 2)
    {
        __builtin_prefetch(dst + 256 + 64 * 0, 0, 2);
        __builtin_prefetch(dst + 256 + 64 * 1, 0, 2);
        asm volatile("vmovdqa64 %2,%%zmm0\n\t"
                     "vmovdqa64 %3,%%zmm1\n\t"
                     "vmovdqa64 %%zmm0,%0\n\t"
                     "vmovdqa64 %%zmm1,%1\n\t"
                     "sfence"
                     : "=m"((dst[0 * 64])), "=m"((dst[1 * 64]))
                     : "m"(*(src + 0 * 64)), "m"(*(src + 1 * 64)));
        n -= 128;
        src = src + 128;
        dst = dst + 128;
    }
    asm volatile("vmovdqa64 %2,%%zmm0\n\t"
                 "vmovdqa64 %3,%%zmm1\n\t"
                 "vmovdqa64 %%zmm0,%0\n\t"
                 "vmovdqa64 %%zmm1,%1\n\t"
                 "sfence"
                 : "=m"((dst[0 * 64])), "=m"((dst[1 * 64]))
                 : "m"(*(src + 0 * 64)), "m"(*(src + 1 * 64)));
    n -= 128;
    src = src + 128;
    dst = dst + 128;

    asm volatile("vmovdqa64 %2,%%zmm0\n\t"
                 "vmovdqa64 %3,%%zmm1\n\t"
                 "vmovdqa64 %%zmm0,%0\n\t"
                 "vmovdqa64 %%zmm1,%1\n\t"
                 "sfence"
                 : "=m"((dst[0 * 64])), "=m"((dst[1 * 64]))
                 : "m"(*(src + 0 * 64)), "m"(*(src + 1 * 64)));
    n -= 128;
    src = src + 128;
    dst = dst + 128;
}

/**
 * Copy 512-byte blocks from one location to another,
 * locations should not overlap.
 */
static inline void
rte_mov256blocks(uint8_t *dst, const uint8_t *src, size_t n)
{

    while (n >= 256 + 256 * 2)
    {
        __builtin_prefetch(dst + 512 + 64 * 0, 0, 2);
        __builtin_prefetch(dst + 512 + 64 * 1, 0, 2);
        __builtin_prefetch(dst + 512 + 64 * 2, 0, 2);
        __builtin_prefetch(dst + 512 + 64 * 3, 0, 2);
        asm volatile("vmovdqa64 %4,%%zmm0\n\t"
                     "vmovdqa64 %5,%%zmm1\n\t"
                     "vmovdqa64 %6,%%zmm2\n\t"
                     "vmovdqa64 %7,%%zmm3\n\t"

                     "vmovdqa64 %%zmm0,%0\n\t"
                     "vmovdqa64 %%zmm1,%1\n\t"
                     "vmovdqa64 %%zmm2,%2\n\t"
                     "vmovdqa64 %%zmm3,%3\n\t"
                     "sfence"

                     : "=m"(dst[0 * 64]), "=m"(dst[1 * 64]),
                       "=m"(dst[2 * 64]), "=m"(dst[3 * 64])
                     : "m"(*(src + 0 * 64)), "m"(*(src + 1 * 64)),
                       "m"(*(src + 2 * 64)), "m"(*(src + 3 * 64)));
        n -= 256;
        src = src + 256;
        dst = dst + 256;
    }

    asm volatile("vmovdqa64 %4,%%zmm0\n\t"
                 "vmovdqa64 %5,%%zmm1\n\t"
                 "vmovdqa64 %6,%%zmm2\n\t"
                 "vmovdqa64 %7,%%zmm3\n\t"

                 "vmovdqa64 %%zmm0,%0\n\t"
                 "vmovdqa64 %%zmm1,%1\n\t"
                 "vmovdqa64 %%zmm2,%2\n\t"
                 "vmovdqa64 %%zmm3,%3\n\t"
                 "sfence"

                 : "=m"(dst[0 * 64]), "=m"(dst[1 * 64]),
                   "=m"(dst[2 * 64]), "=m"(dst[3 * 64])
                 : "m"(*(src + 0 * 64)), "m"(*(src + 1 * 64)),
                   "m"(*(src + 2 * 64)), "m"(*(src + 3 * 64)));
    n -= 256;
    src = src + 256;
    dst = dst + 256;

    asm volatile("vmovdqa64 %4,%%zmm0\n\t"
                 "vmovdqa64 %5,%%zmm1\n\t"
                 "vmovdqa64 %6,%%zmm2\n\t"
                 "vmovdqa64 %7,%%zmm3\n\t"

                 "vmovdqa64 %%zmm0,%0\n\t"
                 "vmovdqa64 %%zmm1,%1\n\t"
                 "vmovdqa64 %%zmm2,%2\n\t"
                 "vmovdqa64 %%zmm3,%3\n\t"
                 "sfence"

                 : "=m"(dst[0 * 64]), "=m"(dst[1 * 64]),
                   "=m"(dst[2 * 64]), "=m"(dst[3 * 64])
                 : "m"(*(src + 0 * 64)), "m"(*(src + 1 * 64)),
                   "m"(*(src + 2 * 64)), "m"(*(src + 3 * 64)));
    n -= 256;
    src = src + 256;
    dst = dst + 256;
}

static inline void *
rte_memcpy_generic(void *dst, const void *src, size_t n)
{
    void *ret = dst;
    size_t dstofss;
    size_t bits;

    /**
     * Copy less than 16 bytes
     */
    if (n < 16)
    {
        return rte_mov15_or_less(dst, src, n);
    }

    /**
     * Fast way when copy size doesn't exceed 512 bytes
     */
    if (n <= 32)
    {
        rte_mov16((uint8_t *)dst, (const uint8_t *)src);
        rte_mov16((uint8_t *)dst - 16 + n,
                  (const uint8_t *)src - 16 + n);
        return ret;
    }
    if (n <= 64)
    {
        rte_mov32((uint8_t *)dst, (const uint8_t *)src);
        rte_mov32((uint8_t *)dst - 32 + n,
                  (const uint8_t *)src - 32 + n);
        return ret;
    }
    if (n <= 512)
    {
        if (n >= 256)
        {
            n -= 256;
            rte_mov256((uint8_t *)dst, (const uint8_t *)src);
            src = (const uint8_t *)src + 256;
            dst = (uint8_t *)dst + 256;
        }
        if (n >= 128)
        {
            n -= 128;
            rte_mov128((uint8_t *)dst, (const uint8_t *)src);
            src = (const uint8_t *)src + 128;
            dst = (uint8_t *)dst + 128;
        }
    COPY_BLOCK_128_BACK63:
        if (n > 64)
        {
            rte_mov64((uint8_t *)dst, (const uint8_t *)src);
            rte_mov64((uint8_t *)dst - 64 + n,
                      (const uint8_t *)src - 64 + n);
            return ret;
        }
        if (n > 0)
            rte_mov64((uint8_t *)dst - 64 + n,
                      (const uint8_t *)src - 64 + n);
        return ret;
    }

    /**
     * Make store aligned when copy size exceeds 512 bytes
     */
    dstofss = ((uintptr_t)dst & 0x3F);
    if (dstofss > 0)
    {
        dstofss = 64 - dstofss;
        n -= dstofss;
        rte_mov64((uint8_t *)dst, (const uint8_t *)src);
        src = (const uint8_t *)src + dstofss;
        dst = (uint8_t *)dst + dstofss;
    }

    /**
     * Copy 512-byte blocks.
     * Use copy block function for better instruction order control,
     * which is important when load is unaligned.
     */
    rte_mov256blocks((uint8_t *)dst, (const uint8_t *)src, n);
    bits = n;
    n = n & 511;
    bits -= n;
    src = (const uint8_t *)src + bits;
    dst = (uint8_t *)dst + bits;

    /**
     * Copy 128-byte blocks.
     * Use copy block function for better instruction order control,
     * which is important when load is unaligned.
     */
    if (n >= 128)
    {
        rte_mov128blocks((uint8_t *)dst, (const uint8_t *)src, n);
        bits = n;
        n = n & 127;
        bits -= n;
        src = (const uint8_t *)src + bits;
        dst = (uint8_t *)dst + bits;
    }

    /**
     * Copy whatever left
     */
    goto COPY_BLOCK_128_BACK63;
}

static inline void *
rte_memcpy(void *dst, const void *src, size_t n)
{
    return rte_memcpy_generic(dst, src, n);
}

static void *allocate(size_t size)
{
    void *ptr = mmap(
        nullptr,                     // Let OS choose the address
        size,                        // Size of mapping
        PROT_READ | PROT_WRITE,      // Read and write permissions
        MAP_PRIVATE | MAP_ANONYMOUS, // Private mapping, not backed by file
        -1,                          // File descriptor (not used with MAP_ANONYMOUS)
        0                            // Offset (not used with MAP_ANONYMOUS)
    );

    if (ptr == MAP_FAILED)
    {
        printf("failed to allocate\n");
        exit(1);
    }

    return ptr;
}
