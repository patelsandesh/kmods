#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <string.h>
#include <immintrin.h>
#include "x86intrin.h"
#include <sys/mman.h>
#include <linux/idxd.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>

#include <cstdlib>
#include <thread>
#include <vector>

static unsigned long n_gb = 2;  // Default 1 GB
static unsigned long k_kb = 16; // Default 4 MB
static void *array1;
static void *array2;
static bool arrays_allocated = false;
static unsigned long avx_last_copy_time_ns;   // Store last copy time in nanoseconds
static unsigned long avx_last_bandwidth_mbps; // Store last bandwidth in MB/s

static unsigned long string_last_copy_time_ns;   // Store last copy time in nanoseconds
static unsigned long string_last_bandwidth_mbps; // Store last bandwidth in MB/s

static bool inprogress = false;
static bool verified = true;

static void *dsa_wq = MAP_FAILED;
static int dedicated_mode = 0;
static int max_retry_count = 10000;
static int resubmit_copy_retry = 8;
static int top_retry_count;

#define DSA_WQ_SIZE 4096

#define GB_TO_BYTES(x) ((unsigned long)(x) << 30)
#define KB_TO_BYTES(x) ((unsigned long)(x) << 10)
#define ALIGNMENT_MASK 0x3F

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

static inline void _rep_movsb(void *d, const void *s, size_t n)
{
    asm volatile("rep movsb"
                 : "=D"(d), "=S"(s), "=c"(n)
                 : "0"(d), "1"(s), "2"(n)
                 : "memory");
}

static inline void _avx_cpy(void *d, const void *s, size_t n)
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

static inline void _avx_async_cpy(void *d, const void *s, size_t n)
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

static inline void _avx_async_pf_cpy(void *d, const void *s, size_t n)
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

static inline void _avx_cpy_unroll(void *d, const void *s, size_t n)
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

static inline void _avx_async_cpy_unroll(void *d, const void *s, size_t n)
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

static inline void _avx_async_pf_cpy_unroll(void *d, const void *s, size_t n)
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

static void deallocate(void *ptr, size_t size)
{
    if (munmap(ptr, size) == -1)
    {
        printf("failed to free\n");
        exit(1);
    }
}

static void cleanup_arrays(void)
{
    unsigned long size = GB_TO_BYTES(n_gb);
    if (arrays_allocated)
    {
        deallocate(array1, size);
        deallocate(array2, size);
        arrays_allocated = false;
        printf("Arrays freed\n");
    }
}

static int allocate_and_initialize_arrays(void)
{
    unsigned long size = GB_TO_BYTES(n_gb);

    cleanup_arrays(); // Clean up any existing arrays

    array1 = allocate(size);
    if (!array1)
    {
        printf("Failed to allocate array1\n");
        return -1;
    }

    array2 = allocate(size);
    if (!array2)
    {
        printf("Failed to allocate array2\n");
        free(array1);
        return -1;
    }

    // Initialize array1 with 1s
    memset(array1, 1, size);

    // Initialize array2 with 2s
    memset(array2, 2, size);

    arrays_allocated = true;
    printf("Arrays allocated and initialized: %lu GB each\n", n_gb);
    return 0;
}

static int verify_copy(void)
{
    unsigned long size = GB_TO_BYTES(n_gb);
    unsigned long i;

    for (i = 0; i < size; i++)
    {
        if (((char *)array1)[i] != ((char *)array2)[i])
        {
            printf("Verification failed at offset %lu, total size %lu, a1 %d a2 %d\n", i, size, ((char *)array1)[i], ((char *)array2)[i]);
            verified = false;
            return verified;
        }
    }

    printf("Verification successful\n");
    verified = true;
    return verified;
}

static void perform_random_copy_avx(void)
{
    unsigned long total_size = GB_TO_BYTES(n_gb);
    unsigned long chunk_size = KB_TO_BYTES(k_kb);
    unsigned long num_chunks = total_size / chunk_size;
    unsigned long *chunk_order;
    unsigned long i;
    struct timespec t1;
    unsigned long start_time, end_time;

    printf("Random copy started \n");

    inprogress = true;

    // Allocate array for random chunk order
    chunk_order = (unsigned long *)malloc(sizeof(unsigned long) * num_chunks);
    if (!chunk_order)
    {
        printf("Failed to allocate chunk order array\n");
        return;
    }

    // Initialize chunk order
    for (i = 0; i < num_chunks; i++)
    {
        chunk_order[i] = i;
    }

    // Shuffle chunk order
    for (i = num_chunks - 1; i > 0; i--)
    {
        unsigned long j = rand() % (i + 1);
        unsigned long temp = chunk_order[i];
        chunk_order[i] = chunk_order[j];
        chunk_order[j] = temp;
    }

    // Start timing
    clock_gettime(CLOCK_REALTIME, &t1);
    start_time = t1.tv_sec * 1000000000 + t1.tv_nsec;
    // Perform copies in random order
    for (i = 0; i < num_chunks; i++)
    {
        unsigned long offset = chunk_order[i] * chunk_size;
        rte_memcpy(array2 + offset, array1 + offset, chunk_size);
    }
    // End timing
    clock_gettime(CLOCK_REALTIME, &t1);
    end_time = t1.tv_sec * 1000000000 + t1.tv_nsec;
    ;

    // Calculate time taken and bandwidth
    avx_last_copy_time_ns = end_time - start_time;

    // Calculate bandwidth in MB/s
    // total_size in bytes / time in seconds = bytes per second
    // Convert to MB/s by dividing by 1024*1024
    avx_last_bandwidth_mbps = (unsigned long)total_size * 1000000000ULL / avx_last_copy_time_ns;
    avx_last_bandwidth_mbps = avx_last_bandwidth_mbps / (1024 * 1024);

    free(chunk_order);
    if (verify_copy() != true)
    {
        printf("Random copy verification failed  ns\n");
        // avx_last_bandwidth_mbps = 99999999999;
    }
    inprogress = false;
    printf("Copy_result \tAVX-asm\t Chunk_size %llu KB\t Time: %llu ms\t Bandwidth: %llu MB/s\n", k_kb, avx_last_copy_time_ns / 1000000, avx_last_bandwidth_mbps);
}

static void perform_random_copy_rep_movsb(void)
{
    unsigned long total_size = GB_TO_BYTES(n_gb);
    unsigned long chunk_size = KB_TO_BYTES(k_kb);
    unsigned long num_chunks = total_size / chunk_size;
    unsigned long *chunk_order;
    unsigned long i;
    struct timespec t1;
    unsigned long start_time, end_time;

    printf("Random copy started \n");

    inprogress = true;

    // Allocate array for random chunk order
    chunk_order = (unsigned long *)malloc(sizeof(unsigned long) * num_chunks);
    if (!chunk_order)
    {
        printf("Failed to allocate chunk order array\n");
        return;
    }

    // Initialize chunk order
    for (i = 0; i < num_chunks; i++)
    {
        chunk_order[i] = i;
    }

    // Shuffle chunk order
    for (i = num_chunks - 1; i > 0; i--)
    {
        unsigned long j = rand() % (i + 1);
        unsigned long temp = chunk_order[i];
        chunk_order[i] = chunk_order[j];
        chunk_order[j] = temp;
    }

    // Start timing
    clock_gettime(CLOCK_REALTIME, &t1);
    start_time = t1.tv_sec * 1000000000 + t1.tv_nsec;
    // Perform copies in random order
    for (i = 0; i < num_chunks; i++)
    {
        unsigned long offset = chunk_order[i] * chunk_size;
        _rep_movsb(array2 + offset, array1 + offset, chunk_size);
    }
    // End timing
    clock_gettime(CLOCK_REALTIME, &t1);
    end_time = t1.tv_sec * 1000000000 + t1.tv_nsec;
    ;

    // Calculate time taken and bandwidth
    avx_last_copy_time_ns = end_time - start_time;

    // Calculate bandwidth in MB/s
    // total_size in bytes / time in seconds = bytes per second
    // Convert to MB/s by dividing by 1024*1024
    avx_last_bandwidth_mbps = (unsigned long)total_size * 1000000000ULL / avx_last_copy_time_ns;
    avx_last_bandwidth_mbps = avx_last_bandwidth_mbps / (1024 * 1024);

    free(chunk_order);
    if (verify_copy() != true)
    {
        printf("Random copy verification failed  ns\n");
        // avx_last_bandwidth_mbps = 99999999999;
    }
    inprogress = false;
    printf("Copy_result \trep_movsb\t Chunk_size %llu KB\t Time: %llu ms\t Bandwidth: %llu MB/s\n", k_kb, avx_last_copy_time_ns / 1000000, avx_last_bandwidth_mbps);
}

static void perform_random_copy_avx_cpy(void)
{
    unsigned long total_size = GB_TO_BYTES(n_gb);
    unsigned long chunk_size = KB_TO_BYTES(k_kb);
    unsigned long num_chunks = total_size / chunk_size;
    unsigned long *chunk_order;
    unsigned long i;
    struct timespec t1;
    unsigned long start_time, end_time;

    printf("Random copy started \n");

    inprogress = true;

    // Allocate array for random chunk order
    chunk_order = (unsigned long *)malloc(sizeof(unsigned long) * num_chunks);
    if (!chunk_order)
    {
        printf("Failed to allocate chunk order array\n");
        return;
    }

    // Initialize chunk order
    for (i = 0; i < num_chunks; i++)
    {
        chunk_order[i] = i;
    }

    // Shuffle chunk order
    for (i = num_chunks - 1; i > 0; i--)
    {
        unsigned long j = rand() % (i + 1);
        unsigned long temp = chunk_order[i];
        chunk_order[i] = chunk_order[j];
        chunk_order[j] = temp;
    }

    // Start timing
    clock_gettime(CLOCK_REALTIME, &t1);
    start_time = t1.tv_sec * 1000000000 + t1.tv_nsec;
    // Perform copies in random order
    for (i = 0; i < num_chunks; i++)
    {
        unsigned long offset = chunk_order[i] * chunk_size;
        _avx_cpy(array2 + offset, array1 + offset, chunk_size);
    }
    // End timing
    clock_gettime(CLOCK_REALTIME, &t1);
    end_time = t1.tv_sec * 1000000000 + t1.tv_nsec;
    ;

    // Calculate time taken and bandwidth
    avx_last_copy_time_ns = end_time - start_time;

    // Calculate bandwidth in MB/s
    // total_size in bytes / time in seconds = bytes per second
    // Convert to MB/s by dividing by 1024*1024
    avx_last_bandwidth_mbps = (unsigned long)total_size * 1000000000ULL / avx_last_copy_time_ns;
    avx_last_bandwidth_mbps = avx_last_bandwidth_mbps / (1024 * 1024);

    free(chunk_order);
    if (verify_copy() != true)
    {
        printf("Random copy verification failed  ns\n");
        // avx_last_bandwidth_mbps = 99999999999;
    }
    inprogress = false;
    printf("Copy_result \t_avx_cpy\t Chunk_size %llu KB\t Time: %llu ms\t Bandwidth: %llu MB/s\n", k_kb, avx_last_copy_time_ns / 1000000, avx_last_bandwidth_mbps);
}

static void perform_random_copy_avx_async_cpy(void)
{
    unsigned long total_size = GB_TO_BYTES(n_gb);
    unsigned long chunk_size = KB_TO_BYTES(k_kb);
    unsigned long num_chunks = total_size / chunk_size;
    unsigned long *chunk_order;
    unsigned long i;
    struct timespec t1;
    unsigned long start_time, end_time;

    printf("Random copy started \n");

    inprogress = true;

    // Allocate array for random chunk order
    chunk_order = (unsigned long *)malloc(sizeof(unsigned long) * num_chunks);
    if (!chunk_order)
    {
        printf("Failed to allocate chunk order array\n");
        return;
    }

    // Initialize chunk order
    for (i = 0; i < num_chunks; i++)
    {
        chunk_order[i] = i;
    }

    // Shuffle chunk order
    for (i = num_chunks - 1; i > 0; i--)
    {
        unsigned long j = rand() % (i + 1);
        unsigned long temp = chunk_order[i];
        chunk_order[i] = chunk_order[j];
        chunk_order[j] = temp;
    }

    // Start timing
    clock_gettime(CLOCK_REALTIME, &t1);
    start_time = t1.tv_sec * 1000000000 + t1.tv_nsec;
    // Perform copies in random order
    for (i = 0; i < num_chunks; i++)
    {
        unsigned long offset = chunk_order[i] * chunk_size;
        _avx_async_cpy(array2 + offset, array1 + offset, chunk_size);
    }
    // End timing
    clock_gettime(CLOCK_REALTIME, &t1);
    end_time = t1.tv_sec * 1000000000 + t1.tv_nsec;
    ;

    // Calculate time taken and bandwidth
    avx_last_copy_time_ns = end_time - start_time;

    // Calculate bandwidth in MB/s
    // total_size in bytes / time in seconds = bytes per second
    // Convert to MB/s by dividing by 1024*1024
    avx_last_bandwidth_mbps = (unsigned long)total_size * 1000000000ULL / avx_last_copy_time_ns;
    avx_last_bandwidth_mbps = avx_last_bandwidth_mbps / (1024 * 1024);

    free(chunk_order);
    if (verify_copy() != true)
    {
        printf("Random copy verification failed  ns\n");
        // avx_last_bandwidth_mbps = 99999999999;
    }
    inprogress = false;
    printf("Copy_result \t_avx_async_cpy\t Chunk_size %llu KB\t Time: %llu ms\t Bandwidth: %llu MB/s\n", k_kb, avx_last_copy_time_ns / 1000000, avx_last_bandwidth_mbps);
}

static void perform_random_copy_avx_async_pf_cpy(void)
{
    unsigned long total_size = GB_TO_BYTES(n_gb);
    unsigned long chunk_size = KB_TO_BYTES(k_kb);
    unsigned long num_chunks = total_size / chunk_size;
    unsigned long *chunk_order;
    unsigned long i;
    struct timespec t1;
    unsigned long start_time, end_time;

    printf("Random copy started \n");

    inprogress = true;

    // Allocate array for random chunk order
    chunk_order = (unsigned long *)malloc(sizeof(unsigned long) * num_chunks);
    if (!chunk_order)
    {
        printf("Failed to allocate chunk order array\n");
        return;
    }

    // Initialize chunk order
    for (i = 0; i < num_chunks; i++)
    {
        chunk_order[i] = i;
    }

    // Shuffle chunk order
    for (i = num_chunks - 1; i > 0; i--)
    {
        unsigned long j = rand() % (i + 1);
        unsigned long temp = chunk_order[i];
        chunk_order[i] = chunk_order[j];
        chunk_order[j] = temp;
    }

    // Start timing
    clock_gettime(CLOCK_REALTIME, &t1);
    start_time = t1.tv_sec * 1000000000 + t1.tv_nsec;
    // Perform copies in random order
    for (i = 0; i < num_chunks; i++)
    {
        unsigned long offset = chunk_order[i] * chunk_size;
        _avx_async_pf_cpy(array2 + offset, array1 + offset, chunk_size);
    }
    // End timing
    clock_gettime(CLOCK_REALTIME, &t1);
    end_time = t1.tv_sec * 1000000000 + t1.tv_nsec;
    ;

    // Calculate time taken and bandwidth
    avx_last_copy_time_ns = end_time - start_time;

    // Calculate bandwidth in MB/s
    // total_size in bytes / time in seconds = bytes per second
    // Convert to MB/s by dividing by 1024*1024
    avx_last_bandwidth_mbps = (unsigned long)total_size * 1000000000ULL / avx_last_copy_time_ns;
    avx_last_bandwidth_mbps = avx_last_bandwidth_mbps / (1024 * 1024);

    free(chunk_order);
    if (verify_copy() != true)
    {
        printf("Random copy verification failed  ns\n");
        // avx_last_bandwidth_mbps = 99999999999;
    }
    inprogress = false;
    printf("Copy_result \t_avx_async_pf_cpy\t Chunk_size %llu KB\t Time: %llu ms\t Bandwidth: %llu MB/s\n", k_kb, avx_last_copy_time_ns / 1000000, avx_last_bandwidth_mbps);
}

static void perform_random_copy_avx_cpy_unroll(void)
{
    unsigned long total_size = GB_TO_BYTES(n_gb);
    unsigned long chunk_size = KB_TO_BYTES(k_kb);
    unsigned long num_chunks = total_size / chunk_size;
    unsigned long *chunk_order;
    unsigned long i;
    struct timespec t1;
    unsigned long start_time, end_time;

    printf("Random copy started \n");

    inprogress = true;

    // Allocate array for random chunk order
    chunk_order = (unsigned long *)malloc(sizeof(unsigned long) * num_chunks);
    if (!chunk_order)
    {
        printf("Failed to allocate chunk order array\n");
        return;
    }

    // Initialize chunk order
    for (i = 0; i < num_chunks; i++)
    {
        chunk_order[i] = i;
    }

    // Shuffle chunk order
    for (i = num_chunks - 1; i > 0; i--)
    {
        unsigned long j = rand() % (i + 1);
        unsigned long temp = chunk_order[i];
        chunk_order[i] = chunk_order[j];
        chunk_order[j] = temp;
    }

    // Start timing
    clock_gettime(CLOCK_REALTIME, &t1);
    start_time = t1.tv_sec * 1000000000 + t1.tv_nsec;
    // Perform copies in random order
    for (i = 0; i < num_chunks; i++)
    {
        unsigned long offset = chunk_order[i] * chunk_size;
        _avx_cpy_unroll(array2 + offset, array1 + offset, chunk_size);
    }
    // End timing
    clock_gettime(CLOCK_REALTIME, &t1);
    end_time = t1.tv_sec * 1000000000 + t1.tv_nsec;
    ;

    // Calculate time taken and bandwidth
    avx_last_copy_time_ns = end_time - start_time;

    // Calculate bandwidth in MB/s
    // total_size in bytes / time in seconds = bytes per second
    // Convert to MB/s by dividing by 1024*1024
    avx_last_bandwidth_mbps = (unsigned long)total_size * 1000000000ULL / avx_last_copy_time_ns;
    avx_last_bandwidth_mbps = avx_last_bandwidth_mbps / (1024 * 1024);

    free(chunk_order);
    if (verify_copy() != true)
    {
        printf("Random copy verification failed  ns\n");
        // avx_last_bandwidth_mbps = 99999999999;
    }
    inprogress = false;
    printf("Copy_result \t_avx_cpy_unroll\t Chunk_size %llu KB\t Time: %llu ms\t Bandwidth: %llu MB/s\n", k_kb, avx_last_copy_time_ns / 1000000, avx_last_bandwidth_mbps);
}

static void perform_random_copy_avx_async_pf_cpy_unroll(void)
{
    unsigned long total_size = GB_TO_BYTES(n_gb);
    unsigned long chunk_size = KB_TO_BYTES(k_kb);
    unsigned long num_chunks = total_size / chunk_size;
    unsigned long *chunk_order;
    unsigned long i;
    struct timespec t1;
    unsigned long start_time, end_time;

    printf("Random copy started \n");

    inprogress = true;

    // Allocate array for random chunk order
    chunk_order = (unsigned long *)malloc(sizeof(unsigned long) * num_chunks);
    if (!chunk_order)
    {
        printf("Failed to allocate chunk order array\n");
        return;
    }

    // Initialize chunk order
    for (i = 0; i < num_chunks; i++)
    {
        chunk_order[i] = i;
    }

    // Shuffle chunk order
    for (i = num_chunks - 1; i > 0; i--)
    {
        unsigned long j = rand() % (i + 1);
        unsigned long temp = chunk_order[i];
        chunk_order[i] = chunk_order[j];
        chunk_order[j] = temp;
    }

    // Start timing
    clock_gettime(CLOCK_REALTIME, &t1);
    start_time = t1.tv_sec * 1000000000 + t1.tv_nsec;
    // Perform copies in random order
    for (i = 0; i < num_chunks; i++)
    {
        unsigned long offset = chunk_order[i] * chunk_size;
        _avx_async_pf_cpy_unroll(array2 + offset, array1 + offset, chunk_size);
    }
    // End timing
    clock_gettime(CLOCK_REALTIME, &t1);
    end_time = t1.tv_sec * 1000000000 + t1.tv_nsec;
    ;

    // Calculate time taken and bandwidth
    avx_last_copy_time_ns = end_time - start_time;

    // Calculate bandwidth in MB/s
    // total_size in bytes / time in seconds = bytes per second
    // Convert to MB/s by dividing by 1024*1024
    avx_last_bandwidth_mbps = (unsigned long)total_size * 1000000000ULL / avx_last_copy_time_ns;
    avx_last_bandwidth_mbps = avx_last_bandwidth_mbps / (1024 * 1024);

    free(chunk_order);
    if (verify_copy() != true)
    {
        printf("Random copy verification failed  ns\n");
        // avx_last_bandwidth_mbps = 99999999999;
    }
    inprogress = false;
    printf("Copy_result \t_avx_async_pf_cpy_unroll\t Chunk_size %llu KB\t Time: %llu ms\t Bandwidth: %llu MB/s\n", k_kb, avx_last_copy_time_ns / 1000000, avx_last_bandwidth_mbps);
}

static void perform_random_copy_memcpy(void)
{
    unsigned long total_size = GB_TO_BYTES(n_gb);
    unsigned long chunk_size = KB_TO_BYTES(k_kb);
    unsigned long num_chunks = total_size / chunk_size;
    unsigned long *chunk_order;

    unsigned long i;
    unsigned long start_time, end_time;
    struct timespec t1;

    inprogress = true;

    printf("Random copy string started \n");

    // Allocate array for random chunk order
    chunk_order = (unsigned long *)malloc(sizeof(unsigned long) * num_chunks);
    if (!chunk_order)
    {
        printf("Failed to allocate chunk order array\n");
        return;
    }

    // Initialize chunk order
    for (i = 0; i < num_chunks; i++)
    {
        chunk_order[i] = i;
    }

    // Shuffle chunk order
    for (i = num_chunks - 1; i > 0; i--)
    {
        unsigned long j = rand() % (i + 1);
        unsigned long temp = chunk_order[i];
        chunk_order[i] = chunk_order[j];
        chunk_order[j] = temp;
    }

    // Start timing
    clock_gettime(CLOCK_REALTIME, &t1);
    start_time = t1.tv_sec * 1000000000 + t1.tv_nsec;

    // Perform copies in random order
    for (i = 0; i < num_chunks; i++)
    {
        unsigned long offset = chunk_order[i] * chunk_size;
        memcpy(array2 + offset, array1 + offset, chunk_size);
    }

    // End timing
    clock_gettime(CLOCK_REALTIME, &t1);
    end_time = t1.tv_sec * 1000000000 + t1.tv_nsec;
    ;

    // Calculate time taken and bandwidth
    string_last_copy_time_ns = end_time - start_time;

    // Calculate bandwidth in MB/s
    // total_size in bytes / time in seconds = bytes per second
    // Convert to MB/s by dividing by 1024*1024
    string_last_bandwidth_mbps = (unsigned long)total_size * 1000000000ULL / string_last_copy_time_ns;
    string_last_bandwidth_mbps = string_last_bandwidth_mbps / (1024 * 1024);

    free(chunk_order);
    if (verify_copy() != true)
    {
        printf("Random copy verification failed  ns\n");
        // string_last_bandwidth_mbps = 99999999999;
    }
    inprogress = false;
    printf("Copy_result \tMEMCPY\t Chunk_size %llu KB\t Time: %llu ms\t Bandwidth: %llu MB/s\n", k_kb, string_last_copy_time_ns / 1000000, string_last_bandwidth_mbps);
}

static inline unsigned int
enqcmd(void *dst, const void *src)
{
    uint8_t retry;
    asm volatile(".byte 0xf2, 0x0f, 0x38, 0xf8, 0x02\t\n"
                 "setz %0\t\n"
                 : "=r"(retry) : "a"(dst), "d"(src));
    return (unsigned int)retry;
}

static void *map_dsa_device(const char *dsa_wq_path)
{
    void *dsa_device;
    int fd;

    fd = open(dsa_wq_path, O_RDWR);
    if (fd < 0)
    {
        printf("open %s failed with errno = %d.\n",
               dsa_wq_path, errno);
        return MAP_FAILED;
    }
    dsa_device = mmap(NULL, DSA_WQ_SIZE, PROT_WRITE,
                      MAP_SHARED | MAP_POPULATE, fd, 0);
    close(fd);
    if (dsa_device == MAP_FAILED)
    {
        printf("mmap failed with errno = %d.\n", errno);
        return MAP_FAILED;
    }
    return dsa_device;
}

static int submit_wi(void *wq, void *descriptor)
{
    int retry = 0;

    _mm_sfence();

    if (dedicated_mode)
    {
        _movdir64b(dsa_wq, descriptor);
    }
    else
    {
        while (1)
        {
            if (enqcmd(dsa_wq, descriptor) == 0)
            {
                break;
            }
            retry++;
            if (retry > max_retry_count)
            {
                printf("Submit work retry %d times.\n", retry);
                exit(1);
            }
        }
    }

    return 0;
}

static int poll_completion(struct dsa_completion_record *completion,
                           enum dsa_opcode opcode)
{
    int retry = 0;

    while (1)
    {
        if (completion->status != DSA_COMP_NONE)
        {
            /* TODO: Error handling here. */
            if (completion->status != DSA_COMP_SUCCESS &&
                completion->status != DSA_COMP_PAGE_FAULT_NOBOF)
            {
                printf("DSA opcode %d failed with status = %d.\n",
                       opcode, completion->status);
                return 1;
            }
            break;
        }
        retry++;
        if (retry > max_retry_count)
        {
            printf("Wait for completion retry %d times.\n", retry);
            return 1;
        }
        _mm_pause();
    }

    if (retry > top_retry_count)
    {
        top_retry_count = retry;
    }

    return 0;
}

static void copy_dsa(void *src, void *dst, size_t len)
{
    struct dsa_completion_record completion __attribute__((aligned(32)));
    struct dsa_hw_desc descriptor;
    uint8_t test_byte;

    memset(&completion, 0, sizeof(completion));
    memset(&descriptor, 0, sizeof(descriptor));

    descriptor.opcode = DSA_OPCODE_MEMMOVE;
    descriptor.flags = IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_CRAV;
    descriptor.xfer_size = len;
    descriptor.src_addr = (uintptr_t)src;
    descriptor.dst_addr = (uintptr_t)dst;
    completion.status = 0;
    descriptor.completion_addr = (uint64_t)&completion;

    // printf("Submitting work to DSA work queue.....\n");

    for (int i = 0; i < resubmit_copy_retry; i++)
    {
        submit_wi(dsa_wq, &descriptor);
        // printf("Polling for completion.....\n");
        poll_completion(&completion, DSA_OPCODE_COMPARE);

        if (completion.status == DSA_COMP_SUCCESS)
        {
            // printf("DSA operation completed!\n");
            return;
        }
    }

    printf("DSA COPY FAILED...\n");
    exit(1);
}

void dsa_cleanup(void)
{
    if (dsa_wq != MAP_FAILED)
    {
        munmap(dsa_wq, DSA_WQ_SIZE);
    }
}

static void perform_random_copy_dsa(void)
{
    unsigned long total_size = GB_TO_BYTES(n_gb);
    unsigned long chunk_size = KB_TO_BYTES(k_kb);
    unsigned long num_chunks = total_size / chunk_size;
    unsigned long *chunk_order;
    char *dsa_path = "/dev/dsa/wq0.1";

    unsigned long i;
    unsigned long start_time, end_time;
    struct timespec t1;

    inprogress = true;

    printf("Configuring DSA......\n");
    dsa_wq = map_dsa_device(dsa_path);
    if (dsa_wq == MAP_FAILED)
    {
        printf("map_dsa_device failed MAP_FAILED\n");
        return;
    }

    printf("Configured work queue.\n");

    printf("Random copy string started \n");

    // Allocate array for random chunk order
    chunk_order = (unsigned long *)malloc(sizeof(unsigned long) * num_chunks);
    if (!chunk_order)
    {
        printf("Failed to allocate chunk order array\n");
        return;
    }

    // Initialize chunk order
    for (i = 0; i < num_chunks; i++)
    {
        chunk_order[i] = i;
    }

    // Shuffle chunk order
    for (i = num_chunks - 1; i > 0; i--)
    {
        unsigned long j = rand() % (i + 1);
        unsigned long temp = chunk_order[i];
        chunk_order[i] = chunk_order[j];
        chunk_order[j] = temp;
    }

    // Start timing
    clock_gettime(CLOCK_REALTIME, &t1);
    start_time = t1.tv_sec * 1000000000 + t1.tv_nsec;

    // Perform copies in random order
    for (i = 0; i < num_chunks; i++)
    {
        unsigned long offset = chunk_order[i] * chunk_size;
        copy_dsa(array2 + offset, array1 + offset, chunk_size);
    }

    // End timing
    clock_gettime(CLOCK_REALTIME, &t1);
    end_time = t1.tv_sec * 1000000000 + t1.tv_nsec;
    ;

    // Calculate time taken and bandwidth
    string_last_copy_time_ns = end_time - start_time;

    // Calculate bandwidth in MB/s
    // total_size in bytes / time in seconds = bytes per second
    // Convert to MB/s by dividing by 1024*1024
    string_last_bandwidth_mbps = (unsigned long)total_size * 1000000000ULL / string_last_copy_time_ns;
    string_last_bandwidth_mbps = string_last_bandwidth_mbps / (1024 * 1024);

    free(chunk_order);
    if (verify_copy() != true)
    {
        printf("Random copy verification failed  ns\n");
        // string_last_bandwidth_mbps = 99999999999;
    }
    inprogress = false;
    printf("Copy_result \tDSA\t Chunk_size %llu KB\t Time: %llu ms\t Bandwidth: %llu MB/s\n", k_kb, string_last_copy_time_ns / 1000000, string_last_bandwidth_mbps);
}

int main(void)
{
    unsigned long chunk_size_kb = 1;
    printf("copy dsa");
    for (chunk_size_kb = 1; chunk_size_kb < 4096; chunk_size_kb *= 2)
    {
        k_kb = chunk_size_kb;
        if (allocate_and_initialize_arrays() == 0)
        {
            perform_random_copy_dsa();
        }
    }
    printf("copy rte_avx");
    for (chunk_size_kb = 1; chunk_size_kb < 512; chunk_size_kb *= 2)
    {
        k_kb = chunk_size_kb;
        if (allocate_and_initialize_arrays() == 0)
        {
            perform_random_copy_avx();
        }
    }
    printf("copy string");
    for (chunk_size_kb = 1; chunk_size_kb < 512; chunk_size_kb *= 2)
    {
        k_kb = chunk_size_kb;
        if (allocate_and_initialize_arrays() == 0)
        {
            perform_random_copy_memcpy();
        }
    }
    printf("copy rep_movsb");
    for (chunk_size_kb = 1; chunk_size_kb < 512; chunk_size_kb *= 2)
    {
        k_kb = chunk_size_kb;
        if (allocate_and_initialize_arrays() == 0)
        {
            perform_random_copy_rep_movsb();
        }
    }
    for (chunk_size_kb = 1; chunk_size_kb < 512; chunk_size_kb *= 2)
    {
        k_kb = chunk_size_kb;
        if (allocate_and_initialize_arrays() == 0)
        {
            perform_random_copy_avx_cpy();
        }
    }
    for (chunk_size_kb = 1; chunk_size_kb < 512; chunk_size_kb *= 2)
    {
        k_kb = chunk_size_kb;
        if (allocate_and_initialize_arrays() == 0)
        {
            perform_random_copy_avx_async_cpy();
        }
    }
    for (chunk_size_kb = 1; chunk_size_kb < 512; chunk_size_kb *= 2)
    {
        k_kb = chunk_size_kb;
        if (allocate_and_initialize_arrays() == 0)
        {
            perform_random_copy_avx_async_pf_cpy();
        }
    }
    for (chunk_size_kb = 1; chunk_size_kb < 512; chunk_size_kb *= 2)
    {
        k_kb = chunk_size_kb;
        if (allocate_and_initialize_arrays() == 0)
        {
            perform_random_copy_avx_cpy_unroll();
        }
    }
    for (chunk_size_kb = 1; chunk_size_kb < 512; chunk_size_kb *= 2)
    {
        k_kb = chunk_size_kb;
        if (allocate_and_initialize_arrays() == 0)
        {
            perform_random_copy_avx_cpy_unroll();
        }
    }
    for (chunk_size_kb = 1; chunk_size_kb < 512; chunk_size_kb *= 2)
    {
        k_kb = chunk_size_kb;
        if (allocate_and_initialize_arrays() == 0)
        {
            perform_random_copy_avx_async_pf_cpy_unroll();
        }
    }

    printf("Memory copy suit finished\n");
    return 0;
}
