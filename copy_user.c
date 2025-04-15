#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <string.h>
#include <immintrin.h>
#include <sys/mman.h>

#include <cstdlib>
#include <thread>
#include <vector>

#include "rte_copy.h"
#include "avx_varients.h"
#include "dsa_copy.h"

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

#define GB_TO_BYTES(x) ((unsigned long)(x) << 30)
#define KB_TO_BYTES(x) ((unsigned long)(x) << 10)
#define ALIGNMENT_MASK 0x3F

#define COPY_USING(func)                               \
    do                                                 \
    {                                                  \
        printf("Copying using function: %s\n", #func); \
        copy_driver(func);                             \
    } while (0)

typedef void *(*copy_func_t)(void *dst, const void *src, long unsigned int n);

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

static void configure_dsa(void)
{
    char *dsa_path = "/dev/dsa/wq0.1";
    printf("Configuring DSA......\n");
    dsa_wq = map_dsa_device(dsa_path);
    if (dsa_wq == MAP_FAILED)
    {
        printf("map_dsa_device failed MAP_FAILED\n");
        return;
    }

    printf("Configured work queue.\n");
}

static void copy_driver(copy_func_t copy_func)
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
        copy_func(array2 + offset, array1 + offset, chunk_size);
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
    printf("Copy_result:\t Chunk_size %llu KB\t Time: %llu ms\t Bandwidth: %llu MB/s\n", k_kb, avx_last_copy_time_ns / 1000000, avx_last_bandwidth_mbps);
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
            configure_dsa();
            COPY_USING(copy_dsa);
        }
    }
    printf("copy rte_avx");
    for (chunk_size_kb = 1; chunk_size_kb < 512; chunk_size_kb *= 2)
    {
        k_kb = chunk_size_kb;
        if (allocate_and_initialize_arrays() == 0)
        {
            COPY_USING(rte_memcpy);
        }
    }
    printf("copy string");
    for (chunk_size_kb = 1; chunk_size_kb < 512; chunk_size_kb *= 2)
    {
        k_kb = chunk_size_kb;
        if (allocate_and_initialize_arrays() == 0)
        {
            COPY_USING(memcpy);
        }
    }
    for (chunk_size_kb = 1; chunk_size_kb < 512; chunk_size_kb *= 2)
    {
        k_kb = chunk_size_kb;
        if (allocate_and_initialize_arrays() == 0)
        {
            COPY_USING(_rep_movsb);
        }
    }
    for (chunk_size_kb = 1; chunk_size_kb < 512; chunk_size_kb *= 2)
    {
        k_kb = chunk_size_kb;
        if (allocate_and_initialize_arrays() == 0)
        {
            COPY_USING(_avx_cpy);
        }
    }
    for (chunk_size_kb = 1; chunk_size_kb < 512; chunk_size_kb *= 2)
    {
        k_kb = chunk_size_kb;
        if (allocate_and_initialize_arrays() == 0)
        {
            COPY_USING(_avx_async_cpy);
        }
    }
    for (chunk_size_kb = 1; chunk_size_kb < 512; chunk_size_kb *= 2)
    {
        k_kb = chunk_size_kb;
        if (allocate_and_initialize_arrays() == 0)
        {
            COPY_USING(_avx_async_pf_cpy);
        }
    }
    for (chunk_size_kb = 1; chunk_size_kb < 512; chunk_size_kb *= 2)
    {
        k_kb = chunk_size_kb;
        if (allocate_and_initialize_arrays() == 0)
        {
            COPY_USING(_avx_cpy_unroll);
        }
    }
    for (chunk_size_kb = 1; chunk_size_kb < 512; chunk_size_kb *= 2)
    {
        k_kb = chunk_size_kb;
        if (allocate_and_initialize_arrays() == 0)
        {
            COPY_USING(_avx_async_cpy_unroll);
        }
    }
    for (chunk_size_kb = 1; chunk_size_kb < 512; chunk_size_kb *= 2)
    {
        k_kb = chunk_size_kb;
        if (allocate_and_initialize_arrays() == 0)
        {
            COPY_USING(_avx_async_pf_cpy_unroll);
        }
    }

    printf("Memory copy suit finished\n");
    return 0;
}
