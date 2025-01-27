#include <linux/compiler.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/slab.h>
#include <linux/random.h>
#include <linux/uaccess.h>
#include <asm/fpu/api.h>
#include <linux/mm.h>
#include <linux/vmalloc.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Sandesh");
MODULE_DESCRIPTION("Memory allocation and copy testing module");

static struct proc_dir_entry *proc_entry;
static unsigned long n_gb = 2;    // Default 1 GB
static unsigned long k_kb = 1024; // Default 4 MB
static void *array1;
static void *array2;
static bool arrays_allocated = false;
static u64 avx_last_copy_time_ns;   // Store last copy time in nanoseconds
static u64 avx_last_bandwidth_mbps; // Store last bandwidth in MB/s

static u64 string_last_copy_time_ns;   // Store last copy time in nanoseconds
static u64 string_last_bandwidth_mbps; // Store last bandwidth in MB/s

static bool inprogress = false;
static bool verified = true;

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

static inline void *
rte_mov15_or_less(void *dst, const void *src, size_t n)
{
    /**
     * Use the following structs to avoid violating C standard
     * alignment requirements and to avoid strict aliasing bugs
     */
    copy_user_enhanced_fast_string(dst, src, n);
    return dst;
}

/**
 * Copy 16 bytes from one location to another,
 * locations should not overlap.
 */
static inline void
rte_mov16(uint8_t *dst, const uint8_t *src)
{
    copy_user_enhanced_fast_string(dst, src, 16);
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
    asm volatile("vmovdqu64 %1,%%zmm1\n\t"
                 "vmovdqu64 %%zmm1,%0\n\t"

                 : "=m"(dst)
                 : "m"(*src));
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
    while (n >= 128)
    {
        asm volatile("vmovdqa64 %2,%%zmm0\n\t"
                     "vmovdqa64 %3,%%zmm1\n\t"
                     "vmovdqa64 %%zmm0,%0\n\t"
                     "vmovdqa64 %%zmm1,%1\n\t"
                     : "=m"((dst[0 * 64])), "=m"((dst[1 * 64]))
                     : "m"(*(src + 0 * 64)), "m"(*(src + 1 * 64)));
        n -= 128;
        src = src + 128;
        dst = dst + 128;
    }
}

/**
 * Copy 512-byte blocks from one location to another,
 * locations should not overlap.
 */
static inline void
rte_mov512blocks(uint8_t *dst, const uint8_t *src, size_t n)
{

    while (n >= 512)
    {
        asm volatile("vmovdqa64 %8,%%zmm0\n\t"
                     "vmovdqa64 %9,%%zmm1\n\t"
                     "vmovdqa64 %10,%%zmm2\n\t"
                     "vmovdqa64 %11,%%zmm3\n\t"
                     "vmovdqa64 %12,%%zmm4\n\t"
                     "vmovdqa64 %13,%%zmm5\n\t"
                     "vmovdqa64 %14,%%zmm6\n\t"
                     "vmovdqa64 %15,%%zmm7\n\t"

                     "vmovdqa64 %%zmm0,%0\n\t"
                     "vmovdqa64 %%zmm1,%1\n\t"
                     "vmovdqa64 %%zmm2,%2\n\t"
                     "vmovdqa64 %%zmm3,%3\n\t"
                     "vmovdqa64 %%zmm4,%4\n\t"
                     "vmovdqa64 %%zmm5,%5\n\t"
                     "vmovdqa64 %%zmm6,%6\n\t"
                     "vmovdqa64 %%zmm7,%7\n\t"

                     : "=m"(dst[0 * 64]), "=m"(dst[1 * 64]),
                       "=m"(dst[2 * 64]), "=m"(dst[3 * 64]),
                       "=m"(dst[4 * 64]), "=m"(dst[5 * 64]),
                       "=m"(dst[6 * 64]), "=m"(dst[7 * 64])
                     : "m"(*(src + 0 * 64)), "m"(*(src + 1 * 64)),
                       "m"(*(src + 2 * 64)), "m"(*(src + 3 * 64)),
                       "m"(*(src + 4 * 64)), "m"(*(src + 5 * 64)),
                       "m"(*(src + 6 * 64)), "m"(*(src + 7 * 64)));
        n -= 512;
        src = src + 512;
        dst = dst + 512;
    }
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
    rte_mov512blocks((uint8_t *)dst, (const uint8_t *)src, n);
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

static void cleanup_arrays(void)
{
    if (arrays_allocated)
    {
        vfree(array1);
        vfree(array2);
        arrays_allocated = false;
        pr_info("Arrays freed\n");
    }
}

static int allocate_and_initialize_arrays(void)
{
    unsigned long size = GB_TO_BYTES(n_gb);

    cleanup_arrays(); // Clean up any existing arrays

    array1 = vmalloc(size);
    if (!array1)
    {
        pr_err("Failed to allocate array1\n");
        return -ENOMEM;
    }

    array2 = vmalloc(size);
    if (!array2)
    {
        pr_err("Failed to allocate array2\n");
        vfree(array1);
        return -ENOMEM;
    }

    // Initialize array1 with 1s
    memset(array1, 1, size);

    // Initialize array2 with 2s
    memset(array2, 2, size);

    arrays_allocated = true;
    pr_info("Arrays allocated and initialized: %lu GB each\n", n_gb);
    return 0;
}

static int verify_copy(void)
{
    unsigned long size = GB_TO_BYTES(n_gb);
    unsigned long i;

    if (verified != true)
    {
        return false;
    }

    for (i = 0; i < size; i++)
    {
        if (((char *)array1)[i] != ((char *)array2)[i])
        {
            pr_err("Verification failed at offset %lu, total size %lu, a1 %d a2 %d\n", i, size, ((char *)array1)[i], ((char *)array2)[i]);
            verified = false;
            return verified;
        }
    }

    pr_info("Verification successful\n");
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
    u64 start_time, end_time;

    pr_info("Random copy avx started \n");

    if (!boot_cpu_has(X86_FEATURE_AVX512F) || !boot_cpu_has(X86_FEATURE_AVX))
        return;

    inprogress = true;

    // Allocate array for random chunk order
    chunk_order = vmalloc(sizeof(unsigned long) * num_chunks);
    if (!chunk_order)
    {
        pr_err("Failed to allocate chunk order array\n");
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
        unsigned long j = get_random_u64() % (i + 1);
        unsigned long temp = chunk_order[i];
        chunk_order[i] = chunk_order[j];
        chunk_order[j] = temp;
    }

    // Start timing
    start_time = ktime_get_ns();
    // Perform copies in random order
    for (i = 0; i < num_chunks; i++)
    {
        unsigned long offset = chunk_order[i] * chunk_size;
        kernel_fpu_begin();
        rte_memcpy(array2 + offset, array1 + offset, chunk_size);
        kernel_fpu_end();
        pr_debug("Copied chunk %lu/%lu\n", i + 1, num_chunks);
    }
    // End timing
    end_time = ktime_get_ns();

    // Calculate time taken and bandwidth
    avx_last_copy_time_ns = end_time - start_time;

    // Calculate bandwidth in MB/s
    // total_size in bytes / time in seconds = bytes per second
    // Convert to MB/s by dividing by 1024*1024
    avx_last_bandwidth_mbps = (u64)total_size * 1000000000ULL / avx_last_copy_time_ns;
    avx_last_bandwidth_mbps = avx_last_bandwidth_mbps / (1024 * 1024);

    vfree(chunk_order);
    if (verify_copy() != true)
    {
        pr_info("Random copy verification failed  ns\n");
        avx_last_bandwidth_mbps = 99999999999;
    }
    inprogress = false;
    pr_info("Random copy completed in %llu ns\n", avx_last_copy_time_ns);
    pr_info("Bandwidth: %llu MB/s\n", avx_last_bandwidth_mbps);
}

static void perform_random_copy_string(void)
{
    unsigned long total_size = GB_TO_BYTES(n_gb);
    unsigned long chunk_size = KB_TO_BYTES(k_kb);
    unsigned long num_chunks = total_size / chunk_size;
    unsigned long *chunk_order;

    unsigned long i;
    u64 start_time, end_time;

    inprogress = true;

    pr_info("Random copy string started \n");

    // Allocate array for random chunk order
    chunk_order = vmalloc(sizeof(unsigned long) * num_chunks);
    if (!chunk_order)
    {
        pr_err("Failed to allocate chunk order array\n");
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
        unsigned long j = get_random_u64() % (i + 1);
        unsigned long temp = chunk_order[i];
        chunk_order[i] = chunk_order[j];
        chunk_order[j] = temp;
    }

    // Start timing
    start_time = ktime_get_ns();

    // Perform copies in random order
    for (i = 0; i < num_chunks; i++)
    {
        unsigned long offset = chunk_order[i] * chunk_size;
        copy_user_enhanced_fast_string(array2 + offset, array1 + offset, chunk_size);
        pr_debug("Copied chunk %lu/%lu\n", i + 1, num_chunks);
    }

    // End timing
    end_time = ktime_get_ns();

    // Calculate time taken and bandwidth
    string_last_copy_time_ns = end_time - start_time;

    // Calculate bandwidth in MB/s
    // total_size in bytes / time in seconds = bytes per second
    // Convert to MB/s by dividing by 1024*1024
    string_last_bandwidth_mbps = (u64)total_size * 1000000000ULL / string_last_copy_time_ns;
    string_last_bandwidth_mbps = string_last_bandwidth_mbps / (1024 * 1024);

    vfree(chunk_order);
    if (verify_copy() != true)
    {
        pr_info("Random copy verification failed  ns\n");
        string_last_bandwidth_mbps = 99999999999;
    }
    inprogress = false;
    pr_info("Random copy completed in %llu ns\n", string_last_copy_time_ns);
    pr_info("Bandwidth: %llu MB/s\n", string_last_bandwidth_mbps);
}

static ssize_t module_write(struct file *file, const char __user *buffer,
                            size_t count, loff_t *data)
{
    char kbuf[32];
    unsigned long new_n, new_k;

    if (count > sizeof(kbuf) - 1)
        return -EINVAL;

    if (copy_from_user(kbuf, buffer, count))
        return -EFAULT;

    kbuf[count] = '\0';

    if (sscanf(kbuf, "%lu %lu", &new_n, &new_k) != 2)
        return -EINVAL;

    if (new_n == 0 || new_k == 0)
        return -EINVAL;

    if (new_k > new_n * 1024 * 1024) // k_kb shouldn't be larger than n_gb in KB
        return -EINVAL;

    n_gb = new_n;
    k_kb = new_k;

    if (allocate_and_initialize_arrays() == 0)
    {
        perform_random_copy_avx();
    }
    if (allocate_and_initialize_arrays() == 0)
    {
        perform_random_copy_string();
    }

    return count;
}

static int module_show(struct seq_file *m, void *v)
{
    seq_printf(m, "Current settings:\n");
    seq_printf(m, "Array size (n): %lu GB\n", n_gb);
    seq_printf(m, "Chunk size (m): %lu MB\n", k_kb);
    seq_printf(m, "Arrays allocated: %s\n", arrays_allocated ? "yes" : "no");
    return 0;
}

static int module_open(struct inode *inode, struct file *file)
{
    return single_open(file, module_show, NULL);
}

static ssize_t custom_read(struct file *file, char __user *user_buffer, size_t count, loff_t *offset)
{
    printk(KERN_INFO "calling our very own custom read method.");
    int greeting_length = 128;
    char *greeting = vmalloc(greeting_length);
    if (*offset > 0)
        return 0;
    snprintf(greeting, greeting_length, "In progress: %s, Verified: %s \n"
                                        "AVX: Time ms %llu, Bandwidth MBps %llu\n"
                                        "String: Time ms %llu, Bandwidth MBps %llu\n",
             inprogress ? "yes" : "no", verified ? "yes" : "no",
             avx_last_copy_time_ns / 1000000, avx_last_bandwidth_mbps,
             string_last_copy_time_ns / 1000000, string_last_bandwidth_mbps);
    copy_to_user(user_buffer, greeting, greeting_length);
    *offset = greeting_length;
    return greeting_length;
}

const struct proc_ops proc_ops = {
    .proc_open = module_open,
    .proc_read = custom_read,
    .proc_write = module_write,
};

static int __init copy_mod_init(void)
{
    pr_info("Copy_mod Memory copy module loading\n");
    proc_entry = proc_create("memory_copy", 0666, NULL, &proc_ops);
    if (!proc_entry)
    {
        pr_err("Failed to create proc entry\n");
        return -ENOMEM;
    }
    if (allocate_and_initialize_arrays() == 0)
    {
        perform_random_copy_avx();
    }
    if (allocate_and_initialize_arrays() == 0)
    {
        perform_random_copy_string();
    }

    pr_info("Memory copy module loaded\n");
    return 0;
}

static void __exit copy_mod_exit(void)
{
    cleanup_arrays();
    if (proc_entry)
    {
        proc_remove(proc_entry);
    }
    pr_info("Memory copy module unloaded\n");
}

module_init(copy_mod_init);
module_exit(copy_mod_exit);
