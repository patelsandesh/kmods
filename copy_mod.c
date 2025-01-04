#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/slab.h>
#include <linux/random.h>
#include <linux/uaccess.h>
#include <asm/uaccess.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Sandesh");
MODULE_DESCRIPTION("Memory allocation and copy testing module");

static struct proc_dir_entry *proc_entry;
static unsigned long n_gb = 1;  // Default 1 GB
static unsigned long m_mb = 64; // Default 64 MB
static void *array1;
static void *array2;
static bool arrays_allocated = false;
static u64 last_copy_time_ns;   // Store last copy time in nanoseconds
static u64 last_bandwidth_mbps; // Store last bandwidth in MB/s
static bool inprogress = false;

#define GB_TO_BYTES(x) ((unsigned long)(x) << 30)
#define MB_TO_BYTES(x) ((unsigned long)(x) << 20)

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
    unsigned long i;

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

static void perform_random_copy(void)
{
    unsigned long total_size = GB_TO_BYTES(n_gb);
    unsigned long chunk_size = MB_TO_BYTES(m_mb);
    unsigned long num_chunks = total_size / chunk_size;
    unsigned long *chunk_order;
    unsigned long i;
    u64 start_time, end_time;

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
        unsigned long j = get_random_int() % (i + 1);
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
    last_copy_time_ns = end_time - start_time;

    // Calculate bandwidth in MB/s
    // total_size in bytes / time in seconds = bytes per second
    // Convert to MB/s by dividing by 1024*1024
    last_bandwidth_mbps = (u64)total_size * 1000000000ULL / last_copy_time_ns;
    last_bandwidth_mbps = last_bandwidth_mbps / (1024 * 1024);

    vfree(chunk_order);
    inprogress = false;
    pr_info("Random copy completed in %llu ns\n", last_copy_time_ns);
    pr_info("Bandwidth: %llu MB/s\n", last_bandwidth_mbps);
}

static ssize_t module_write(struct file *file, const char __user *buffer,
                            size_t count, loff_t *data)
{
    char kbuf[32];
    unsigned long new_n, new_m;

    if (count > sizeof(kbuf) - 1)
        return -EINVAL;

    if (copy_from_user(kbuf, buffer, count))
        return -EFAULT;

    kbuf[count] = '\0';

    if (sscanf(kbuf, "%lu %lu", &new_n, &new_m) != 2)
        return -EINVAL;

    if (new_n == 0 || new_m == 0)
        return -EINVAL;

    if (new_m > new_n * 1024) // m_mb shouldn't be larger than n_gb in MB
        return -EINVAL;

    n_gb = new_n;
    m_mb = new_m;

    if (allocate_and_initialize_arrays() == 0)
    {
        perform_random_copy();
    }

    return count;
}

static int module_show(struct seq_file *m, void *v)
{
    seq_printf(m, "Current settings:\n");
    seq_printf(m, "Array size (n): %lu GB\n", n_gb);
    seq_printf(m, "Chunk size (m): %lu MB\n", m_mb);
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
    snprintf(greeting, greeting_length, "In progress: %s, Time ms %lu, Bandwidth MBps %lu\n", inprogress ? "yes" : "no", last_copy_time_ns / 1000000, last_bandwidth_mbps);
    copy_to_user(user_buffer, greeting, greeting_length);
    *offset = greeting_length;
    return greeting_length;
}

static struct file_operations proc_ops = {
    .open = module_open,
    .read = custom_read,
    .write = module_write,
};

static int __init copy_mod_init(void)
{
    proc_entry = proc_create("memory_copy", 0666, NULL, &proc_ops);
    if (!proc_entry)
    {
        pr_err("Failed to create proc entry\n");
        return -ENOMEM;
    }
    if (allocate_and_initialize_arrays() == 0)
    {
        perform_random_copy();
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