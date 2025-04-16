#include "x86intrin.h"
#include <sys/mman.h>
#include <linux/idxd.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>

static void *dsa_wq = MAP_FAILED;
static int dedicated_mode = 0;
static int max_retry_count = 1000000;
static int resubmit_copy_retry = 8;
static int top_retry_count;

#define DSA_WQ_SIZE 4096

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

static void* copy_dsa(void *dst, const void *src, size_t len)
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
            return dst;
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