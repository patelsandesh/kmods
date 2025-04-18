#include <linux/err.h>
#include <linux/delay.h>
#include <linux/dma-mapping.h>
#include <linux/dmaengine.h>
#include <linux/freezer.h>
#include <linux/init.h>
#include <linux/kthread.h>
#include <linux/sched/task.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/random.h>
#include <linux/slab.h>
#include <linux/wait.h>

#define TRANSFER_SIZE 512 /* Size of data to transfer in bytes */

/* Module metadata */
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Claude");
MODULE_DESCRIPTION("DMA Transfer Module");
MODULE_VERSION("1.0");

static struct device *dummy_device;
static dma_addr_t src_dma_addr;
static dma_addr_t dst_dma_addr;
static void *src_cpu_addr;
static void *dst_cpu_addr;
static struct dma_chan *dma_channel;

static int __init dma_transfer_init(void)
{
    int ret = 0;
    struct dma_device *dma_dev;
    struct device *dev;
    dma_cap_mask_t mask;
    struct dma_async_tx_descriptor *tx;
    dma_cookie_t cookie;

    printk(KERN_INFO "DMA Transfer Module: Initializing\n");

    /* Create a dummy device for DMA operations */
    dummy_device = kzalloc(sizeof(struct device), GFP_KERNEL);
    if (!dummy_device)
    {
        printk(KERN_ERR "DMA Transfer Module: Failed to allocate dummy device\n");
        return -ENOMEM;
    }

    device_initialize(dummy_device);
    dev_set_name(dummy_device, "dma_dummy_device");

    /* Allocate and initialize source and destination buffers */
    src_cpu_addr = kmalloc(TRANSFER_SIZE, GFP_KERNEL);
    if (!src_cpu_addr)
    {
        printk(KERN_ERR "DMA Transfer Module: Failed to allocate source buffer\n");
        ret = -ENOMEM;
        goto free_device;
    }

    dst_cpu_addr = kmalloc(TRANSFER_SIZE, GFP_KERNEL);
    ;
    if (!dst_cpu_addr)
    {
        printk(KERN_ERR "DMA Transfer Module: Failed to allocate destination buffer\n");
        ret = -ENOMEM;
        goto free_src;
    }

    /* Fill source buffer with sample data */
    memset(src_cpu_addr, 0xAB, TRANSFER_SIZE);
    memset(dst_cpu_addr, 0, TRANSFER_SIZE);

    /* Get a DMA channel */
    dma_cap_zero(mask);
    dma_cap_set(DMA_MEMCPY, mask);

    dma_channel = dma_request_channel(mask, NULL, NULL);
    if (!dma_channel)
    {
        printk(KERN_ERR "DMA Transfer Module: No DMA channel available\n");
        ret = -ENODEV;
        goto free_dst;
    }

    dev = dmaengine_get_dma_device(dma_channel);
    dma_dev = dma_channel->device;

    struct page *pg = virt_to_page(src_cpu_addr);
    unsigned long pg_off = offset_in_page(src_cpu_addr);
    src_dma_addr = dma_map_page(dev, pg, pg_off, TRANSFER_SIZE, DMA_BIDIRECTIONAL);

    pg = virt_to_page(dst_cpu_addr);
    pg_off = offset_in_page(dst_cpu_addr);
    dst_dma_addr = dma_map_page(dev, pg, pg_off, TRANSFER_SIZE, DMA_BIDIRECTIONAL);

    /* Prepare DMA transfer */
    tx = dma_dev->device_prep_dma_memcpy(dma_channel, dst_dma_addr, src_dma_addr,
                                         TRANSFER_SIZE, DMA_PREP_INTERRUPT);
    if (!tx)
    {
        printk(KERN_ERR "DMA Transfer Module: Failed to prepare DMA transfer\n");
        ret = -EINVAL;
        goto free_channel;
    }

    /* Set completion callback */
    tx->callback = NULL; /* No callback for simplicity */

    /* Submit transaction */
    cookie = tx->tx_submit(tx);
    if (dma_submit_error(cookie))
    {
        printk(KERN_ERR "DMA Transfer Module: Failed to submit DMA transaction\n");
        ret = -EINVAL;
        goto free_channel;
    }

    int status = dma_sync_wait(dma_channel, cookie);
    dmaengine_terminate_sync(dma_channel);
    if (status == DMA_ERROR)
    {
        printk(KERN_INFO "DMA Transfer Module: DMA submit failed!\n");
    }

    /* Verify the transfer */
    if (memcmp(src_cpu_addr, dst_cpu_addr, TRANSFER_SIZE) == 0)
    {
        printk(KERN_INFO "DMA Transfer Module: DMA transfer successful!\n");
    }
    else
    {
        printk(KERN_WARNING "DMA Transfer Module: DMA transfer verification failed!\n");
    }

    return 0;

free_channel:
    dma_release_channel(dma_channel);
free_dst:
    dma_free_coherent(dummy_device, TRANSFER_SIZE, dst_cpu_addr, dst_dma_addr);
free_src:
    dma_free_coherent(dummy_device, TRANSFER_SIZE, src_cpu_addr, src_dma_addr);
free_device:
    kfree(dummy_device);

    return ret;
}

static void __exit dma_transfer_exit(void)
{
    printk(KERN_INFO "DMA Transfer Module: Cleaning up\n");

    /* Release DMA channel if it was acquired */
    if (dma_channel)
    {
        dma_release_channel(dma_channel);
    }

    /* Free the DMA buffers */
    if (src_cpu_addr)
    {
        dma_free_coherent(dummy_device, TRANSFER_SIZE, src_cpu_addr, src_dma_addr);
    }

    if (dst_cpu_addr)
    {
        dma_free_coherent(dummy_device, TRANSFER_SIZE, dst_cpu_addr, dst_dma_addr);
    }

    /* Free the dummy device */
    if (dummy_device)
    {
        kfree(dummy_device);
    }

    printk(KERN_INFO "DMA Transfer Module: Unloaded\n");
}

module_init(dma_transfer_init);
module_exit(dma_transfer_exit);