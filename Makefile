obj-m += dma_mod.o

all: user module

module:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

user: copy_user.c
	g++ -march=native --static -o copy_user copy_user.c

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean