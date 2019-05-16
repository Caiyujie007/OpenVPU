#ifndef __FPGA_DDR__
#define __FPGA_DDR__

#include <stdio.h>
#include <vector>
#include <stdint.h>
#include "vpu_basic.h"

#define FPGA_DDR_BASE_ADDRESS 0x00000000
#define FPGA_DDR_SIZE         0x100000000
#define MIN_BLOCK_SIZE        (Tk*2)

typedef struct{
unsigned char available;         /* whether block is avaiable */
uint64_t blocksize;          /* block size */
uint64_t board_DDR_address;  /* the address of DDR on FPGA board */
}mem_control_block;

using namespace std;

#define FPGA_NULL ((void *)0xFFFFFFFF)

void Debug_mcb();
void *FPGA_DDR_malloc(unsigned int numbytes);
void FPGA_DDR_free(void *firstbyte);

#endif
