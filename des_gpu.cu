
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <strings.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define ERR(source) (perror(source), fprintf(stderr,"%s:%d\n",__FILE__,__LINE__), exit(EXIT_FAILURE))

#ifndef DES_CONSTANTS
#define DES_CONSTANTS

int PC_1[56] = {
	57,	49,	41,	33,	25,	17,	9,
	1,	58,	50,	42,	34,	26,	18,
	10,	2,	59,	51,	43,	35,	27,
	19,	11,	3,	60,	52,	44,	36,
	63,	55,	47,	39,	31,	23,	15,
	7,	62,	54,	46,	38,	30,	22,
	14,	6,	61,	53,	45,	37,	29,
	21,	13,	5,	28,	20,	12,	4
};

int PC_2[48] = {
	14, 17, 11, 24, 1,	5,
	3,	28, 15, 6,	21, 10,
	23, 19, 12, 4,	26, 8,
	16, 7,	27, 20, 13, 2,
	41, 52, 31, 37, 47, 55,
	30, 40, 51, 45, 33, 48,
	44, 49, 39, 56, 34, 53,
	46, 42, 50, 36, 29, 32
};

int IP[64] = {
	58,	50,	42,	34,	26,	18,	10,	2,
	60,	52,	44,	36,	28,	20,	12,	4,
	62,	54,	46,	38,	30,	22,	14,	6,
	64,	56,	48,	40,	32,	24,	16,	8,
	57,	49,	41,	33,	25,	17,	 9,	1,
	59,	51,	43,	35,	27,	19,	11,	3,
	61,	53,	45,	37,	29,	21,	13,	5,
	63,	55,	47,	39,	31,	23,	15,	7
};

int E_BIT[48] = {
	32,	1,	2,	3,	4,	5,
	4,	5,	6,	7,	8,	9,
	8,	9,	10,	11,	12,	13,
	12,	13,	14,	15,	16,	17,
	16,	17,	18,	19,	20,	21,
	20,	21,	22,	23,	24,	25,
	24,	25,	26,	27,	28,	29,
	28,	29,	30,	31,	32,	1
};

int S1[64] = {
	14,	4,	13,	1,	2,	15,	11,	8,	3,	10,	6,	12,	5,	9,	0,	7,
	0,	15,	7,	4,	14,	2,	13,	1,	10,	6,	12,	11,	9,	5,	3,	8,
	4,	1,	14,	8,	13,	6,	2,	11,	15,	12,	9,	7,	3,	10,	5,	0,
	15,	12,	8,	2,	4,	9,	1,	7,	5,	11,	3,	14,	10,	0,	6,	13
};

int S2[64] = {
	15,	1,	8,	14,	6,	11,	3,	4,	9,	7,	2,	13,	12,	0,	5,	10,
	3,	13,	4,	7,	15,	2,	8,	14,	12,	0,	1,	10,	6,	9,	11,	5,
	0,	14,	7,	11,	10,	4,	13,	1,	5,	8,	12,	6,	9,	3,	2,	15,
	13,	8,	10,	1,	3,	15,	4,	2,	11,	6,	7,	12,	0,	5,	14,	9,
};

int S3[64] = {
	10,	0,	9,	14,	6,	3,	15,	5,	1,	13,	12,	7,	11,	4,	2,	8,
	13,	7,	0,	9,	3,	4,	6,	10,	2,	8,	5,	14,	12,	11,	15,	1,
	13,	6,	4,	9,	8,	15,	3,	0,	11,	1,	2,	12,	5,	10,	14,	7,
	1,	10,	13,	0,	6,	9,	8,	7,	4,	15,	14,	3,	11,	5,	2,	12
};

int S4[64] = {
	7,	13,	14,	3,	0,	6,	9,	10,	1,	2,	8,	5,	11,	12,	4,	15,
	13,	8,	11,	5,	6,	15,	0,	3,	4,	7,	2,	12,	1,	10,	14,	9,
	10,	6,	9,	0,	12,	11,	7,	13,	15,	1,	3,	14,	5,	2,	8,	4,
	3,	15,	0,	6,	10,	1,	13,	8,	9,	4,	5,	11,	12,	7,	2,	14
};

int S5[64] = {
	2,	12,	4,	1,	7,	10,	11,	6,	8,	5,	3,	15,	13,	0,	14,	9,
	14,	11,	2,	12,	4,	7,	13,	1,	5,	0,	15,	10,	3,	9,	8,	6,
	4,	2,	1,	11,	10,	13,	7,	8,	15,	9,	12,	5,	6,	3,	0,	14,
	11,	8,	12,	7,	1,	14,	2,	13,	6,	15,	0,	9,	10,	4,	5,	3
};

int S6[64] = {
	12,	1,	10,	15,	9,	2,	6,	8,	0,	13,	3,	4,	14,	7,	5,	11,
	10,	15,	4,	2,	7,	12,	9,	5,	6,	1,	13,	14,	0,	11,	3,	8,
	9,	14,	15,	5,	2,	8,	12,	3,	7,	0,	4,	10,	1,	13,	11,	6,
	4,	3,	2,	12,	9,	5,	15,	10,	11,	14,	1,	7,	6,	0,	8,	13,
};

int S7[64] = {
	4,	11,	2,	14,	15,	0,	8,	13,	3,	12,	9,	7,	5,	10,	6,	1,
	13,	0,	11,	7,	4,	9,	1,	10,	14,	3,	5,	12,	2,	15,	8,	6,
	1,	4,	11,	13,	12,	3,	7,	14,	10,	15,	6,	8,	0,	5,	9,	2,
	6,	11,	13,	8,	1,	4,	10,	7,	9,	5,	0,	15,	14,	2,	3,	12,
};

int S8[64] = {
	13,	2,	8,	4,	6,	15,	11,	1,	10,	9,	3,	14,	5,	0,	12,	7,
	1,	15,	13,	8,	10,	3,	7,	4,	12,	5,	6,	11,	0,	14,	9,	2,
	7,	11,	4,	1,	9,	12,	14,	2,	0,	6,	10,	13,	15,	3,	5,	8,
	2,	1,	14,	7,	4,	10,	8,	13,	15,	12,	9,	0,	3,	5,	6,	11,
};

int * S_POINTER[8] = {
	S1, S2, S3, S4, S5, S6, S7, S8
};

int P[32] = {
	16,	7,	20, 21,
	29,	12, 28, 17,
	1,	15, 23, 26,
	5,	18, 31, 10,
	2,	8,	24, 14,
	32, 27, 3,	9,
	19, 13, 30,	6,
	22, 11, 4,	25
};

int IP_REV[64] = {
	40,	8, 48, 16, 56, 24, 64, 32,
	39, 7, 47, 15, 55, 23, 63, 31,
	38, 6, 46, 14, 54, 22, 62, 30,
	37, 5, 45, 13, 53, 21, 61, 29,
	36, 4, 44, 12, 52, 20, 60, 28,
	35, 3, 43, 11, 51, 19, 59, 27,
	34, 2, 42, 10, 50, 18, 58, 26,
	33, 1, 41,	9, 49, 17, 57, 25
};

int SHIFTS[16] = {
	1,
	1,
	2,
	2,
	2,
	2,
	2,
	2,
	1,
	2,
	2,
	2,
	2,
	2,
	2,
	1
};

#endif

#ifndef DES_CUDA_CONSTANTS
#define DES_CUDA_CONSTANTS

__constant__ int PC_1_CUDA[56] = {
	57,	49,	41,	33,	25,	17,	9,
	1,	58,	50,	42,	34,	26,	18,
	10,	2,	59,	51,	43,	35,	27,
	19,	11,	3,	60,	52,	44,	36,
	63,	55,	47,	39,	31,	23,	15,
	7,	62,	54,	46,	38,	30,	22,
	14,	6,	61,	53,	45,	37,	29,
	21,	13,	5,	28,	20,	12,	4
};

__constant__ int PC_2_CUDA[48] = {
	14, 17, 11, 24, 1,	5,
	3,	28, 15, 6,	21, 10,
	23, 19, 12, 4,	26, 8,
	16, 7,	27, 20, 13, 2,
	41, 52, 31, 37, 47, 55,
	30, 40, 51, 45, 33, 48,
	44, 49, 39, 56, 34, 53,
	46, 42, 50, 36, 29, 32
};

__constant__ int IP_CUDA[64] = {
	58,	50,	42,	34,	26,	18,	10,	2,
	60,	52,	44,	36,	28,	20,	12,	4,
	62,	54,	46,	38,	30,	22,	14,	6,
	64,	56,	48,	40,	32,	24,	16,	8,
	57,	49,	41,	33,	25,	17,	 9,	1,
	59,	51,	43,	35,	27,	19,	11,	3,
	61,	53,	45,	37,	29,	21,	13,	5,
	63,	55,	47,	39,	31,	23,	15,	7
};

__constant__ int E_BIT_CUDA[48] = {
	32,	1,	2,	3,	4,	5,
	4,	5,	6,	7,	8,	9,
	8,	9,	10,	11,	12,	13,
	12,	13,	14,	15,	16,	17,
	16,	17,	18,	19,	20,	21,
	20,	21,	22,	23,	24,	25,
	24,	25,	26,	27,	28,	29,
	28,	29,	30,	31,	32,	1
};

__constant__ int S1_CUDA[64] = {
	14,	4,	13,	1,	2,	15,	11,	8,	3,	10,	6,	12,	5,	9,	0,	7,
	0,	15,	7,	4,	14,	2,	13,	1,	10,	6,	12,	11,	9,	5,	3,	8,
	4,	1,	14,	8,	13,	6,	2,	11,	15,	12,	9,	7,	3,	10,	5,	0,
	15,	12,	8,	2,	4,	9,	1,	7,	5,	11,	3,	14,	10,	0,	6,	13
};

__constant__ int S2_CUDA[64] = {
	15,	1,	8,	14,	6,	11,	3,	4,	9,	7,	2,	13,	12,	0,	5,	10,
	3,	13,	4,	7,	15,	2,	8,	14,	12,	0,	1,	10,	6,	9,	11,	5,
	0,	14,	7,	11,	10,	4,	13,	1,	5,	8,	12,	6,	9,	3,	2,	15,
	13,	8,	10,	1,	3,	15,	4,	2,	11,	6,	7,	12,	0,	5,	14,	9,
};

__constant__ int S3_CUDA[64] = {
	10,	0,	9,	14,	6,	3,	15,	5,	1,	13,	12,	7,	11,	4,	2,	8,
	13,	7,	0,	9,	3,	4,	6,	10,	2,	8,	5,	14,	12,	11,	15,	1,
	13,	6,	4,	9,	8,	15,	3,	0,	11,	1,	2,	12,	5,	10,	14,	7,
	1,	10,	13,	0,	6,	9,	8,	7,	4,	15,	14,	3,	11,	5,	2,	12
};

__constant__ int S4_CUDA[64] = {
	7,	13,	14,	3,	0,	6,	9,	10,	1,	2,	8,	5,	11,	12,	4,	15,
	13,	8,	11,	5,	6,	15,	0,	3,	4,	7,	2,	12,	1,	10,	14,	9,
	10,	6,	9,	0,	12,	11,	7,	13,	15,	1,	3,	14,	5,	2,	8,	4,
	3,	15,	0,	6,	10,	1,	13,	8,	9,	4,	5,	11,	12,	7,	2,	14
};

__constant__ int S5_CUDA[64] = {
	2,	12,	4,	1,	7,	10,	11,	6,	8,	5,	3,	15,	13,	0,	14,	9,
	14,	11,	2,	12,	4,	7,	13,	1,	5,	0,	15,	10,	3,	9,	8,	6,
	4,	2,	1,	11,	10,	13,	7,	8,	15,	9,	12,	5,	6,	3,	0,	14,
	11,	8,	12,	7,	1,	14,	2,	13,	6,	15,	0,	9,	10,	4,	5,	3
};

__constant__ int S6_CUDA[64] = {
	12,	1,	10,	15,	9,	2,	6,	8,	0,	13,	3,	4,	14,	7,	5,	11,
	10,	15,	4,	2,	7,	12,	9,	5,	6,	1,	13,	14,	0,	11,	3,	8,
	9,	14,	15,	5,	2,	8,	12,	3,	7,	0,	4,	10,	1,	13,	11,	6,
	4,	3,	2,	12,	9,	5,	15,	10,	11,	14,	1,	7,	6,	0,	8,	13,
};

__constant__ int S7_CUDA[64] = {
	4,	11,	2,	14,	15,	0,	8,	13,	3,	12,	9,	7,	5,	10,	6,	1,
	13,	0,	11,	7,	4,	9,	1,	10,	14,	3,	5,	12,	2,	15,	8,	6,
	1,	4,	11,	13,	12,	3,	7,	14,	10,	15,	6,	8,	0,	5,	9,	2,
	6,	11,	13,	8,	1,	4,	10,	7,	9,	5,	0,	15,	14,	2,	3,	12,
};

__constant__ int S8_CUDA[64] = {
	13,	2,	8,	4,	6,	15,	11,	1,	10,	9,	3,	14,	5,	0,	12,	7,
	1,	15,	13,	8,	10,	3,	7,	4,	12,	5,	6,	11,	0,	14,	9,	2,
	7,	11,	4,	1,	9,	12,	14,	2,	0,	6,	10,	13,	15,	3,	5,	8,
	2,	1,	14,	7,	4,	10,	8,	13,	15,	12,	9,	0,	3,	5,	6,	11,
};

__constant__ int * S_POINTER_CUDA[8] = {
	S1_CUDA, S2_CUDA, S3_CUDA, S4_CUDA, S5_CUDA, S6_CUDA, S7_CUDA, S8_CUDA
};

__constant__ int P_CUDA[32] = {
	16,	7,	20, 21,
	29,	12, 28, 17,
	1,	15, 23, 26,
	5,	18, 31, 10,
	2,	8,	24, 14,
	32, 27, 3,	9,
	19, 13, 30,	6,
	22, 11, 4,	25
};

__constant__ int IP_REV_CUDA[64] = {
	40,	8, 48, 16, 56, 24, 64, 32,
	39, 7, 47, 15, 55, 23, 63, 31,
	38, 6, 46, 14, 54, 22, 62, 30,
	37, 5, 45, 13, 53, 21, 61, 29,
	36, 4, 44, 12, 52, 20, 60, 28,
	35, 3, 43, 11, 51, 19, 59, 27,
	34, 2, 42, 10, 50, 18, 58, 26,
	33, 1, 41,	9, 49, 17, 57, 25
};

__constant__ int SHIFTS_CUDA[16] = {
	1,
	1,
	2,
	2,
	2,
	2,
	2,
	2,
	1,
	2,
	2,
	2,
	2,
	2,
	2,
	1
};

#endif

typedef unsigned long uint32;
typedef unsigned long long uint64;


__host__ uint64 generate_key(int key_size);
__host__ void generate_subkeys(uint64 key, uint64 * subkeys);
__host__ unsigned char get_S_value(unsigned char B, int s_idx);
__host__ uint32 f(uint32 R, uint64 K);
__host__ uint64 encrypt_message(uint64 message, uint64 key);
__global__ void brute_force(uint64 * message, uint64 * encrypted_message, uint64 * cracked_key, volatile int * has_key);
__device__ void generate_subkeys_gpu(uint64 key, uint64 * subkeys);
__device__ unsigned char get_S_value_gpu(unsigned char B, int s_idx);
__device__ uint32 f_gpu(uint32 R, uint64 K);
__device__ uint64 encrypt_message_gpu(uint64 message, uint64 key);
__device__ __host__ void printBits(uint64 n);
__device__ __host__ uint64 permute(uint64 key, int * table, int size);


__global__ void brute_force(uint64 * message, uint64 * encrypted_message, uint64 * cracked_key, volatile int * has_key) {
    
    uint64 i = blockIdx.x * blockDim.x + threadIdx.x;
    uint64 stride = blockDim.x * gridDim.x;

    while(i < ~(0ULL) && *has_key == 0) {
        uint64 currentValue = encrypt_message_gpu(*message, i);
	
        if (currentValue == *encrypted_message) {
	        *cracked_key = i;
	        *has_key = 1;   
        }
        
        i += stride;
    }
}

// helper function for debugging purposes
__device__ __host__ void printBits(uint64 n) { 
    uint64 i; 
    for (i = 1ULL << 63; i > 0; i = i / 2) {
        (n & i) ? printf("1") : printf("0"); 
    }
    printf("\n");
} 

__device__ __host__ uint64 permute(uint64 key, int * table, int size) {
    uint64 permuted_key = 0;

    for(int i = 0; i < size; i++) {
        int bit = (key >> (table[i] - 1)) & 1U;
        if(bit == 1) permuted_key |= 1ULL << i;
    }

    return permuted_key;
}

__device__ void generate_subkeys_gpu(uint64 key, uint64 * subkeys) {
    int size_PC1 = sizeof(PC_1_CUDA)/sizeof(PC_1_CUDA[0]);
    int size_PC2 = sizeof(PC_2_CUDA)/sizeof(PC_2_CUDA[0]);

    uint64 permuted_key = permute(key, PC_1_CUDA, size_PC1);

    uint32 C[17], D[17];

    C[0]  = (uint32) (permuted_key >> 28  & 0xFFFFFFF);
    D[0]  = (uint32) (permuted_key >> 0 & 0xFFFFFFF);

    // apply left shifts
    for(int i = 1; i <= 16; i++) {

        C[i] = C[i-1] << SHIFTS_CUDA[i-1];
        D[i] = D[i-1] << SHIFTS_CUDA[i-1];

        C[i] |= C[i] >> 28;
        D[i] |= D[i] >> 28;

        C[i] &= ~(3UL << 28);
        D[i] &= ~(3UL << 28);

        uint64 merged_subkey = ((uint64)C[i] << 28) | D[i];
        subkeys[i-1] = permute(merged_subkey, PC_2_CUDA, size_PC2);
    }
}

__device__ uint64 encrypt_message_gpu(uint64 message, uint64 key) {
    uint64 K[16];
    uint32 L[17], R[17];

    generate_subkeys_gpu(key, K);

    int size_IP = sizeof(IP_CUDA)/sizeof(IP_CUDA[0]);
    uint64 IP_message = permute(message, IP_CUDA, size_IP);

    L[0]  = (uint32) (IP_message >> 32 & 0xFFFFFFFF);
    R[0]  = (uint32) (IP_message >> 0 & 0xFFFFFFFF);

    for(int i = 1; i <= 16; i++) {
        L[i] = R[i-1];
        R[i] = L[i-1] ^ f_gpu(R[i-1], K[i-1]);
    }

    uint64 RL = ((uint64) R[16] << 32) | L[16];
    uint64 encrypted_message = permute(RL, IP_REV_CUDA, 64);

    return encrypted_message;
}

__device__ uint32 f_gpu(uint32 R, uint64 K) {
    int size_E = sizeof(E_BIT_CUDA)/sizeof(E_BIT_CUDA[0]);
    unsigned char S[8];
    uint32 s_string = 0;
    uint64 expanded_R = permute(R, E_BIT_CUDA, size_E);

    uint64 R_xor_K = expanded_R ^ K;

    for(int i = 0; i < 8; i++) {
        S[i] = get_S_value_gpu((unsigned char) (R_xor_K >> 6*(7 - i)) & 0x3F, i);
        s_string |= S[i];
        s_string <<= (i != 7) ? 4 : 0;
    }
    return (uint32) permute(s_string, P_CUDA, 32);
}

__device__ unsigned char get_S_value_gpu(unsigned char B, int s_idx) {
    unsigned int i = (((B >> 5) & 1U) << 1) | ((B >> 0) & 1U);
    unsigned int j = 0;

    for(int k = 4; k > 0; k--) {
        j |= ((B >> k) & 1U);
        j <<= (k != 1) ? 1 : 0;
    }

    return (unsigned char) S_POINTER_CUDA[s_idx][16 * i + j];
}

__host__ uint64 generate_key(int key_size) {

    srand(time(NULL));
    uint64 key = 0;

    for(int i = 0; i < key_size; i++) {
        const uint64 bit = (uint64) rand() % 2;
        key = (key & ~(1ULL << i)) | (bit << i);
    }
    return key;
}

__host__ void generate_subkeys(uint64 key, uint64 * subkeys) {
    int size_PC1 = sizeof(PC_1)/sizeof(PC_1[0]);
    int size_PC2 = sizeof(PC_2)/sizeof(PC_2[0]);

    uint64 permuted_key = permute(key, PC_1, size_PC1);

    uint32 C[17], D[17];

    C[0]  = (uint32) (permuted_key >> 28  & 0xFFFFFFF);
    D[0]  = (uint32) (permuted_key >> 0 & 0xFFFFFFF);

    // apply left shifts
    for(int i = 1; i <= 16; i++) {

        C[i] = C[i-1] << SHIFTS[i-1];
        D[i] = D[i-1] << SHIFTS[i-1];

        C[i] |= C[i] >> 28;
        D[i] |= D[i] >> 28;

        C[i] &= ~(3UL << 28);
        D[i] &= ~(3UL << 28);

        uint64 merged_subkey = ((uint64)C[i] << 28) | D[i];
        subkeys[i-1] = permute(merged_subkey, PC_2, size_PC2);
    }
}

__host__ uint64 encrypt_message(uint64 message, uint64 key) {
    uint64 K[16];
    uint32 L[17], R[17];

    generate_subkeys(key, K);

    int size_IP = sizeof(IP)/sizeof(IP[0]);
    uint64 IP_message = permute(message, IP, size_IP);

    L[0]  = (uint32) (IP_message >> 32 & 0xFFFFFFFF);
    R[0]  = (uint32) (IP_message >> 0 & 0xFFFFFFFF);

    for(int i = 1; i <= 16; i++) {
        L[i] = R[i-1];
        R[i] = L[i-1] ^ f(R[i-1], K[i-1]);
    }

    uint64 RL = ((uint64) R[16] << 32) | L[16];
    uint64 encrypted_message = permute(RL, IP_REV, 64);

    return encrypted_message;
}

__host__ uint32 f(uint32 R, uint64 K) {
    int size_E = sizeof(E_BIT)/sizeof(E_BIT[0]);
    unsigned char S[8];
    uint32 s_string = 0;
    uint64 expanded_R = permute(R, E_BIT, size_E);

    uint64 R_xor_K = expanded_R ^ K;

    for(int i = 0; i < 8; i++) {
        S[i] = get_S_value((unsigned char) (R_xor_K >> 6*(7 - i)) & 0x3F, i);
        s_string |= S[i];
        s_string <<= (i != 7) ? 4 : 0;
    }
    return (uint32) permute(s_string, P, 32);
}

__host__ unsigned char get_S_value(unsigned char B, int s_idx) {
    unsigned int i = (((B >> 5) & 1U) << 1) | ((B >> 0) & 1U);
    unsigned int j = 0;

    for(int k = 4; k > 0; k--) {
        j |= ((B >> k) & 1U);
        j <<= (k != 1) ? 1 : 0;
    }

    return (unsigned char) S_POINTER[s_idx][16 * i + j];
}

int main(int argc, char ** argv) {

    uint64 data = 0x0123456789ABCDEF;

    if(argc != 2) {
        printf("Usage: %s <key_size>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int key_size = atoi(argv[1]);
    if(key_size > 64) {
        printf("Key size reduced to 64 bits.");
        key_size = 64;
    }
    uint64 key = generate_key(key_size);
    uint64 encrypted_message = encrypt_message(data, key);
    clock_t start, end;
    float time_elapsed;

    // --------- GPU ------------

    int * has_key = NULL;
    int temp = 0;
    uint64 * cracked_key = NULL;
    uint64 found_key;
    uint64 * d_data = NULL;
    uint64 * d_msg = NULL;

    cudaError_t error;

    if((error = cudaMalloc((void **) &has_key, sizeof(int))) != cudaSuccess) {
        ERR(cudaGetErrorString(error));
    }

    if((error = cudaMalloc((void **) &cracked_key, sizeof(uint64))) != cudaSuccess) {
        ERR(cudaGetErrorString(error));
    }

    if((error = cudaMemcpy(has_key, &temp, sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess) {
        ERR(cudaGetErrorString(error));
    }
    
    if((error = cudaMalloc((void **) &d_data, sizeof(uint64))) != cudaSuccess) {
        ERR(cudaGetErrorString(error));
    }
    
    if((error = cudaMalloc((void **) &d_msg, sizeof(uint64))) != cudaSuccess) {
        ERR(cudaGetErrorString(error));
    }
    
    if((error = cudaMemcpy(d_msg, &encrypted_message, sizeof(uint64), cudaMemcpyHostToDevice)) != cudaSuccess) {
        ERR(cudaGetErrorString(error));
    }
    
    if((error = cudaMemcpy(d_data, &data, sizeof(uint64), cudaMemcpyHostToDevice)) != cudaSuccess) {
        ERR(cudaGetErrorString(error));
    }

    printf("\nGPU : Brute forcing DES...\n");
    start = clock();

    brute_force<<<256, 128>>>(d_data, d_msg, cracked_key, has_key);

    if((error = cudaDeviceSynchronize()) != cudaSuccess) ERR(cudaGetErrorString(error));
    
    end = clock();
    time_elapsed = ((float) (end - start)) / CLOCKS_PER_SEC;
    
    if((error = cudaMemcpy(&found_key, cracked_key, sizeof(uint64), cudaMemcpyDeviceToHost)) != cudaSuccess) {
        ERR(cudaGetErrorString(error));
    }

    printf("GPU : Key found!\n");
    printf("GPU : Time elapsed - %f\n", time_elapsed);
    printf("GPU : Cracked key: %llX\n", found_key);

    cudaFree(has_key);
    cudaFree(cracked_key);
    cudaFree(d_data);
    cudaFree(d_msg);

    // --------- CPU -------------

    printf("CPU : Brute forcing DES...\n");
    
    start = clock();

    for(uint64 i = 0; i <= ~(0ULL); i++) {
        uint64 msg = encrypt_message(data, i);
        //printBits(i);
        if(msg == encrypted_message) {
            end = clock();
            time_elapsed = ((float) (end - start)) / CLOCKS_PER_SEC;
            printf("CPU : Key found!\n");
            printf("CPU : Found key: %llX\n", i);
            printf("CPU : Time elapsed - %f\n", time_elapsed);
            break;
        }
    }

    return EXIT_SUCCESS;
}