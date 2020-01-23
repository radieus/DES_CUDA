#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <strings.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__constant__ int PC_1[56] = {
	57,	49,	41,	33,	25,	17,	9,
	1,	58,	50,	42,	34,	26,	18,
	10,	2,	59,	51,	43,	35,	27,
	19,	11,	3,	60,	52,	44,	36,
	63,	55,	47,	39,	31,	23,	15,
	7,	62,	54,	46,	38,	30,	22,
	14,	6,	61,	53,	45,	37,	29,
	21,	13,	5,	28,	20,	12,	4
};

__constant__ int PC_2[48] = {
	14, 17, 11, 24, 1,	5,
	3,	28, 15, 6,	21, 10,
	23, 19, 12, 4,	26, 8,
	16, 7,	27, 20, 13, 2,
	41, 52, 31, 37, 47, 55,
	30, 40, 51, 45, 33, 48,
	44, 49, 39, 56, 34, 53,
	46, 42, 50, 36, 29, 32
};

__constant__ int IP[64] = {
	58,	50,	42,	34,	26,	18,	10,	2,
	60,	52,	44,	36,	28,	20,	12,	4,
	62,	54,	46,	38,	30,	22,	14,	6,
	64,	56,	48,	40,	32,	24,	16,	8,
	57,	49,	41,	33,	25,	17,	 9,	1,
	59,	51,	43,	35,	27,	19,	11,	3,
	61,	53,	45,	37,	29,	21,	13,	5,
	63,	55,	47,	39,	31,	23,	15,	7
};

__constant__ int E_BIT[48] = {
	32,	1,	2,	3,	4,	5,
	4,	5,	6,	7,	8,	9,
	8,	9,	10,	11,	12,	13,
	12,	13,	14,	15,	16,	17,
	16,	17,	18,	19,	20,	21,
	20,	21,	22,	23,	24,	25,
	24,	25,	26,	27,	28,	29,
	28,	29,	30,	31,	32,	1
};

__constant__ int S1[64] = {
	14,	4,	13,	1,	2,	15,	11,	8,	3,	10,	6,	12,	5,	9,	0,	7,
	0,	15,	7,	4,	14,	2,	13,	1,	10,	6,	12,	11,	9,	5,	3,	8,
	4,	1,	14,	8,	13,	6,	2,	11,	15,	12,	9,	7,	3,	10,	5,	0,
	15,	12,	8,	2,	4,	9,	1,	7,	5,	11,	3,	14,	10,	0,	6,	13
};

__constant__ int S2[64] = {
	15,	1,	8,	14,	6,	11,	3,	4,	9,	7,	2,	13,	12,	0,	5,	10,
	3,	13,	4,	7,	15,	2,	8,	14,	12,	0,	1,	10,	6,	9,	11,	5,
	0,	14,	7,	11,	10,	4,	13,	1,	5,	8,	12,	6,	9,	3,	2,	15,
	13,	8,	10,	1,	3,	15,	4,	2,	11,	6,	7,	12,	0,	5,	14,	9,
};

__constant__ int S3[64] = {
	10,	0,	9,	14,	6,	3,	15,	5,	1,	13,	12,	7,	11,	4,	2,	8,
	13,	7,	0,	9,	3,	4,	6,	10,	2,	8,	5,	14,	12,	11,	15,	1,
	13,	6,	4,	9,	8,	15,	3,	0,	11,	1,	2,	12,	5,	10,	14,	7,
	1,	10,	13,	0,	6,	9,	8,	7,	4,	15,	14,	3,	11,	5,	2,	12
};

__constant__ int S4[64] = {
	7,	13,	14,	3,	0,	6,	9,	10,	1,	2,	8,	5,	11,	12,	4,	15,
	13,	8,	11,	5,	6,	15,	0,	3,	4,	7,	2,	12,	1,	10,	14,	9,
	10,	6,	9,	0,	12,	11,	7,	13,	15,	1,	3,	14,	5,	2,	8,	4,
	3,	15,	0,	6,	10,	1,	13,	8,	9,	4,	5,	11,	12,	7,	2,	14
};

__constant__ int S5[64] = {
	2,	12,	4,	1,	7,	10,	11,	6,	8,	5,	3,	15,	13,	0,	14,	9,
	14,	11,	2,	12,	4,	7,	13,	1,	5,	0,	15,	10,	3,	9,	8,	6,
	4,	2,	1,	11,	10,	13,	7,	8,	15,	9,	12,	5,	6,	3,	0,	14,
	11,	8,	12,	7,	1,	14,	2,	13,	6,	15,	0,	9,	10,	4,	5,	3
};

__constant__ int S6[64] = {
	12,	1,	10,	15,	9,	2,	6,	8,	0,	13,	3,	4,	14,	7,	5,	11,
	10,	15,	4,	2,	7,	12,	9,	5,	6,	1,	13,	14,	0,	11,	3,	8,
	9,	14,	15,	5,	2,	8,	12,	3,	7,	0,	4,	10,	1,	13,	11,	6,
	4,	3,	2,	12,	9,	5,	15,	10,	11,	14,	1,	7,	6,	0,	8,	13,
};

__constant__ int S7[64] = {
	4,	11,	2,	14,	15,	0,	8,	13,	3,	12,	9,	7,	5,	10,	6,	1,
	13,	0,	11,	7,	4,	9,	1,	10,	14,	3,	5,	12,	2,	15,	8,	6,
	1,	4,	11,	13,	12,	3,	7,	14,	10,	15,	6,	8,	0,	5,	9,	2,
	6,	11,	13,	8,	1,	4,	10,	7,	9,	5,	0,	15,	14,	2,	3,	12,
};

__constant__ int S8[64] = {
	13,	2,	8,	4,	6,	15,	11,	1,	10,	9,	3,	14,	5,	0,	12,	7,
	1,	15,	13,	8,	10,	3,	7,	4,	12,	5,	6,	11,	0,	14,	9,	2,
	7,	11,	4,	1,	9,	12,	14,	2,	0,	6,	10,	13,	15,	3,	5,	8,
	2,	1,	14,	7,	4,	10,	8,	13,	15,	12,	9,	0,	3,	5,	6,	11,
};

__constant__ int* ALL_S[8] = {
	S1, S2, S3, S4, S5, S6, S7, S8
};

__constant__ int P[32] = {
	16,	7,	20, 21,
	29,	12, 28, 17,
	1,	15, 23, 26,
	5,	18, 31, 10,
	2,	8,	24, 14,
	32, 27, 3,	9,
	19, 13, 30,	6,
	22, 11, 4,	25
};

__constant__ int IP_REVERSED[64] = {
	40,	8, 48, 16, 56, 24, 64, 32,
	39, 7, 47, 15, 55, 23, 63, 31,
	38, 6, 46, 14, 54, 22, 62, 30,
	37, 5, 45, 13, 53, 21, 61, 29,
	36, 4, 44, 12, 52, 20, 60, 28,
	35, 3, 43, 11, 51, 19, 59, 27,
	34, 2, 42, 10, 50, 18, 58, 26,
	33, 1, 41,	9, 49, 17, 57, 25
};

__constant__ int SHIFTS[16] = {
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

typedef unsigned long long uint64;
typedef unsigned long uint32;

__device__ void printBits(uint64 n);
__device__ uint64 generateKey(int key_size);
__device__ uint64 getBit(uint64 number, int bitIdx);
__device__ uint64 permute(uint64 key, int* table, int length);
__device__ void splitKey(uint64 key, uint32* C, uint32* D, int size);
__device__ uint64 shiftKeys(uint64 value, int shifts);
__device__ void createSubkeys(uint64 key, uint64* subKeys);
__device__ uint64 encryptMessage(uint64 key, uint64 message);
__device__ uint32 func(uint32 data, uint64 key);

__global__ void crack(uint64* message, uint64* encrypted_message, uint64* cracked_key, volatile int* has_key) {
    
    uint64 i = blockIdx.x * blockDim.x + threadIdx.x;
    uint64 stride = blockDim.x * gridDim.x;

    while(i < ~(0ULL) && *has_key == 0) {

        uint64 currentValue = encryptMessage(*message, i);
	
        if (currentValue == *encrypted_message) {
	        *cracked_key = i;
	        *has_key = 1;   
        }
        
        i += stride;
    }
}

__device__ void printBits(uint64 n)
{ 
    uint64 i; 

    for (i = 1ULL << 63; i > 0; i = i  >> 1)
        (n & i) ? printf("1") : printf("0"); 

    printf("\n");
}

__device__ uint64 generateKey(int key_size)
{
    srand(time(NULL));
    uint64 key = 0;

    for (int i = 0; i < key_size; i++) {
        const uint64 bit = (uint64) rand() % 2;
        key = (key & ~(1ULL << i)) | (bit << i);
    }

    return key;
}

__device__ uint64 getBit(uint64 number, int bitIdx)
{
	return 1ULL & (number >> bitIdx);
}

__device__ uint64 permute(uint64 key, int* table, int length)
{
    uint64 permKey = 0;

    for (int i = 0; i < length; i++) {
        uint64 bit = (key >> (table[i] - 1)) & 1U;
        permKey = (permKey & ~(1UL << i)) | (bit << i);
    }

    return permKey;
}

__device__ void splitKey(uint64 key, uint32* C, uint32* D, int size)
{
    if (size == 64) {
        C[0] = key & 0xFFFFFFFF;
        D[0] = (key >> size/2) & 0xFFFFFFFF;
    } else if (size == 56) {
        C[0] = key & 0xFFFFFFF;
        D[0] = (key >> size/2) & 0xFFFFFFF;
    }
}

__device__ uint64 shiftKeys(uint64 value, int shifts)
{
    return (value << shifts) | (value >> (28 - shifts));
}

__device__ void createSubkeys(uint64 key, uint64* subKeys) 
{
    uint64 key_plus;
    key_plus = permute(key, PC_1, 56);

	uint32 C[17];
	uint32 D[17];

	splitKey(key_plus, &C[0], &D[0], 56);

    for (int i = 1; i < 17; i++) {
        C[i] = shiftKeys(C[i-1], SHIFTS[i-1]);
        D[i] = shiftKeys(D[i-1], SHIFTS[i-1]);
    }

	for (int i = 0; i < 16; i++) {
		subKeys[i] = C[i + 1] << 28 | D[i + 1];
		subKeys[i] = permute(subKeys[i], PC_2, 48);
	}
}

__device__ uint64 encryptMessage(uint64 message, uint64 key)
{	
	uint64 K[16];
	createSubkeys(key, K);

	uint32 L[17];
	uint32 R[17];
	uint64 ip = permute(message, IP, 64);

	splitKey(ip, &L[0], &R[0], 64);

	for (int i = 1; i < 17; i++) {
		L[i] = R[i-1];
		R[i] = L[i-1] ^ func(R[i-1], K[i-1]);
	}

    uint64 RL = ((uint64) R[16] << 32) | L[16];

    return permute(RL, IP_REVERSED, 64);
}

__device__ uint32 func(uint32 data, uint64 key)
{
	uint64 R_exp = permute(data, E_BIT, 48);
	uint64 xorr = R_exp ^ key;

	uint64 S[8];
	uint64 B[8];

	for (int i = 0; i < 8; i++) {
		B[i] = 0;
		for (int j = 0; j < 6; j++) {
			B[i] = (B[i] & ~(1ULL << j)) | (getBit(xorr, j + (7 - i) * 6) << j);
		}
		uint64 FirstLast = getBit(B[i], 5) << 1 | getBit(B[i], 0);
		uint64 Middle = getBit(B[i], 4) << 3 | getBit(B[i], 3) << 2 | getBit(B[i], 2) << 1 | getBit(B[i], 1);

		S[i] = ALL_S[i][(int)FirstLast * 16 + (int)Middle];
	}

	uint64 result = 0;
	for (int i = 0; i < 8; i++) {
		result |= S[i] << (28 - 4 * i);
	}

	return permute(result, P, 32);
}


int main(int argc, char** argv) 
{ 
    uint64 message = 0x0123456789ABCDEF;
	int key_length = 0;
	clock_t start, end;
	float time_total;

    // GPU part
    volatile int* has_key;
    uint64* cracked_key;

    //int N = 1 << 64;
    cudaMallocManaged(&cracked_key, sizeof(uint64));
    cudaMallocManaged(&has_key, sizeof(volatile int));
    
    uint64 key = generateKey(key_length);
    uint64 encrypted_message = encryptMessage(message, key);
    
	//printf("%llX\n", key);
	// ~~~ GPU ~~~

    printf("CPU : Brute-forcing DES...\n");
	start = clock();
    crack<<<512,1024>>>(message, encrypted_message, cracked_key, has_key);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    end = clock();
    time_total = ((float) (end - start)) / CLOCKS_PER_SEC;
    printf("GPU : Key found!\n");
    printf("GPU : Found key: %llX\n", cracked_key);
    printf("GPU : Total time: %f\n", time_total);


    return 0;

}