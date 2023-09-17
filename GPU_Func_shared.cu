#include "Func.h"

/////////////////////////////////////////////////////////////////////////
// 1. �Լ��� Colab ȯ�濡�� �����ؾ� �մϴ�.
// 2. �����Ӱ� �����ϼŵ� ������ ��� �Լ����� GPU�� Ȱ���ؾ� �մϴ�.
// 3. CPU_Func.cu�� �ִ� Image_Check�Լ����� True�� Return�Ǿ�� �ϸ�, CPU�ڵ忡 ���� �ӵ��� ����� �մϴ�.
/////////////////////////////////////////////////////////////////////////

/*
    cudaError_t err;
    err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
    if (err != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
*/

__global__ void GPU_Gray(uint8_t* buffer, uint8_t* result)
{
    unsigned int i = (blockIdx.x*blockDim.x + threadIdx.x)*3;
    int tmp = (buffer[i] * 114 + buffer[i + 1] * 587 + buffer[i + 2] * 299)/1000;
    //int tmp = (buffer[i] * 116 + buffer[i + 1] * 601 + buffer[i + 2] * 306)>>10;

    result[i] = result[i+1] = result[i+2] = tmp;
    return;
}
void GPU_Grayscale(uint8_t* buf, uint8_t* gray, uint8_t start_add, int len)
{
    uint8_t* buffer, *result;
    int size = len+2-start_add;

    cudaMalloc((void**)&buffer, size);
    cudaMalloc((void**)&result, size);
    cudaMemcpy(buffer, buf+start_add, size, cudaMemcpyHostToDevice);

    GPU_Gray<<<800, 1000>>> (buffer, result);
    cudaMemcpy(gray+start_add, result, size, cudaMemcpyDeviceToHost);

    cudaFree(buffer);
    cudaFree(result);
    return;
}


__global__ void GPU_Make_Filter(float* filter, float sigma)
{
    float s = sigma*sigma*2;
    int i = threadIdx.y-2, j = threadIdx.x-2;

    filter[threadIdx.y*blockDim.x+threadIdx.x] = exp(-(i*i+j*j)/s)/(3.14*s);
    return;
}
__global__ void GPU_Conv2d_5x5(float* filter_, uint8_t* buffer_, uint8_t* result)
{
    __shared__ uint8_t buffer[5000];
    __shared__ float filter[25];
    float v = 0;
    int x = threadIdx.x, y = blockIdx.x;
    
    for(int i = -2; i < 3; i++)
        if(y+i < 0 || y+i >= gridDim.x)
            buffer[(i+2)*blockDim.x+x] = 0;
        else
            buffer[(i+2)*blockDim.x+x] = buffer_[((y+i)*blockDim.x+x)*3];
    if(x<25)
        filter[x] = filter_[x];
    __syncthreads();

	for (int i = 0; i < 5; i++)
        for (int j = -2; j < 3; j++)
            v += x+j < 0 || x+j >= blockDim.x ? 0 : buffer[i*blockDim.x+x+j]*filter[i*5+(j+2)];

    int idx = (blockIdx.x*blockDim.x+threadIdx.x)*3;
    result[idx] = result[idx+1] = result[idx+2] = v;
    return;
}
void GPU_Noise_Reduction(int width, int height, uint8_t *gray, uint8_t *gaussian)
{
    uint8_t *buffer, *result;
    float* filter, sigma = 1.0;
    int size = width*height*3;
    dim3 gauss(5, 5);

    cudaMalloc((void**)&filter, 100);
    cudaMalloc((void**)&buffer, size);
    cudaMalloc((void**)&result, size);
    cudaMemcpy(buffer, gray, size, cudaMemcpyHostToDevice);
    
    GPU_Make_Filter<<<1, gauss>>> (filter, sigma);
    GPU_Conv2d_5x5<<<height, width>>> (filter, buffer, result);
    cudaDeviceSynchronize();
    cudaMemcpy(gaussian, result, size, cudaMemcpyDeviceToHost);

    cudaFree(filter);
    cudaFree(buffer);
    cudaFree(result);
    return;
}


__global__ void GPU_Make_Operator(int* filter_x, int* filter_y)
{
    const int x = threadIdx.x, y = threadIdx.y;
    int idx = y*blockDim.x+x;

    switch(y)
    {
        case 0:
        {
            filter_x[idx] = x-1;
            filter_y[idx] = 1+(x&1);
            break;
        }
        case 1:
        {
            filter_x[idx] = (x-1)<<1;
            filter_y[idx] = 0;
            break;
        }
        case 2:
        {
            filter_x[idx] = x-1;
            filter_y[idx] = -1-(x&1);
            break;
        }
        default:
            filter_x[idx] = filter_y[idx] = 0;
    }
    return;
}
__global__ void GPU_Conv2d_3x3(int* filter_x_, int* filter_y_, uint8_t* buffer_, uint8_t* result_sobel, uint8_t* result_angle)
{
    __shared__ uint8_t buffer[3000];
    __shared__ float filter_x[9], filter_y[9];
    int x = threadIdx.x, y = blockIdx.x, gx = 0, gy = 0, t, idx;
    
    for(int i = -1; i < 2; i++)
        if(y+i < 0 || y+i >= gridDim.x)
            buffer[(i+1)*blockDim.x+x] = 0;
        else
            buffer[(i+1)*blockDim.x+x] = buffer_[((y+i)*blockDim.x+x)*3];
    if(x<9)
    {
        filter_x[x] = filter_x_[x];
        filter_y[x] = filter_y_[x];
    }
    __syncthreads();

	for (int i = 0; i < 3; i++)
        for (int j = -1; j < 2; j++)
            if(x+j >= 0 && x+j < blockDim.x)
            {
                t = buffer[i*blockDim.x+x+j];
                gx += filter_x[i*3+(j+1)]*t;
                gy += filter_y[i*3+(j+1)]*t;
            }
    t = sqrtf(gx*gx + gy*gy);

    idx = blockIdx.x*blockDim.x+threadIdx.x;
    float t_angle = t ? atan2f(gy,gx)*57.325 : 0;
    if ((t_angle > -22.5 && t_angle <= 22.5) || (t_angle > 157.5 || t_angle <= -157.5))
        result_angle[idx] = 0;
    else if ((t_angle > 22.5 && t_angle <= 67.5) || (t_angle > -157.5 && t_angle <= -112.5))
        result_angle[idx] = 45;
    else if ((t_angle > 67.5 && t_angle <= 112.5) || (t_angle > -112.5 && t_angle <= -67.5))
        result_angle[idx] = 90;
    else if ((t_angle > 112.5 && t_angle <= 157.5) || (t_angle > -67.5 && t_angle <= -22.5))
        result_angle[idx] = 135;

    uint8_t v = t > 255 ? 255 : t;
    idx *= 3;
    result_sobel[idx] = result_sobel[idx+1] = result_sobel[idx+2] = v;

    return;
}
void GPU_Intensity_Gradient(int width, int height, uint8_t* gaussian, uint8_t* sobel, uint8_t*angle)
{
    uint8_t *buffer, *result_sobel, *result_angle;
    int size = width*height*3, *filter_x, *filter_y;
    dim3 sobel_size(3,3);

    cudaMalloc((void**)&filter_x, 36);
    cudaMalloc((void**)&filter_y, 36);
    cudaMalloc((void**)&buffer, size);
    cudaMalloc((void**)&result_sobel, size);
    cudaMalloc((void**)&result_angle, width*height);
    cudaMemcpy(buffer, gaussian, size, cudaMemcpyHostToDevice);

    GPU_Make_Operator<<<1, sobel_size>>> (filter_x, filter_y);
    GPU_Conv2d_3x3<<<height, width>>> (filter_x, filter_y, buffer, result_sobel, result_angle);
    cudaMemcpy(sobel, result_sobel, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(angle, result_angle, height*width, cudaMemcpyDeviceToHost);

    cudaFree(filter_x);
    cudaFree(filter_y);
    cudaFree(buffer);
    cudaFree(result_sobel);
    cudaFree(result_angle);
    return;
}


__global__ void GPU_NMS(uint8_t* angle, uint8_t* sobel, uint8_t* result, uint8_t* min, uint8_t* max)
{
    const int x = threadIdx.x, y = blockIdx.x;
    int idx = (y*blockDim.x+x)*3;
    uint8_t p1, p2;
    unsigned int v = sobel[idx];

    if(y < 1 || y >= gridDim.x-1 || x < 1 || x >= blockDim.x-1)
        result[idx] = result[idx+1] = result[idx+2] = 0;
    else
    {
        switch(angle[idx/3])
        {
            case 0:
            {
                p1 = sobel[idx+blockDim.x*3];
                p2 = sobel[idx-blockDim.x*3];
                break;
            }
            case 45:
            {
                p1 = sobel[idx+(blockDim.x-1)*3];
                p2 = sobel[idx-(blockDim.x-1)*3];
                break;
            }
            case 90:
            {
                p1 = sobel[idx+3];
                p2 = sobel[idx-3];
                break;
            }
            case 135:
            {
                p1 = sobel[idx-(blockDim.x+1)*3];
                p2 = sobel[idx+(blockDim.x+1)*3];
                break;
            }
            default:
                p1 = p2 = 0;
        }
        
        atomicMin((unsigned int*)min, v);
        atomicMax((unsigned int*)max, v);
        result[idx] = result[idx+1] = result[idx+2] = v>=p1 && v>=p2 ? v : 0;
    }

    return;
}
void GPU_Non_maximum_Suppression(int width, int height, uint8_t *angle,uint8_t *sobel, uint8_t *suppression_pixel, uint8_t& min, uint8_t& max)
{
    uint8_t *buffer_angle, *buffer_sobel, *result, *g_min, *g_max;
    int size = width*height*3;

    cudaMalloc((void**)&buffer_sobel, size);
    cudaMalloc((void**)&buffer_angle, height*width);
    cudaMalloc((void**)&result, size);
    cudaMalloc((void**)&g_min, sizeof(uint8_t));
    cudaMalloc((void**)&g_max, sizeof(uint8_t));
    cudaMemset(g_min, 255, sizeof(uint8_t));
    cudaMemset(g_max, 0, sizeof(uint8_t));
    cudaMemcpy(buffer_sobel, sobel, size, cudaMemcpyHostToDevice);
    cudaMemcpy(buffer_angle, angle, height*width, cudaMemcpyHostToDevice);

    GPU_NMS<<<height, width>>> (buffer_angle, buffer_sobel, result, g_min, g_max);
    cudaMemcpy(suppression_pixel, result, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&min, g_min, sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max, g_max, sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(buffer_sobel);
    cudaFree(buffer_angle);
    cudaFree(result);
    cudaFree(g_min);
    cudaFree(g_max);
    return;
}

/*
__global__ void GPU_Hysteresis(uint8_t* buffer, uint8_t* temp, uint8_t* g_min, uint8_t* g_max)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    uint8_t max = *g_max, min = *g_min, tmp = buffer[idx*3];
    uint8_t diff = max - min;
    uint8_t low_t = min + diff/100, high_t = min + diff/5;
    //int low_t = max + min*99, high_t = max*20 + min*80, tmp = buffer[idx]*100;

    tmp = tmp < low_t ? 0 : tmp < high_t ? 123 : 255;
    temp[idx] = tmp;
    return;
}
__global__ void GPU_Hysteresis_Check(uint8_t* temp, uint8_t* result)
{
    const int x = threadIdx.x, y = blockIdx.x;
    int idx = y*blockDim.x+x;
    int tmp = temp[idx];

    if(tmp == 123)
        for (int i = y-1; i < y+2 && tmp < 255; i++)
            for (int j = x-1; j < x+2 && tmp < 255; j++)
                if ((i < gridDim.x && j < blockDim.x) && (i >= 0 && j >= 0) && temp[i*blockDim.x+j] == 255)
                    tmp = 255;
    
    idx *= 3;
    result[idx] = result[idx+1] = result[idx+2] = tmp < 255 ? 0 : 255;
    return;
}
void GPU_Hysteresis_Thresholding(int width, int height, uint8_t *suppression_pixel,uint8_t *hysteresis, uint8_t min, uint8_t max)
{
    uint8_t *buffer, *temp, *result, *g_min, *g_max;
    int size = width*height*3;

    cudaMalloc((void**)&buffer, size);
    cudaMalloc((void**)&temp, width*height);
    cudaMalloc((void**)&result, size);
    cudaMalloc((void**)&g_min, sizeof(uint8_t));
    cudaMalloc((void**)&g_max, sizeof(uint8_t));
    cudaMemcpy(buffer, suppression_pixel, size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_min, &min, sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(g_max, &max, sizeof(uint8_t), cudaMemcpyHostToDevice);

    GPU_Hysteresis<<<height, width>>> (buffer, temp, g_min, g_max);
    cudaDeviceSynchronize();
    GPU_Hysteresis_Check<<<height, width>>> (temp, result);
    
    cudaMemcpy(hysteresis, result, size, cudaMemcpyDeviceToHost);

    cudaFree(buffer);
    cudaFree(temp);
    cudaFree(result);
    cudaFree(g_min);
    cudaFree(g_max);
    return;
}
*/
__global__ void GPU_Double_Thresholding(uint8_t* buffer, uint8_t* result, uint8_t* g_min, uint8_t* g_max)
{
    const int x = threadIdx.x, y = blockIdx.x;
    int idx = (y*blockDim.x+x)*3;
    uint8_t max = *g_max, min = *g_min, tmp = buffer[idx];
    uint8_t diff = max - min;
    uint8_t low_t = min + diff/100, high_t = min + diff/5;
    //int low_t = max + min*99, high_t = max*20 + min*80, tmp = buffer[idx]*100;

    tmp = tmp < low_t ? 0 : tmp < high_t ? 123 : 255;

    if(tmp == 123)
        for (int i = y-1; i < y+2 && tmp < 255; i++)
            for (int j = x-1; j < x+2 && tmp < 255; j++)
                if ((i < gridDim.x && j < blockDim.x) && (i >= 0 && j >= 0) && buffer[(i*blockDim.x+j)*3] >= high_t)
                    tmp = 255;
    
    result[idx] = result[idx+1] = result[idx+2] = tmp < 255 ? 0 : 255;
    return;
}
void GPU_Hysteresis_Thresholding(int width, int height, uint8_t *suppression_pixel,uint8_t *hysteresis, uint8_t min, uint8_t max)
{
    uint8_t *buffer, *result, *g_min, *g_max;
    int size = width*height*3*sizeof(uint8_t);

    cudaMalloc((void**)&buffer, size);
    cudaMalloc((void**)&result, size);
    cudaMalloc((void**)&g_min, sizeof(uint8_t));
    cudaMalloc((void**)&g_max, sizeof(uint8_t));
    cudaMemcpy(buffer, suppression_pixel, size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_min, &min, sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(g_max, &max, sizeof(uint8_t), cudaMemcpyHostToDevice);

    GPU_Double_Thresholding<<<height, width>>> (buffer, result, g_min, g_max);
    
    cudaMemcpy(hysteresis, result, size, cudaMemcpyDeviceToHost);

    cudaFree(buffer);
    cudaFree(result);
    cudaFree(g_min);
    cudaFree(g_max);
    return;
}