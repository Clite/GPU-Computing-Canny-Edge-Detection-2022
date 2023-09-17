#include "Func.h"

/////////////////////////////////////////////////////////////////////////
// 1. �Լ��� Colab ȯ�濡�� �����ؾ� �մϴ�.
// 2. �����Ӱ� �����ϼŵ� ������ ��� �Լ����� GPU�� Ȱ���ؾ� �մϴ�.
// 3. CPU_Func.cu�� �ִ� Image_Check�Լ����� True�� Return�Ǿ�� �ϸ�, CPU�ڵ忡 ���� �ӵ��� ����� �մϴ�.
/////////////////////////////////////////////////////////////////////////

/*
    // ########## ERROR CHECK ########## 
    cudaError_t err;
    err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
    if (err != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
*/

/*
    1. GPU_Grayscale - RGB to GRAY
        GPU_Gray : CPU's Grayscale, line 4~9
*/
__global__ void GPU_Gray(uint8_t* buffer, uint8_t* result)  // Kernel func
{
    // Temp var for fast access
    unsigned int i = (blockIdx.x*blockDim.x + threadIdx.x)*3;   // Index
    int tmp = (buffer[i]*114 + buffer[i+1]*587 + buffer[i+2]*299)/1000;   // Scaling for avoid floating point op
    //int tmp = (buffer[i] * 116 + buffer[i+1] * 601 + buffer[i+2] * 306)>>10;  // Scaling 1024 version

    result[i] = result[i+1] = result[i+2] = tmp;    // R = G = B
    return;
}
void GPU_Grayscale(uint8_t* buf, uint8_t* gray, uint8_t start_add, int len) // Host func
{
    uint8_t* buffer, *result;
    int size = len+2-start_add;

    // Device memory allocation & Initialize
    cudaMalloc((void**)&buffer, size);
    cudaMalloc((void**)&result, size);
    cudaMemcpy(buffer, buf+start_add, size, cudaMemcpyHostToDevice);

    // Kernel func launch (gridsize : height(800), blocksize : width(1000))
    GPU_Gray<<<800, 1000>>> (buffer, result);

    // Device -> CPU data copy
    cudaMemcpy(gray+start_add, result, size, cudaMemcpyDeviceToHost);

    // Free device's dynamic allocate mem
    cudaFree(buffer);
    cudaFree(result);
    return;
}


/*
    2. GPU_Noise_Reduction - Gaussian Blurring
        GPU_Make_Filter : CPU's Noise_Reduction, line 24~31
        GPU_Conv2d_5x5 : CPU's Noise_Reduction, line 46(34)~53
*/
__global__ void GPU_Make_Filter(float* filter, float sigma) // Kernel func
{
    // Temp var
    float s = sigma*sigma*2;
    int i = threadIdx.y-2, j = threadIdx.x-2;

    filter[threadIdx.y*blockDim.x+threadIdx.x] = exp(-(i*i+j*j)/s)/(3.14*s);    // 2D Gaussian func
    return;
}
__global__ void GPU_Conv2d_5x5(float* filter, uint8_t* buffer, uint8_t* result) // Kernel func
{
    // Temp var
    float v = 0;    // ACC
    int x = threadIdx.x, y = blockIdx.x, x_pixel, y_pixel;
    
    /* CONVOLUTION WITHOUT ZERO-PADDING */

	for (int i = -2; i < 3; i++)
    {
        y_pixel = y+i;  // Y-index
        if(y_pixel >= 0 && y_pixel <= gridDim.x-1)  // Skip if index is not valid
        {
            y_pixel *= blockDim.x*3;    // Index scaling
            for (int j = -2; j < 3; j++)
            {
                x_pixel = x+j;   // X-index
                v += x_pixel < 0 || x_pixel >= blockDim.x ? 0 : buffer[y_pixel+x_pixel*3]*filter[(i+2)*5+j+2];  // Convolution if index is valid
            }
        }
    }

    int idx = (blockIdx.x*blockDim.x+threadIdx.x)*3;
    result[idx] = result[idx+1] = result[idx+2] = v;    // R = G = B
    return;
}
void GPU_Noise_Reduction(int width, int height, uint8_t *gray, uint8_t *gaussian)   // Host func
{
    uint8_t *buffer, *result;
    float* filter, sigma = 1.0;
    int size = width*height*3*sizeof(uint8_t);
    dim3 gauss(5, 5);   // Filter size

    // Device memory allocation & Initialize
    cudaMalloc((void**)&filter, 25*sizeof(float));
    cudaMalloc((void**)&buffer, size);
    cudaMalloc((void**)&result, size);
    cudaMemcpy(buffer, gray, size, cudaMemcpyHostToDevice);
    
    // Kernel func launch (blocksize : Filter size(5*5))
    GPU_Make_Filter<<<1, gauss>>> (filter, sigma);
    // Kernel func launch (gridsize : height, blocksize : width)
    GPU_Conv2d_5x5<<<height, width>>> (filter, buffer, result);

    // Device -> CPU data copy
    cudaMemcpy(gaussian, result, size, cudaMemcpyDeviceToHost);

    // Free device's dynamic allocate mem
    cudaFree(filter);
    cudaFree(buffer);
    cudaFree(result);
    return;
}


/*
    3. GPU_Intensity_Gradient - Calculate the gradient (size and angle) for each pixel
        GPU_Make_Operator : CPU's Intensity_Gradient, line 68~73
        GPU_Conv2d_3x3 : CPU's Intensity_Gradient, line 85(80)~114
*/
__global__ void GPU_Make_Operator(short* filter_x, short* filter_y) // Kernel func
{
    // Temp var
    const short x = threadIdx.x, y = threadIdx.y;
    int idx = y*blockDim.x+x;

    switch(y)   // Make sobel operator using switch-case
    {
        case 0:
        {
            filter_x[idx] = x-1;        // -1 0 1
            filter_y[idx] = 1+(x&1);    // 1 2 1
            break;
        }
        case 1:
        {
            filter_x[idx] = (x-1)<<1;   // -2 0 2
            filter_y[idx] = 0;          // 0 0 0
            break;
        }
        case 2:
        {
            filter_x[idx] = x-1;        // -1 0 1
            filter_y[idx] = -1-(x&1);   // -1 -2 -1
            break;
        }
        default:
            filter_x[idx] = filter_y[idx] = 0;
    }
    return;
}
__global__ void GPU_Conv2d_3x3(short* filter_x, short* filter_y, uint8_t* buffer, uint8_t* result_sobel, uint8_t* result_angle) // Kernel func
{
    // Temp var
    int x = threadIdx.x, y = blockIdx.x, x_pixel, y_pixel, t, idx;
    short gx = 0, gy = 0;
    
    /* CONVOLUTION WITHOUT ZERO-PADDING */
    
	for (int i = -1; i < 2; i++)
    {
        y_pixel = y+i;  // Y-index
        if(y_pixel >= 0 && y_pixel <= gridDim.x-1)  // Skip if index is not valid
        {
            y_pixel *= blockDim.x*3;    // Index scaling
            for (int j = -1; j < 2; j++)
            {
                x_pixel = x+j;   // X-index
                if(x_pixel >= 0 && x_pixel < blockDim.x)  // Convolution if index is valid
                {
                    t = buffer[y_pixel+x_pixel*3];
                    gx += filter_x[(i+1)*3+j+1]*t;
                    gy += filter_y[(i+1)*3+j+1]*t;
                }
            }
        }
    }
    t = sqrtf(gx*gx + gy*gy);   // Calculate gradient intensity

    // Quantize gradient angles in units of 45 degrees (0 <= angle < π)
    idx = blockIdx.x*blockDim.x+threadIdx.x;
    float t_angle = t ? atan2f(gy,gx)*57.325 : 0;   // atan(gy/gx)*180/π
    if ((t_angle > -22.5 && t_angle <= 22.5) || (t_angle > 157.5 || t_angle <= -157.5))
        result_angle[idx] = 0;
    else if ((t_angle > 22.5 && t_angle <= 67.5) || (t_angle > -157.5 && t_angle <= -112.5))
        result_angle[idx] = 45;
    else if ((t_angle > 67.5 && t_angle <= 112.5) || (t_angle > -112.5 && t_angle <= -67.5))
        result_angle[idx] = 90;
    else if ((t_angle > 112.5 && t_angle <= 157.5) || (t_angle > -67.5 && t_angle <= -22.5))
        result_angle[idx] = 135;

    uint8_t v = t > 255 ? 255 : t;  // Calibrate to 0 ~ 255
    idx *= 3;
    result_sobel[idx] = result_sobel[idx+1] = result_sobel[idx+2] = v;  // Intensity

    return;
}
void GPU_Intensity_Gradient(int width, int height, uint8_t* gaussian, uint8_t* sobel, uint8_t*angle)    // Host func
{
    uint8_t *buffer, *result_sobel, *result_angle;
    int size = width*height*3*sizeof(uint8_t);
    short* filter_x, *filter_y;
    dim3 sobel_size(3,3);   // Filter size

    // Device memory allocation & Initialize
    cudaMalloc((void**)&filter_x, 9*sizeof(short));
    cudaMalloc((void**)&filter_y, 9*sizeof(short));
    cudaMalloc((void**)&buffer, size);
    cudaMalloc((void**)&result_sobel, size);
    cudaMalloc((void**)&result_angle, width*height*sizeof(uint8_t));
    cudaMemcpy(buffer, gaussian, size, cudaMemcpyHostToDevice);

    // Kernel func launch (blocksize : Filter size(3*3))
    GPU_Make_Operator<<<1, sobel_size>>> (filter_x, filter_y);
    // Kernel func launch (gridsize : height, blocksize : width)
    GPU_Conv2d_3x3<<<height, width>>> (filter_x, filter_y, buffer, result_sobel, result_angle);

    // Device -> CPU data copy
    cudaMemcpy(sobel, result_sobel, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(angle, result_angle, height*width, cudaMemcpyDeviceToHost);

    // Free device's dynamic allocate mem
    cudaFree(filter_x);
    cudaFree(filter_y);
    cudaFree(buffer);
    cudaFree(result_sobel);
    cudaFree(result_angle);
    return;
}


/*
    4. GPU_Non_maximum_Suppression - Suppress not locally maximum value 
        GPU_NMS : CPU's Non_maximum_Suppression, line 122~156
*/
__global__ void GPU_NMS(uint8_t* angle, uint8_t* sobel, uint8_t* result, uint8_t* min, uint8_t* max)    // Kernel func
{
    // Temp var
    const int x = threadIdx.x, y = blockIdx.x;
    int idx = (y*blockDim.x+x)*3;
    uint8_t p1, p2;
    unsigned int v = sobel[idx];

    if(y < 1 || y >= gridDim.x-1 || x < 1 || x >= blockDim.x-1) // Corner pixel is 0
        result[idx] = result[idx+1] = result[idx+2] = 0;
    else
    {
        switch(angle[idx/3])    // Compare intensity with two adjacent pixels according to quantized angle
        {
            case 0: // Up & down pixel
            {
                p1 = sobel[idx+blockDim.x*3];
                p2 = sobel[idx-blockDim.x*3];
                break;
            }
            case 45:    // Upper left & lower right pixel
            {
                p1 = sobel[idx+(blockDim.x-1)*3];
                p2 = sobel[idx-(blockDim.x-1)*3];
                break;
            }
            case 90:    // Left & right pixel
            {
                p1 = sobel[idx+3];
                p2 = sobel[idx-3];
                break;
            }
            case 135:   // Upper right & lower left pixel
            {
                p1 = sobel[idx-(blockDim.x+1)*3];
                p2 = sobel[idx+(blockDim.x+1)*3];
                break;
            }
            default:
                p1 = p2 = 0;
        }

        // Set min, max with atomic op & result
        atomicMin((unsigned int*)min, v);
        atomicMax((unsigned int*)max, v);
        v = v<p1 || v<p2 ? 0 : v;   // Suppression if not locally maximum
        result[idx] = result[idx+1] = result[idx+2] = v;
    }
    return;
}
void GPU_Non_maximum_Suppression(int width, int height, uint8_t *angle,uint8_t *sobel, uint8_t *suppression_pixel, uint8_t& min, uint8_t& max)  // Host func
{
    uint8_t *buffer_angle, *buffer_sobel, *result, *g_min, *g_max;
    int size = width*height*3*sizeof(uint8_t);

    // Device memory allocation & Initialize
    cudaMalloc((void**)&buffer_sobel, size);
    cudaMalloc((void**)&buffer_angle, height*width*sizeof(uint8_t));
    cudaMalloc((void**)&result, size);
    cudaMalloc((void**)&g_min, sizeof(uint8_t));
    cudaMalloc((void**)&g_max, sizeof(uint8_t));
    cudaMemset(g_min, 255, sizeof(uint8_t));
    cudaMemset(g_max, 0, sizeof(uint8_t));
    cudaMemcpy(buffer_sobel, sobel, size, cudaMemcpyHostToDevice);
    cudaMemcpy(buffer_angle, angle, height*width, cudaMemcpyHostToDevice);

    // Kernel func launch (gridsize : height, blocksize : width)
    GPU_NMS<<<height, width>>> (buffer_angle, buffer_sobel, result, g_min, g_max);
    
    // Device -> CPU data copy
    cudaMemcpy(suppression_pixel, result, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&min, g_min, sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max, g_max, sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Free device's dynamic allocate mem
    cudaFree(buffer_sobel);
    cudaFree(buffer_angle);
    cudaFree(result);
    cudaFree(g_min);
    cudaFree(g_max);
    return;
}


/*
    5. GPU_Hysteresis_Thresholding - Double Thresholding & hysteresis
        GPU_Double_Thresholding : CPU's Hysteresis_Thresholding, line 176~200
        GPU_Hysteresis_Check : CPU's Hysteresis_Thresholding, line 202~219 / CPU's Hysteresis_check
*/
// Old Version
/*
__global__ void GPU_Double_Thresholding(uint8_t* buffer, uint8_t* temp, uint8_t* g_min, uint8_t* g_max)    // Kernel func
{
    // Temp var
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    uint8_t max = *g_max, min = *g_min, tmp = buffer[idx*3];
    uint8_t diff = max - min;
    uint8_t low_t = min + diff/100, high_t = min + diff/5;
    //int low_t = max + min*99, high_t = max*20 + min*80, tmp = buffer[idx]*100;

    // Check current pixel is not an edge, a weak edge, or a strong edge
    tmp = tmp < low_t ? 0 : tmp < high_t ? 123 : 255;
    temp[idx] = tmp;
    return;
}
__global__ void GPU_Hysteresis_Check(uint8_t* temp, uint8_t* result)
{
    // Temp var
    const int x = threadIdx.x, y = blockIdx.x;
    int idx = y*blockDim.x+x;
    int tmp = temp[idx];

    // CPU's Hysteresis_check
    if(tmp == 123)  // Weak edge
        for (int i = y-1; i < y+2 && tmp < 255; i++)
            for (int j = x-1; j < x+2 && tmp < 255; j++)
                if ((i < gridDim.x && j < blockDim.x) && (i >= 0 && j >= 0) && temp[i*blockDim.x+j] == 255)   // Adjacent strong edge is exist
                    tmp = 255;    // Treat it as a strong edge
    
    // leaving only a strong edge
    idx *= 3;
    result[idx] = result[idx+1] = result[idx+2] = tmp < 255 ? 0 : 255;
    return;
}
void GPU_Hysteresis_Thresholding(int width, int height, uint8_t *suppression_pixel,uint8_t *hysteresis, uint8_t min, uint8_t max)   // Host func
{
    uint8_t *buffer, *temp, *result, *g_min, *g_max;
    int size = width*height*3*sizeof(uint8_t);

    // Device memory allocation & Initialize
    cudaMalloc((void**)&buffer, size);
    cudaMalloc((void**)&temp, width*height*sizeof(uint8_t));
    cudaMalloc((void**)&result, size);
    cudaMalloc((void**)&g_min, sizeof(uint8_t));
    cudaMalloc((void**)&g_max, sizeof(uint8_t));
    cudaMemcpy(buffer, suppression_pixel, size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_min, &min, sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(g_max, &max, sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Kernel func launch (gridsize : height, blocksize : width)
    GPU_Double_Thresholding<<<height, width>>> (buffer, temp, g_min, g_max);
    cudaDeviceSynchronize();    // For host-device synchronize, but correct even if don't
    GPU_Hysteresis_Check<<<height, width>>> (temp, result);
    
    // Device -> CPU data copy
    cudaMemcpy(hysteresis, result, size, cudaMemcpyDeviceToHost);

    // Free device's dynamic allocate mem
    cudaFree(buffer);
    cudaFree(temp);
    cudaFree(result);
    cudaFree(g_min);
    cudaFree(g_max);
    return;
}
*/
// New Version
/*
    5. GPU_Hysteresis_Thresholding - Double Thresholding & hysteresis
        GPU_Double_Thresholding : CPU's Hysteresis_Thresholding, Hysteresis_check
*/
__global__ void GPU_Double_Thresholding(uint8_t* buffer, uint8_t* result, uint8_t* g_min, uint8_t* g_max)    // Kernel func
{
    // Temp var
    const int x = threadIdx.x, y = blockIdx.x;
    int idx = (y*blockDim.x+x)*3;
    uint8_t max = *g_max, min = *g_min, tmp = buffer[idx];
    uint8_t diff = max - min;
    uint8_t low_t = min + diff/100, high_t = min + diff/5;
    //int low_t = max + min*99, high_t = max*20 + min*80, tmp = buffer[idx]*100;

    // Check current pixel is not an edge, a weak edge, or a strong edge
    tmp = tmp < low_t ? 0 : tmp < high_t ? 123 : 255;

    // CPU's Hysteresis_check
    if(tmp == 123)  // Weak edge
        for (int i = y-1; i < y+2 && tmp < 255; i++)
            for (int j = x-1; j < x+2 && tmp < 255; j++)
                if ((i < gridDim.x && j < blockDim.x) && (i >= 0 && j >= 0) && buffer[(i*blockDim.x+j)*3] >= high_t)   // Adjacent strong edge is exist
                    tmp = 255;    // Treat it as a strong edge
    
    // leaving only a strong edge
    result[idx] = result[idx+1] = result[idx+2] = tmp < 255 ? 0 : 255;
    return;
}
void GPU_Hysteresis_Thresholding(int width, int height, uint8_t *suppression_pixel,uint8_t *hysteresis, uint8_t min, uint8_t max)   // Host func
{
    uint8_t *buffer, *result, *g_min, *g_max;
    int size = width*height*3*sizeof(uint8_t);

    // Device memory allocation & Initialize
    cudaMalloc((void**)&buffer, size);
    cudaMalloc((void**)&result, size);
    cudaMalloc((void**)&g_min, sizeof(uint8_t));
    cudaMalloc((void**)&g_max, sizeof(uint8_t));
    cudaMemcpy(buffer, suppression_pixel, size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_min, &min, sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(g_max, &max, sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Kernel func launch (gridsize : height, blocksize : width)
    GPU_Double_Thresholding<<<height, width>>> (buffer, result, g_min, g_max);
    
    // Device -> CPU data copy
    cudaMemcpy(hysteresis, result, size, cudaMemcpyDeviceToHost);

    // Free device's dynamic allocate mem
    cudaFree(buffer);
    cudaFree(result);
    cudaFree(g_min);
    cudaFree(g_max);
    return;
}