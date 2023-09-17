# GPU-Computing-Canny-Edge-Detection-2022
C++ CUDA를 사용한 Canny edge detection 알고리즘 병렬처리

+ 기존 c++ 기반 코드를 CUDA를 사용한 병렬처리를 활용해 수행시간 감소
+ Colab 등의 Jupyter Notebook 기반 환경에서 사용시 아래 코드를 순서대로 입력시 테스트 가능
  - !nvcc -o (파일 이름) Canny.cu CPU_Func.cu GPU_Func.cu
  - !./(파일 이름)
