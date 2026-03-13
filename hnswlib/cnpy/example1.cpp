#include"cnpy.h"
#include<complex>
#include<cstdlib>
#include<iostream>
#include<map>
#include<string>

const int Nx = 128;
const int Ny = 64;
const int Nz = 32;

float half_to_float(uint16_t h) {
    // 参考 IEEE 754 半精度转换公式
    int s = (h >> 15) & 0x1;                   // 符号位
    int e = (h >> 10) & 0x1F;                  // 指数部分
    int f = h & 0x3FF;                         // 尾数部分
    if (e == 0) {                              // 次正规数
        return (s ? -1 : 1) * std::ldexp(f, -24);
    } else if (e == 31) {                      // 特殊值（NaN 或 Infinity）
        return (s ? -1 : 1) * (f ? NAN : INFINITY);
    } else {                                   // 规范化数
        return (s ? -1 : 1) * std::ldexp(f + 1024, e - 15 - 10);
    }
}


void load_from_msmarco(std::vector<float>& base_data, 
                       int file_numbers) {

    for (int i = 0; i < file_numbers; i++) {
        std::string embfile_name = "/ssddata/0.6b_128d_dataset/encoding" + std::to_string(i) + "_float16.npy";
        std::string lensfile_name = "/ssddata/0.6b_128d_dataset/doclens" + std::to_string(i) + ".npy";
        cnpy::NpyArray arr_npy = cnpy::npy_load(embfile_name);
        cnpy::NpyArray lens_npy = cnpy::npy_load(lensfile_name);
        uint16_t* raw_vec_data = arr_npy.data<uint16_t>();
        size_t num_elements = arr_npy.shape[0] * arr_npy.shape[1];
        // int* lens_data = lens_npy.data<int>();
        std::complex<int>* lens_data = lens_npy.data<std::complex<int>>();
        size_t doc_num = lens_npy.shape[0];
        int offset = 0;
        std::cout << arr_npy.shape[0] << " " << arr_npy.shape[1] << " " << num_elements << std::endl;
        std::cout << doc_num << std::endl;
        std::cout << lens_npy.word_size << std::endl;
        assert (doc_num == 25000);
        
        for (size_t i = 0; i < num_elements; ++i) {
            base_data.push_back(static_cast<float>(half_to_float(raw_vec_data[i])));
        }
        std::cout << base_data[0] << std::endl;
        std::cout << base_data[25 * 128 + 25] << std::endl;
        std::cout << base_data[2500 * 128 + 0] << std::endl;
        std::cout << base_data[249999 * 128 + 127] << std::endl;

        std::cout << lens_data[0].real() << std::endl;
        std::cout << lens_data[25].real() << std::endl;
        std::cout << lens_data[2500].real() << std::endl;
        std::cout << lens_data[24999].real() << std::endl;
    }
}

int main()
{
    std::vector<float> base_data;
    std::vector<int> base_vec_num;
    std::vector<float> query_data;
    // Generate dataset
    load_from_msmarco(base_data, 1);
    return 0;
    //set random seed so that result is reproducible (for testing)
    srand(0);
    //create random data
    std::vector<std::complex<double>> data(Nx*Ny*Nz);
    for(int i = 0;i < Nx*Ny*Nz;i++) data[i] = std::complex<double>(rand(),rand());

    //save it to file
    cnpy::npy_save("arr1.npy",&data[0],{Nz,Ny,Nx},"w");

    //load it into a new array
    cnpy::NpyArray arr = cnpy::npy_load("arr1.npy");
    std::complex<double>* loaded_data = arr.data<std::complex<double>>();
    
    //make sure the loaded data matches the saved data
    assert(arr.word_size == sizeof(std::complex<double>));
    assert(arr.shape.size() == 3 && arr.shape[0] == Nz && arr.shape[1] == Ny && arr.shape[2] == Nx);
    for(int i = 0; i < Nx*Ny*Nz;i++) assert(data[i] == loaded_data[i]);

    //append the same data to file
    //npy array on file now has shape (Nz+Nz,Ny,Nx)
    cnpy::npy_save("arr1.npy",&data[0],{Nz,Ny,Nx},"a");

    //now write to an npz file
    //non-array variables are treated as 1D arrays with 1 element
    double myVar1 = 1.2;
    char myVar2 = 'a';
    cnpy::npz_save("out.npz","myVar1",&myVar1,{1},"w"); //"w" overwrites any existing file
    cnpy::npz_save("out.npz","myVar2",&myVar2,{1},"a"); //"a" appends to the file we created above
    cnpy::npz_save("out.npz","arr1",&data[0],{Nz,Ny,Nx},"a"); //"a" appends to the file we created above

    //load a single var from the npz file
    cnpy::NpyArray arr2 = cnpy::npz_load("out.npz","arr1");

    //load the entire npz file
    cnpy::npz_t my_npz = cnpy::npz_load("out.npz");
    
    //check that the loaded myVar1 matches myVar1
    cnpy::NpyArray arr_mv1 = my_npz["myVar1"];
    double* mv1 = arr_mv1.data<double>();
    assert(arr_mv1.shape.size() == 1 && arr_mv1.shape[0] == 1);
    assert(mv1[0] == myVar1);

    // load
    cnpy::NpyArray arr_npy = cnpy::npy_load("/ssddata/0.6b_128d_dataset/encoding0_float16.npy");

    if (arr_npy.word_size == 2) {
        std::cout << "Detected float16 data, converting to float32..." << std::endl;
        // 将 float16 数据指针读取为 uint16_t（存储形式）
        uint16_t* raw_data = arr_npy.data<uint16_t>();
        size_t num_elements = arr_npy.shape[0] * arr_npy.shape[1]; // 例如一维数据
        // 转换到 float32
        std::vector<float> converted_data(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
            converted_data[i] = static_cast<float>(half_to_float(raw_data[i])); // 手动转换
        }
        // 打印转换后的数据

        std::cout << arr_npy.word_size << std::endl;
        std::cout << arr_npy.shape.size() << std::endl;
        std::cout << arr_npy.shape[0] << std::endl;
        std::cout << arr_npy.shape[1] << std::endl;
        std::cout << converted_data[0 * 128 + 0] << std::endl;
        std::cout << converted_data[1 * 128 + 1] << std::endl;
        std::cout << converted_data[2 * 128 + 2] << std::endl;
        std::cout << converted_data[1000 * 128 + 127] << std::endl;
        std::cout << converted_data[2500 * 128 + 0] << std::endl;
        std::cout << converted_data[249999 * 128 + 127] << std::endl;
        std::cout << std::endl;
    }

    // std::complex<float>* loaded_npy_data = arr_npy.data<std::complex<float>>();
    // std::cout << arr_npy.word_size << std::endl;
    // std::cout << sizeof(std::complex<float>) << std::endl;
    // std::cout << arr_npy.shape.size() << std::endl;
    // std::cout << arr_npy.shape[0] << std::endl;
    // std::cout << arr_npy.shape[1] << std::endl;
    // std::cout << loaded_npy_data[0 * 128 + 0] << std::endl;
    // std::cout << loaded_npy_data[1 * 128 + 1] << std::endl;
    // std::cout << loaded_npy_data[2 * 128 + 2] << std::endl;
    // std::cout << loaded_npy_data[1000 * 128 + 127] << std::endl;
    // std::cout << loaded_npy_data[2500 * 128 + 0] << std::endl;
    // std::cout << loaded_npy_data[249999 * 128 + 127] << std::endl;
    // // for(int i = 0; i < Nx*Ny*Nz;i++) assert(data[i] == loaded_data[i]);

    // //make sure the loaded data matches the saved data
    // assert(arr.word_size == sizeof(std::complex<float>));
    // assert(arr.shape.size() == 2 && arr.shape[0] == 1598945);

}
