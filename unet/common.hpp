#ifndef UNET_COMMON_H_
#define UNET_COMMON_H_

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"


#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace nvinfer1;


// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}


ILayer* convBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int p = ksize / 2;
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(g);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-3);

    // hard_swish = x * hard_sigmoid
    auto hsig = network->addActivation(*bn1->getOutput(0), ActivationType::kHARD_SIGMOID);
    assert(hsig);
    hsig->setAlpha(1.0 / 6.0);
    hsig->setBeta(0.5);
    auto ew = network->addElementWise(*bn1->getOutput(0), *hsig->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}


ILayer* doubleConv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, std::string lname, int midch){

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, midch, DimsHW{ksize, ksize}, weightMap[lname + ".double_conv.0.weight"], weightMap[lname + ".double_conv.0.bias"]);
    conv1->setStrideNd(DimsHW{1, 1});
    conv1->setPaddingNd(DimsHW{1, 1});
    conv1->setNbGroups(1);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".double_conv.1", 0);

    // IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + ".double_conv.3.weight"], weightMap[lname + ".double_conv.3.bias"]);
    conv2->setStrideNd(DimsHW{1, 1});
    conv2->setPaddingNd(DimsHW{1, 1});
    conv2->setNbGroups(1);

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".double_conv.4", 0);

    // IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kLEAKY_RELU);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    return relu2;
}

ILayer* down(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int p, std::string lname){
    IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{2, 2});
    assert(pool1);
    ILayer* dcov1 = doubleConv(network,weightMap,*pool1->getOutput(0),outch,3,lname+".maxpool_conv.1",outch);
    assert(dcov1);
    return dcov1;
}

ILayer* up(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input1,  ITensor& input2, int resize,  int outch, int midch, std::string lname){
    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * resize * 2 * 2));
    for (int i = 0; i < resize * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    Weights deconvwts1{DataType::kFLOAT, deval, resize * 2 * 2};
    IDeconvolutionLayer* deconv1 = network->addDeconvolutionNd(input1, resize, DimsHW{2, 2}, deconvwts1, emptywts);
    deconv1->setStrideNd(DimsHW{2, 2});
    deconv1->setNbGroups(resize);
    weightMap["deconvwts."+lname] = deconvwts1;

    int diffx = input2.getDimensions().d[1]-deconv1->getOutput(0)->getDimensions().d[1];
    int diffy = input2.getDimensions().d[2]-deconv1->getOutput(0)->getDimensions().d[2];
    // IPoolingLayer* pool1 = network->addPooling(dcov1, PoolingType::kMAX, DimsHW{2, 2});
    // pool1->setStrideNd(DimsHW{2, 2});
    // dcov1->add_pading
    IPaddingLayer* pad1 = network->addPaddingNd(*deconv1->getOutput(0), DimsHW{diffx / 2, diffy / 2}, DimsHW{diffx - (diffx / 2), diffy - (diffy / 2)} );
    // dcov1->setPaddingNd(DimsHW{diffx / 2, diffx - diffx / 2},DimsHW{diffy / 2, diffy - diffy / 2});
    ITensor* inputTensors[] = {&input2, pad1->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 2);
    assert(cat);

    // free(deval);
    if (midch==64){
        ILayer* dcov1 = doubleConv(network,weightMap,*cat->getOutput(0),outch,3,lname+".conv",outch);
        assert(dcov1);
        return dcov1;
    }else{
        int midch1 = outch/2;
        ILayer* dcov1 = doubleConv(network,weightMap,*cat->getOutput(0),midch1,3,lname+".conv",outch);
        assert(dcov1);
        return dcov1;
    }
    
    // assert(dcov1);

    // return dcov1;
}

ILayer* outConv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, std::string lname){
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, 1, DimsHW{1, 1}, weightMap[lname + ".conv.weight"], weightMap[lname + ".conv.bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1});
    conv1->setPaddingNd(DimsHW{0, 0});
    conv1->setNbGroups(1);
    return conv1;
}


int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
                strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

#endif
