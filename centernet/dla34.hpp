#ifndef YOLOV5_COMMON_H_
#define YOLOV5_COMMON_H_

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

cv::Mat preprocess_img(cv::Mat& img) {
    int w, h, x, y;
    float r_w = Yolo::INPUT_W / (img.cols*1.0);
    float r_h = Yolo::INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        w = Yolo::INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (Yolo::INPUT_H - h) / 2;
    } else {
        w = r_h * img.cols;
        h = Yolo::INPUT_H;
        x = (Yolo::INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(Yolo::INPUT_H, Yolo::INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    int l, r, t, b;
    float r_w = Yolo::INPUT_W / (img.cols * 1.0);
    float r_h = Yolo::INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
    return a.conf > b.conf;
}

void nms(std::vector<Yolo::Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5) {
    int det_size = sizeof(Yolo::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

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
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
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
    Weights scale{ DataType::kFLOAT, scval, len };

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* convBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int p = ksize / 2;
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ s, s });
    conv1->setPaddingNd(DimsHW{ p, p });
    conv1->setNbGroups(g);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-3);

    auto relu = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu);
    return relu;
}

IActivationLayer* basicBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int conv1s, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{3, 3}, weightMap[lname + ".conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{conv1s, conv1s});
    conv1->setPaddingNd(DimsHW{1, 1});
    conv1->setNbGroups(1);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + ".conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{1, 1});
    conv2->setPaddingNd(DimsHW{1, 1});
    conv2->setNbGroups(1);

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".bn2", 1e-5);

    return bn2;
}

IActivationLayer* project(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + ".0.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1});
    conv1->setPaddingNd(DimsHW{0, 0});
    conv1->setNbGroups(1);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);
    return bn1;
}


ILayer* root(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor** inputTensors, int kTensors, int outch, std::string lname) {
    auto cat = network->addConcatenation(inputTensors, kTensors);
    // cat->setAxis(0);
    auto conv = convBlock(network, weightMap, *cat->getOutput(0), outch, 1, 1, 1, ".root");
    return conv;
}

ILayer* dcn(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ILayer& inputlayer, int outch, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + ".conv.conv_offset_mask.weight"], weightMap[lname + ".conv.conv_offset_mask.bias"]);
    conv1->setStrideNd(DimsHW{1, 1});
    conv1->setPaddingNd(DimsHW{1, 1});
    conv1->setNbGroups(1);
    ISliceLayer *s1 = network->addSlice(*conv1->getOutput(0), Dims3{ 0, 0, 0 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer *s2 = network->addSlice(*conv1->getOutput(0), Dims3{ 0, 0, 0 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    auto sig = network->addActivation(*s2->getOutput(0), ActivationType::kSIGMOID);
    auto dcn1 = network->addDCNv2Layer(network, weightMap, inputlayer, s1, sig);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, dcn1->getOutput(0), lname + ".actf.0", 1e-5);
    auto relu = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu);
    return relu;
}

//Weights emptywts{DataType::kFLOAT, nullptr, 0};
//IDeconvolutionLayer* dconv1 = network->addDeconvolutionNd(input, outch, DimsHW{ 4, 4}, weightMap[lname + ".weight"], emptywts);
// conv1->setStrideNd(DimsHW{2, 2});
// conv1->setPaddingNd(DimsHW{2, 2});
// dconv1->setNbGroups(128);


ILayer* level0(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int p = ksize / 2;
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + ".0.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ s, s });
    conv1->setPaddingNd(DimsHW{ p, p });
    conv1->setNbGroups(g);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-3);

    auto relu = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu);
    return relu;
}

//level0 == level1

ILayer* level2(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int oc1, int oc2, int oc3, int ksize, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int p = ksize / 2;
    auto block1 = basicBlock(network, weightMap, input, oc1, 2, ".tree1");
    IPoolingLayer *pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{2, 2});
    pool1->setStrideNd(DimsHW{2, 2});
    auto pro1 = project(network, weightMap, *pool1->getOutput(0), oc1, ".project");

    auto ew1 = network->addElementWise(block1->getOutput(0), pro1->getOutput(0), ElementWiseOperation::kSUM);

    IActivationLayer* relu1 = network->addActivation(ew1->getOutput(0), ActivationType::kRELU);

    auto block2 = basicBlock(network, weightMap, *relu1->getOutput(0), oc2, 1, ".tree2");
    auto ew2 = network->addElementWise(*relu1->getOutput(0), block2->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* relu2 = network->addActivation(ew2->getOutput(0), ActivationType::kRELU);
    ITensor* inputTensors[] = { *relu1->getOutput(0), *relu2->getOutput(0) };
    auto relu3 = root(network, weightMap, inputTensors, 2, oc3, lname);
    assert(relu3);
    return relu3;
}


ILayer* level3(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int p = ksize / 2;
    auto block1 = basicBlock(network, weightMap, input, oc1, 2, ".tree1.tree1");
    IPoolingLayer *pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{2, 2});
    pool1->setStrideNd(DimsHW{2, 2});
    auto pro1 = project(network, weightMap, *pool1->getOutput(0), oc1, ".tree1.project");

    auto ew1 = network->addElementWise(*block1->getOutput(0), *pro1->getOutput(0), ElementWiseOperation::kSUM);

    IActivationLayer* relu1 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);

    auto block2 = basicBlock(network, weightMap, *relu1->getOutput(0), oc2, 1, ".tree1.tree2");
    auto ew2 = network->addElementWise(*relu1->getOutput(0), block2->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* relu2 = network->addActivation(ew2->getOutput(0), ActivationType::kRELU);
    ITensor* inputTensors[] = { *relu1->getOutput(0), *relu2->getOutput(0) };
    auto relu3 = root(network, weightMap, inputTensors, 2, oc3, ".tree1");
    assert(relu3);

    auto block3 = basicBlock(network, weightMap, relu3->getOutput(0), oc1, 1, ".tree2.tree1");
    auto ew3 = network->addElementWise(block3->getOutput(0), relu3->getOutput(0), ElementWiseOperation::kSUM);

    IActivationLayer* relu4 = network->addActivation(ew3->getOutput(0), ActivationType::kRELU);

    auto block4 = basicBlock(network, weightMap, *relu4->getOutput(0), oc2, 1, ".tree2.tree2");
    auto ew4 = network->addElementWise(*relu4->getOutput(0), block4->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* relu5 = network->addActivation(ew4->getOutput(0), ActivationType::kRELU);

    IPoolingLayer *pool2 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{2, 2});
    pool2->setStrideNd(DimsHW{2, 2});
    ITensor* inputTensors[] = { *pool2->getOutput(0), relu3->getOutput(0), *relu4->getOutput(0), *relu5->getOutput(0) };
    auto relu6 = root(network, weightMap, inputTensors, 4, oc3, ".tree2");
    assert(relu6);

    return relu6;
}

ILayer* level4(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int p = ksize / 2;
    auto block1 = basicBlock(network, weightMap, input, oc1, 2, ".tree1.tree1");
    IPoolingLayer *pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{2, 2});
    pool1->setStrideNd(DimsHW{2, 2});
    auto pro1 = project(network, weightMap, *pool1->getOutput(0), oc1, ".tree1.project");

    auto ew1 = network->addElementWise(*block1->getOutput(0), *pro1->getOutput(0), ElementWiseOperation::kSUM);

    IActivationLayer* relu1 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);

    auto block2 = basicBlock(network, weightMap, *relu1->getOutput(0), oc2, 1, ".tree1.tree2");
    auto ew2 = network->addElementWise(*relu1->getOutput(0), block2->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* relu2 = network->addActivation(ew2->getOutput(0), ActivationType::kRELU);
    ITensor* inputTensors[] = { *relu1->getOutput(0), *relu2->getOutput(0) };
    auto relu3 = root(network, weightMap, inputTensors, 2, oc3, ".tree1");
    assert(relu3);

    auto block3 = basicBlock(network, weightMap, relu3->getOutput(0), oc1, 1, ".tree2.tree1");
    auto ew3 = network->addElementWise(block3->getOutput(0), relu3->getOutput(0), ElementWiseOperation::kSUM);

    IActivationLayer* relu4 = network->addActivation(ew3->getOutput(0), ActivationType::kRELU);

    auto block4 = basicBlock(network, weightMap, *relu4->getOutput(0), oc2, 1, ".tree2.tree2");
    auto ew4 = network->addElementWise(*relu4->getOutput(0), block4->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* relu5 = network->addActivation(ew4->getOutput(0), ActivationType::kRELU);

    IPoolingLayer *pool2 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{2, 2});
    pool2->setStrideNd(DimsHW{2, 2});
    ITensor* inputTensors[] = { *pool2->getOutput(0), relu3->getOutput(0), *relu4->getOutput(0), *relu5->getOutput(0) };
    auto relu6 = root(network, weightMap, inputTensors, 4, oc3, ".tree2");
    assert(relu6);

    return relu6;
}

ILayer* level5(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int oc1, int oc2, int oc3, int ksize, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int p = ksize / 2;
    auto block1 = basicBlock(network, weightMap, input, oc1, 2, ".tree1");
    IPoolingLayer *pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{2, 2});
    pool1->setStrideNd(DimsHW{2, 2});
    auto pro1 = project(network, weightMap, *pool1->getOutput(0), oc1, ".project");

    auto ew1 = network->addElementWise(block1->getOutput(0), pro1->getOutput(0), ElementWiseOperation::kSUM);

    IActivationLayer* relu1 = network->addActivation(ew1->getOutput(0), ActivationType::kRELU);

    auto block2 = basicBlock(network, weightMap, *relu1->getOutput(0), oc2, 1, ".tree2");
    auto ew2 = network->addElementWise(*relu1->getOutput(0), block2->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* relu2 = network->addActivation(ew2->getOutput(0), ActivationType::kRELU);
    ITensor* inputTensors[] = { *pool1->getOutput(0), *relu1->getOutput(0), *relu2->getOutput(0) };
    auto relu3 = root(network, weightMap, inputTensors, 3, oc3, lname);
    assert(relu3);
    return relu3;
}


inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
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

IPluginV2Layer* addDCNv2Layer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, IConvolutionLayer* t0, IConvolutionLayer* t1, IConvolutionLayer* t2)
{
    auto creator = getPluginRegistry()->getPluginCreator("DCNv2", "001");

    std::vector<float> anchors_yolo = getAnchors(weightMap);
    PluginField pluginMultidata[4];
    int NetData[4];
    NetData[0] = Yolo::CLASS_NUM;
    NetData[1] = Yolo::INPUT_W;
    NetData[2] = Yolo::INPUT_H;
    NetData[3] = Yolo::MAX_OUTPUT_BBOX_COUNT;
    pluginMultidata[0].data = NetData;
    pluginMultidata[0].length = 3;
    pluginMultidata[0].name = "netdata";
    pluginMultidata[0].type = PluginFieldType::kFLOAT32;
    int scale[3] = { 8, 16, 32 };
    int plugindata[3][8];
    std::string names[3];
    for (int k = 1; k < 4; k++)
    {
        plugindata[k - 1][0] = Yolo::INPUT_W / scale[k - 1];
        plugindata[k - 1][1] = Yolo::INPUT_H / scale[k - 1];
        for (int i = 2; i < 8; i++)
        {
            plugindata[k - 1][i] = int(anchors_yolo[(k - 1) * 6 + i - 2]);
        }
        pluginMultidata[k].data = plugindata[k - 1];
        pluginMultidata[k].length = 8;
        names[k - 1] = "yolodata" + std::to_string(k);
        pluginMultidata[k].name = names[k - 1].c_str();
        pluginMultidata[k].type = PluginFieldType::kFLOAT32;
    }

    PluginFieldCollection pluginData;
    pluginData.nbFields = 11;
    pluginData.fields = pluginMultidata;
    IPluginV2 *pluginObj = creator->createPlugin("DCNlayer", &pluginData);
    ITensor* inputTensors_dcn[] = { t2->getOutput(0), t1->getOutput(0), t0->getOutput(0) };
    auto dcn = network->addPluginV2(inputTensors_dcn, 3, *pluginObj);
    return dcn;
}


//example plugin layer.
IPluginV2Layer* addYoLoLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, IConvolutionLayer* det0, IConvolutionLayer* det1, IConvolutionLayer* det2)
{
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    std::vector<float> anchors_yolo = getAnchors(weightMap);
    PluginField pluginMultidata[4];
    int NetData[4];
    NetData[0] = Yolo::CLASS_NUM;
    NetData[1] = Yolo::INPUT_W;
    NetData[2] = Yolo::INPUT_H;
    NetData[3] = Yolo::MAX_OUTPUT_BBOX_COUNT;
    pluginMultidata[0].data = NetData;
    pluginMultidata[0].length = 3;
    pluginMultidata[0].name = "netdata";
    pluginMultidata[0].type = PluginFieldType::kFLOAT32;
    int scale[3] = { 8, 16, 32 };
    int plugindata[3][8];
    std::string names[3];
    for (int k = 1; k < 4; k++)
    {
        plugindata[k - 1][0] = Yolo::INPUT_W / scale[k - 1];
        plugindata[k - 1][1] = Yolo::INPUT_H / scale[k - 1];
        for (int i = 2; i < 8; i++)
        {
            plugindata[k - 1][i] = int(anchors_yolo[(k - 1) * 6 + i - 2]);
        }
        pluginMultidata[k].data = plugindata[k - 1];
        pluginMultidata[k].length = 8;
        names[k - 1] = "yolodata" + std::to_string(k);
        pluginMultidata[k].name = names[k - 1].c_str();
        pluginMultidata[k].type = PluginFieldType::kFLOAT32;
    }
    PluginFieldCollection pluginData;
    pluginData.nbFields = 4;
    pluginData.fields = pluginMultidata;
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", &pluginData);
    ITensor* inputTensors_yolo[] = { det2->getOutput(0), det1->getOutput(0), det0->getOutput(0) };
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);
    return yolo;
}
#endif

