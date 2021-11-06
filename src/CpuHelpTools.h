#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

void print(std::vector<cv::Mat> input);
void print(std::vector<std::vector<cv::Mat>> input);


void read(std::vector<std::vector<cv::Mat>>& inputs, std::vector<int>& classes);

void prepareData(std::vector<std::vector<cv::Mat>>& inputs);

std::vector<cv::Mat> CreateCube(int chanels, int width, int height);
std::vector<std::vector<cv::Mat>> CreateTensor(int numberOfCubes, int chanels, int width, int height);