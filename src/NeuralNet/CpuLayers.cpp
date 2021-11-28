#include "NeuralNet/CpuLayers.h"
#include <numeric>

namespace cpu
{

	std::vector<cv::Mat> SoftMax::compute(std::vector<cv::Mat>& rawInput)
	{
		//std::cout << "SoftMax\n";
		//print(rawInput);
		cv::Mat input = rawInput[0];
		cv::exp(input, this->output);
		std::cout << "Softmax exp: " << this->output << '\n';
		float norm = cv::sum(this->output)[0];
		std::cout << "Softmax norm: " << norm << '\n';
		output = output / norm;

		return std::vector<cv::Mat>{1, output};
	}

	std::vector<cv::Mat> SoftMax::backPropagate(std::vector<cv::Mat>& rawInput)
	{
		//std::cout << "SoftMax\n";
		//print(rawInput);
		cv::Mat answer = rawInput[0];
		return std::vector<cv::Mat>{1, cv::Mat(output - answer)};
	}

	std::vector<cv::Mat> ReLU::compute(std::vector<cv::Mat>& input)
	{
		//std::cout << "ReLU\n";
		//print(input);
		this->inputs = input;
		std::vector<cv::Mat> result = CreateCube(input.size(), input[0].cols, input[0].rows);

#pragma omp parallel for
		for (int c = 0; c < input.size(); ++c)
		{
			std::transform(input[c].begin<double>(), input[c].end<double>(), result[c].begin<double>(),
				[](double& value) {return value < 0.0f ? 0.0f : value; });
		}

		return result;
	}

	std::vector<cv::Mat> ReLU::backPropagate(std::vector<cv::Mat>& input)
	{
		//std::cout << "ReLU\n";
		//print(input);
		std::vector<cv::Mat> dx;
		std::transform(input.begin(), input.end(), std::back_inserter(dx),
			[](cv::Mat& value) {
				cv::Mat valueCopy;
				value.copyTo(valueCopy);
				return valueCopy;
			});

#pragma omp parallel for
		for (int c = 0; c < dx.size(); ++c)
		{
			for (int w = 0; w < dx[c].cols; ++w)
			{
				for (int h = 0; h < dx[c].rows; ++h)
				{
					if (this->inputs[c].at<double>(h, w) < 0.0)
					{
						dx[c].at<double>(h, w) = 0;
					}
				}
			}
		}

		return dx;
	}

	FullyConnected::FullyConnected(int numInputs, int numOutput, float lambda) :
		lambda{ lambda },
		weights(numInputs, numOutput, CV_64FC1)
	{
		cv::RNG rng(23);
		rng.fill(weights, cv::RNG::UNIFORM, 0, 0.01);
		bias = cv::Mat::zeros(1, numOutput, CV_64FC1);
	}

	std::vector<cv::Mat> FullyConnected::compute(std::vector<cv::Mat>& input)
	{
		this->inputs = input[0];
		cv::Mat result = this->inputs * this->weights + bias;
		return std::vector<cv::Mat>(1, result);
	}

	std::vector<cv::Mat> FullyConnected::backPropagate(std::vector<cv::Mat>& input)
	{
		cv::Mat dy = input[0];
		if (dy.rows == this->inputs.rows)
		{
			dy = dy.t();
		}

		cv::Mat dw = dy * this->inputs;

		cv::Mat db;
		cv::reduce(dy, db, 1, cv::REDUCE_SUM, CV_64FC1);
		cv::Mat dx = dy.t() * this->weights.t();

		this->weights -= this->lambda * dw.t();
		this->bias -= this->lambda * db.t();

		return std::vector<cv::Mat>(1, dx);
	}


	std::vector<cv::Mat> StretchLayer::compute(std::vector<cv::Mat>& input)
	{
		//std::cout << "StretchLayer\n";
		//print(input);
		this->chanels = input.size();
		this->width = input[0].cols;
		this->height = input[0].rows;

		cv::Mat result(1, width * height * chanels, CV_64FC1);

		int r = 0;
		for (int c = 0; c < this->chanels; ++c)
		{
			for (int h = 0; h < this->height; ++h)
			{
				for (int w = 0; w < this->width; ++w)
				{
					result.at<double>(0, r) = input[c].at<double>(h, w);
					++r;
				}
			}
		}
		return std::vector<cv::Mat>{ 1, result };
	}

	std::vector<cv::Mat> StretchLayer::backPropagate(std::vector<cv::Mat>& rawInput)
	{
		//std::cout << "StretchLayer\n";
		//print(rawInput);
		std::vector<cv::Mat> result = CreateCube(this->chanels, this->width, this->height);
		cv::Mat input = rawInput[0];

		int r = 0;
		for (int c = 0; c < this->chanels; ++c)
		{
			for (int w = 0; w < this->width; ++w)
			{
				for (int h = 0; h < this->height; ++h)
				{
					result[c].at<double>(h, w) = input.at<double>(0, r);
					++r;
				}
			}
		}

		return result;
	}

	ConvLayer::ConvLayer(int inputChanels, int numFilters, int kernelSize, int stride, float lambda, int padding) :
		inputChanels{ inputChanels }, numFilters{ numFilters }, kernelSize{ kernelSize }, stride{ stride }, lambda{ lambda },
		padding{ padding },
		weights(CreateTensor(numFilters, inputChanels, kernelSize, kernelSize)),
		bias(numFilters, 0.0f)
	{
		cv::Mat mean = cv::Mat::zeros(1, 1, CV_64FC1);
		cv::Mat sigma = cv::Mat::ones(1, 1, CV_64FC1) / sqrt(numFilters * kernelSize * kernelSize);
		cv::RNG rng(23);
		for (auto& filter : this->weights)
		{
			for (auto& matrix : filter)
			{
				rng.fill(matrix, cv::RNG::NORMAL, mean, sigma);
			}
		}
	}

	double convolution(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& filter, cv::Rect convPart)
	{
		//std::cout << "\nBegin convolution\n\n";
		double result = 0.0f;
		for (int i = 0; i < filter.size(); ++i)
		{
			result += cv::sum(input[i](convPart).mul(filter[i]))[0];
			//std::cout << "input: \n" << input[i](convPart) << "\n" << "filter: \n" << filter[i] << '\n';
			//std::cout << "result: " << cv::sum(input[i](convPart).mul(filter[i])) << '\n';
		}
		//std::cout << "Convolution result: " << result << '\n';
		//std::cout << "\nEnd convolution\n\n";
		return result;
	}

	std::vector<cv::Mat> ConvLayer::compute(std::vector<cv::Mat>& input)
	{
		int chanels = input.size();
		int width = input[0].cols + 2 * this->padding;
		int height = input[0].rows + 2 * this->padding;

		this->inputs = CreateCube(chanels, width, height);
		for (int i = 0; i < chanels; ++i)
		{
			input[i].copyTo(this->inputs[i](cv::Rect(padding, padding, input[i].cols, input[i].rows)));
		}

		int outputWidth = (width - this->kernelSize) / this->stride + 1;
		int outputHeight = (height - this->kernelSize) / this->stride + 1;

		std::vector<cv::Mat> output = CreateCube(this->numFilters, outputWidth, outputHeight);

		for (int f = 0; f < this->numFilters; ++f)
		{
			for (int w = 0; w < outputWidth; ++w)
			{
				for (int h = 0; h < outputHeight; ++h)
				{
					output[f].at<double>(h, w) = convolution(this->inputs, this->weights[f], cv::Rect(w, h, kernelSize, kernelSize)) + this->bias[f];
				}
			}
		}

		return output;
	}

	std::vector<cv::Mat> ConvLayer::backPropagate(std::vector<cv::Mat>& dy)
	{
		std::vector<cv::Mat> dx = CreateCube(this->inputs.size(), this->inputs[0].rows, this->inputs[0].cols);
		std::vector<std::vector<cv::Mat>> dw = CreateTensor(this->weights.size(), this->weights[0].size(),
			this->weights[0][0].rows, this->weights[0][0].cols);
		std::vector<double> db(this->bias.size(), 0.0);

		int filterSize = dy.size();
		int width = dy[0].rows;
		int height = dy[0].cols;

		for (int f = 0; f < filterSize; ++f)
		{
			for (int w = 0; w < width; ++w)
			{
				for (int h = 0; h < height; ++h)
				{
					// dw[f,:,:,:] += dy[f,w,h] * this->inputs[:,w:w+self.K,h:h+self.K]
					cv::Rect patch(h, w, this->kernelSize, this->kernelSize);
					double coeff = dy[f].at<double>(w, h);
					for (int c = 0; c < this->inputs.size(); ++c)
					{
						dw[f][c] += coeff * this->inputs[c](patch);
					}

					// dx[:,w:w+self.K,h:h+self.K] += dy[f,w,h] * this->weights[f,:,:,:]
					for (int c = 0; c < dx.size(); ++c)
					{
						dx[c](patch) += coeff * this->weights[f][c];
					}
				}
			}

			db[f] = cv::sum(dy[f])[0];
		}

		// this->weights -= this->lambda * dw
		for (int f = 0; f < this->weights.size(); ++f)
		{
			for (int m = 0; m < this->weights[f].size(); ++m)
			{
				/*
				std::cout << "before\n";
				print(this->weights[f][m]);
				std::cout << '\n';
				*/
				this->weights[f][m] -= this->lambda * dw[f][m];
				/*
				std::cout << "after\n";
				print(this->weights[f][m]);
				std::cout << '\n';
				std::cout << "dw\n";
				print(this->lambda * dw[f][m]);
				std::cout << '\n';
				*/
			}
		}
		this->dwMemmory = dw;

		// this->bias -= this->lambda * db
		for (int f = 0; f < this->bias.size(); ++f)
		{
			this->bias[f] -= this->lambda * db[f];
		}

		return dx;
	}
};