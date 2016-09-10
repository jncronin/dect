/* Copyright (C) 2016 by John Cronin
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.

* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/

#include <stdio.h>
#include <utility>

#include <CL/cl.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>

static cl::Context *context;
static cl::Program *program;
static cl::Kernel *kernel;
static cl::CommandQueue *queue;

extern int enhanced;

#define checkErr(err, name) \
	if ((err) != CL_SUCCESS) { \
		std::cerr << "ERROR: " << (name) << " (" << (err) << ")" << std::endl; \
		return(err); \
	} \

int opencl_dump_platforms()
{
	std::vector< cl::Platform > platformList;
	cl::Platform::get(&platformList);
	checkErr(platformList.size() != 0 ? CL_SUCCESS : -1, "cl::Platform::get");
	//std::cerr << "Platform number is: " << platformList.size() << std::endl;

	int id = 1;
	for (auto it = platformList.begin(); it < platformList.end(); it++, id++)
	{
		std::string platformVendor;
		std::string platformName;
		it->getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
		it->getInfo((cl_platform_info)CL_PLATFORM_NAME, &platformName);
		std::cout << " " << id << ": OpenCL " << platformName << " (" << platformVendor << ")" << std::endl;
	}

	return 0;
}

int opencl_init(int platform)
{
	cl_int err;

	std::vector<cl::Platform> platformList;
	cl::Platform::get(&platformList);
	checkErr(platformList.size() != 0 ? CL_SUCCESS : -1, "cl::Platform::get");

	checkErr(platform < (int)platformList.size() ? CL_SUCCESS : -1, "invalid platform");

	cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[platform])(), 0 };

	context = new cl::Context(
		CL_DEVICE_TYPE_DEFAULT,
		cprops,
		NULL,
		NULL,
		&err);

	checkErr(err, "Context::Context()");

	std::vector<cl::Device> devices;
	devices = context->getInfo<CL_CONTEXT_DEVICES>();
	checkErr(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");

	std::ifstream file("dect.cl");
	checkErr(file.is_open() ? CL_SUCCESS : -1, "dect.cl");
	std::string prog(
		std::istreambuf_iterator<char>(file),
		(std::istreambuf_iterator<char>()));
	cl::Program::Sources source(
		1,
		std::make_pair(prog.c_str(), prog.length() + 1));
	
	program = new cl::Program(*context, source);
	err = program->build(devices, "");
	checkErr(err, "Program::build()");

	kernel = new cl::Kernel(*program, enhanced ? "dect2" : "dect", &err);
	checkErr(err, "Kernel::Kernel()");

	queue = new cl::CommandQueue(*context, devices[0], 0, &err);
	checkErr(err, "CommandQueue::CommandQueue()");

	return 0;
}

int dect_algo_opencl(const int16_t *a, const int16_t *b,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	uint8_t *x, uint8_t *y, uint8_t *z,
	size_t out_size,
	float min_step,
	int16_t *m,
	float mr,
	int idx_adjust)
{
	cl_int err;

	/* Build output buffers */
	cl::Buffer outx(
		*context,
		CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		out_size,
		x,
		&err);
	checkErr(err, "Buffer::Buffer()");
	cl::Buffer outy(
		*context,
		CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		out_size,
		y,
		&err);
	checkErr(err, "Buffer::Buffer()");
	cl::Buffer outz(
		*context,
		CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		out_size,
		z,
		&err);
	checkErr(err, "Buffer::Buffer()");

	cl::Buffer outm(
		*context,
		CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		m ? out_size * 2 : sizeof(cl_mem),
		m,
		&err);
	checkErr(err, "Buffer::Buffer()");

	/* Input buffers */
	cl::Buffer ina(
		*context,
		CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		out_size * 2,
		(void*)a,
		&err);
	cl::Buffer inb(
		*context,
		CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		out_size * 2,
		(void*)b,
		&err);

	err = kernel->setArg(0, ina);
	checkErr(err, "Kernel::setArg(0)");
	err = kernel->setArg(1, inb);
	checkErr(err, "Kernel::setArg(1)");
	err = kernel->setArg(2, alphaa);
	checkErr(err, "Kernel::setArg(2)");
	err = kernel->setArg(3, betaa);
	checkErr(err, "Kernel::setArg(3)");
	err = kernel->setArg(4, gammaa);
	checkErr(err, "Kernel::setArg(4)");
	err = kernel->setArg(5, alphab);
	checkErr(err, "Kernel::setArg(5)");
	err = kernel->setArg(6, betab);
	checkErr(err, "Kernel::setArg(6)");
	err = kernel->setArg(7, gammaa);
	checkErr(err, "Kernel::setArg(7)");
	err = kernel->setArg(8, outx);
	checkErr(err, "Kernel::setArg(8)");
	err = kernel->setArg(9, outy);
	checkErr(err, "Kernel::setArg(9)");
	err = kernel->setArg(10, outz);
	checkErr(err, "Kernel::setArg(10)");
	err = kernel->setArg(11, min_step);
	checkErr(err, "Kernel::setArg(11)");
	err = kernel->setArg(12, outm);
	checkErr(err, "Kernel::setArg(12)");
	err = kernel->setArg(13, mr);
	checkErr(err, "Kernel::setArg(13)");
	err = kernel->setArg(14, m ? 1 : 0);
	checkErr(err, "Kernel::setArg(14)");
	err = kernel->setArg(15, idx_adjust);
	checkErr(err, "Kernel::setArg(15)");

	cl::Event event;

	/* Run the kernel */
	err = queue->enqueueNDRangeKernel(
		*kernel,
		cl::NullRange,
		cl::NDRange(out_size),
		cl::NDRange(1, 1),
		NULL,
		&event);
	checkErr(err, "CommandQueue::enqueueNDRangeKernel()");

	/* Wait for completion */
	event.wait();

	/* Get output buffers */
	err = queue->enqueueReadBuffer(
		outx,
		CL_TRUE,
		0,
		out_size,
		x);
	checkErr(err, "ComamndQueue::enqueueReadBuffer()");

	err = queue->enqueueReadBuffer(
		outy,
		CL_TRUE,
		0,
		out_size,
		y);
	checkErr(err, "ComamndQueue::enqueueReadBuffer()");

	err = queue->enqueueReadBuffer(
		outz,
		CL_TRUE,
		0,
		out_size,
		z);
	checkErr(err, "ComamndQueue::enqueueReadBuffer()");

	return 0;
}
