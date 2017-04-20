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
#include <sstream>

#ifdef _MSC_VER
#include <tchar.h>
#include <Windows.h>
#endif

#include "../libdect/dect.cl"

static cl::Context *context;
static cl::Program *program;
static cl::Kernel *kernel;
static cl::CommandQueue *queue;
static cl::Device device;
static std::vector<cl::Device> devices;

static int is_init = 0;
static int use_double = 1;

#define checkErr(err, name) \
	if ((err) != CL_SUCCESS) { \
		std::cerr << "ERROR: " << (name) << " (" << (err) << ")" << std::endl; \
		return(err); \
	} \

#define checkErr2(err, name) \
	if ((err) != CL_SUCCESS) { \
		std::cerr << "ERROR: " << (name) << " (" << (err) << ")" << std::endl; \
		return(NULL); \
	} \


int opencl_dump_platforms()
{
	std::vector< cl::Platform > platformList;
	cl::Platform::get(&platformList);
	checkErr(platformList.size() != 0 ? CL_SUCCESS : -1, "cl::Platform::get");
	//std::cerr << "Platform number is: " << platformList.size() << std::endl;

	int id = 2;
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

int opencl_get_device_count()
{
	std::vector< cl::Platform > platformList;
	cl::Platform::get(&platformList);
	return platformList.size();
}

const char *opencl_get_device_name(int idx)
{
	std::vector< cl::Platform > platformList;
	cl::Platform::get(&platformList);
	checkErr2(platformList.size() != 0 ? CL_SUCCESS : -1, "cl::Platform::get");
	//std::cerr << "Platform number is: " << platformList.size() << std::endl;

	int id = 0;
	for (auto it = platformList.begin(); it < platformList.end(); it++, id++)
	{
		if (idx == id)
		{
			std::string platformVendor;
			std::string platformName;
			it->getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
			it->getInfo((cl_platform_info)CL_PLATFORM_NAME, &platformName);

			std::stringstream ss;
			ss << "OpenCL " << platformName << " (" << platformVendor << ")" << std::endl;

			auto str = ss.str();

			char *ret = new char[str.size() + 1];
			std::copy(str.begin(), str.end(), ret);
			ret[str.size()] = '\0'; // don't forget the terminating 0
			return ret;
		}
	}

	return NULL;
}

int opencl_init(int platform, int enhanced)
{
	cl_int err;
	is_init = 0;

	std::vector<cl::Platform> platformList;
	cl::Platform::get(&platformList);
	checkErr(platformList.size() != 0 ? CL_SUCCESS : -1, "cl::Platform::get");

	checkErr(platform < (int)platformList.size() ? CL_SUCCESS : -1, "invalid platform");

	cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[platform])(), 0 };

	context = new cl::Context(
		CL_DEVICE_TYPE_GPU,
		cprops,
		NULL,
		NULL,
		&err);

	if (err != CL_SUCCESS)
	{
		context = new cl::Context(
			CL_DEVICE_TYPE_DEFAULT,
			cprops,
			NULL,
			NULL,
			&err);
	}

	checkErr(err, "Context::Context()");

	devices = context->getInfo<CL_CONTEXT_DEVICES>();
	checkErr(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");

	cl::Program::Sources source(
		1,
		std::make_pair(ks.c_str(), ks.length() + 1));
	
	program = new cl::Program(*context, source);
	err = program->build(devices, "");

	if (err != CL_SUCCESS)
	{
		auto new_ks = std::string("#define double float\n\n").append(ks);
		cl::Program::Sources source2(
			1,
			std::make_pair(new_ks.c_str(), new_ks.length() + 1));

		program = new cl::Program(*context, source2);
		err = program->build(devices, "");
		use_double = 0;
	}
	checkErr(err, "Program::build()");

	kernel = new cl::Kernel(*program, enhanced == 3 ? "dect2" : "dect", &err);
	checkErr(err, "Kernel::Kernel()");

	queue = new cl::CommandQueue(*context, devices[0], 0, &err);
	checkErr(err, "CommandQueue::CommandQueue()");
	
	device = devices[0];

	if (use_double == 0)
	{
		printf("Warning: no double precision support in OpenCL device - potential for lack of accuracy\n");
	}

	/*cl_uint native_double_width = device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE>();
	if (native_double_width == 0) {
		printf("No double precision support in OpenCL device.\n");
		return -1;
	}*/

	is_init = 1;
	return 0;
}

int dect_algo_opencl(int enhanced,
	const int16_t *a, const int16_t *b,
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

	if (is_init == 0)
		return -1;

	/* Build output buffers */
	cl::Buffer outx(
		*context,
		CL_MEM_WRITE_ONLY,
		out_size,
		NULL,
		&err);
	checkErr(err, "Buffer::Buffer()");
	cl::Buffer outy(
		*context,
		CL_MEM_WRITE_ONLY,
		out_size,
		NULL,
		&err);
	checkErr(err, "Buffer::Buffer()");
	cl::Buffer outz(
		*context,
		CL_MEM_WRITE_ONLY,
		out_size,
		NULL,
		&err);
	checkErr(err, "Buffer::Buffer()");

	int dummy_buf[4];
	cl::Buffer outm(
		*context,
		CL_MEM_WRITE_ONLY,
		m ? out_size * 2 : sizeof(cl_mem),
		NULL,
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
	err = use_double ? kernel->setArg(2, (double)alphaa) : kernel->setArg(2, (float)alphaa);
	checkErr(err, "Kernel::setArg(2)");
	err = use_double ? kernel->setArg(3, (double)betaa) : kernel->setArg(3, (float)betaa);
	checkErr(err, "Kernel::setArg(3)");
	err = use_double ? kernel->setArg(4, (double)gammaa) : kernel->setArg(4, (float)gammaa);
	checkErr(err, "Kernel::setArg(4)");
	err = use_double ? kernel->setArg(5, (double)alphab) : kernel->setArg(5, (float)alphab);
	checkErr(err, "Kernel::setArg(5)");
	err = use_double ? kernel->setArg(6, (double)betab) : kernel->setArg(6, (float)betab);
	checkErr(err, "Kernel::setArg(6)");
	err = use_double ? kernel->setArg(7, (double)gammab) : kernel->setArg(7, (float)gammab);
	checkErr(err, "Kernel::setArg(7)");
	err = kernel->setArg(8, outx);
	checkErr(err, "Kernel::setArg(8)");
	err = kernel->setArg(9, outy);
	checkErr(err, "Kernel::setArg(9)");
	err = kernel->setArg(10, outz);
	checkErr(err, "Kernel::setArg(10)");
	err = use_double ? kernel->setArg(11, (double)min_step) : kernel->setArg(11, (float)min_step);
	checkErr(err, "Kernel::setArg(11)");
	err = kernel->setArg(12, outm);
	checkErr(err, "Kernel::setArg(12)");
	err = use_double ? kernel->setArg(13, (double)mr) : kernel->setArg(13, (float)min_step);
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
		cl::NullRange,
		NULL,
		&event);
	checkErr(err, "CommandQueue::enqueueNDRangeKernel()");

	/* Wait for completion */
	event.wait();

	/* Get output buffers */
	cl::Event eventx, eventy, eventz, eventm;
	err = queue->enqueueReadBuffer(
		outx,
		CL_FALSE,
		0,
		out_size,
		x,
		NULL,
		&eventx);
	checkErr(err, "CommandQueue::enqueueReadBuffer()");

	err = queue->enqueueReadBuffer(
		outy,
		CL_FALSE,
		0,
		out_size,
		y,
		NULL,
		&eventy);
	checkErr(err, "CommandQueue::enqueueReadBuffer()");

	err = queue->enqueueReadBuffer(
		outz,
		CL_FALSE,
		0,
		out_size,
		z,
		NULL,
		&eventz);
	checkErr(err, "CommandQueue::enqueueReadBuffer()");

	if (m)
	{
		err = queue->enqueueReadBuffer(
			outm,
			CL_FALSE,
			0,
			out_size * 2,
			m,
			NULL,
			&eventm);
		checkErr(err, "CommandQueue::enqueueReadBuffer()");
	}

	// Need to wait for copying completion prior to
	//  returning to main as next thing will be libtiff
	//  writing these buffers
	eventx.wait();
	eventy.wait();
	eventz.wait();
	if (m)
		eventm.wait();

	return 0;
}
