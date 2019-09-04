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

#if HAS_OPENCL

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

#define IN_LIBDECT
#include "libdect.h"

static cl::Context *context;
static cl::Program *program;
static cl::Kernel *kernel;
static cl::CommandQueue *queue;
static cl::Device device;
static std::vector<cl::Device> devices;

static int is_init = 0;
static int use_double = 1;
static libdect_output_type _otype = libdect_output_type::u8;

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

int opencl_init(int platform, int enhanced, int use_single_fp,
	libdect_output_type otype)
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

	std::string f8_kern = std::string("#define FPTYPE float\n#define OTYPE uchar\n#define OTYPE_MAX 255.0\n").append(ks);
	std::string f16_kern = std::string("#define FPTYPE float\n#define OTYPE ushort\n#define OTYPE_MAX 65535.0\n").append(ks);
	std::string d8_kern = std::string("#define FPTYPE double\n#define OTYPE uchar\n#define OTYPE_MAX 255.0\n").append(ks);
	std::string d16_kern = std::string("#define FPTYPE double\n#define OTYPE ushort\n#define OTYPE_MAX 65535.0\n").append(ks);
	std::string ff32_kern = std::string("#define FPTYPE float\n#define OTYPE float\n#define OTYPE_MAX 1.0\n#define FLOOR_FUNC \n").append(ks);
	std::string df32_kern = std::string("#define FPTYPE double\n#define OTYPE float\n#define OTYPE_MAX 1.0\n#define FLOOR_FUNC \n").append(ks);
	std::string ff64_kern = std::string("#define FPTYPE float\n#define OTYPE double\n#define OTYPE_MAX 1.0\n#define FLOOR_FUNC \n").append(ks);
	std::string df64_kern = std::string("#define FPTYPE double\n#define OTYPE double\n#define OTYPE_MAX 1.0\n#define FLOOR_FUNC \n").append(ks);
	_otype = otype;

	if (use_single_fp)
		err = CL_BUILD_ERROR;	// force attempt to use single fp
	else
	{
		const char *cstr;
		size_t len;

		switch (otype)
		{
			case libdect_output_type::u8:
				cstr = d8_kern.c_str();
				len = d8_kern.length();
				break;
			case libdect_output_type::u16:
				cstr = d16_kern.c_str();
				len = d16_kern.length();
				break;
			case libdect_output_type::f32:
				cstr = df32_kern.c_str();
				len = df32_kern.length();
				break;
			case libdect_output_type::f64:
				cstr = df64_kern.c_str();
				len = df64_kern.length();
				break;
			default:
				return -1;
		}

		cl::Program::Sources source(
			1,
			std::make_pair(cstr, len + 1));

		program = new cl::Program(*context, source);
		err = program->build(devices, "");
	}

	if (err != CL_SUCCESS)
	{
		const char *cstr;
		size_t len;

		switch (otype)
		{
		case libdect_output_type::u8:
			cstr = f8_kern.c_str();
			len = f8_kern.length();
			break;
		case libdect_output_type::u16:
			cstr = f16_kern.c_str();
			len = f16_kern.length();
			break;
		case libdect_output_type::f32:
			cstr = ff32_kern.c_str();
			len = ff32_kern.length();
			break;
		case libdect_output_type::f64:
			cstr = ff64_kern.c_str();
			len = ff64_kern.length();
			break;
		default:
			return -1;
		}
		cl::Program::Sources source2(
			1,
			std::make_pair(cstr, len + 1));

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

	if (use_double == 0 && use_single_fp == 0)
	{
		printf("Warning: no double precision support in OpenCL device - defaulting to single\n");
	}

	/*cl_uint native_double_width = device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE>();
	if (native_double_width == 0) {
		printf("No double precision support in OpenCL device.\n");
		return -1;
	}*/

	is_init = 1;
	return 0;
}

static inline cl_int set_float_arg(cl::Kernel *kernel,
	cl_uint index, float val)
{
	if (use_double)
		return kernel->setArg(index, (double)val);
	else
		return kernel->setArg(index, (float)val);
}

int dect_algo_opencl(int enhanced,
	const int16_t *a, const int16_t *b,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	void *x, void *y, void *z,
	size_t pix_count,
	float min_step,
	int16_t *m,
	float mr,
	int idx_adjust)
{
	cl_int err;

	if (is_init == 0)
		return -1;

	auto out_size = pix_count;
	switch (_otype)
	{
	case libdect_output_type::u16:
		out_size *= 2;
		break;
	case libdect_output_type::f32:
		out_size *= 4;
		break;
	case libdect_output_type::f64:
		out_size *= 8;
		break;
	}

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
		m ? pix_count * 2 : sizeof(cl_mem),
		NULL,
		&err);
	checkErr(err, "Buffer::Buffer()");

	/* Input buffers */
	cl::Buffer ina(
		*context,
		CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		pix_count * 2,
		(void*)a,
		&err);
	cl::Buffer inb(
		*context,
		CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		pix_count * 2,
		(void*)b,
		&err);

	err = kernel->setArg(0, ina);
	checkErr(err, "Kernel::setArg(0)");
	err = kernel->setArg(1, inb);
	checkErr(err, "Kernel::setArg(1)");
	err = set_float_arg(kernel, 2, alphaa);
	checkErr(err, "Kernel::setArg(2)");
	err = set_float_arg(kernel, 3, betaa);
	checkErr(err, "Kernel::setArg(3)");
	err = set_float_arg(kernel, 4, gammaa);
	checkErr(err, "Kernel::setArg(4)");
	err = set_float_arg(kernel, 5, alphab);
	checkErr(err, "Kernel::setArg(5)");
	err = set_float_arg(kernel, 6, betab);
	checkErr(err, "Kernel::setArg(6)");
	err = set_float_arg(kernel, 7, gammab);
	checkErr(err, "Kernel::setArg(7)");
	err = kernel->setArg(8, outx);
	checkErr(err, "Kernel::setArg(8)");
	err = kernel->setArg(9, outy);
	checkErr(err, "Kernel::setArg(9)");
	err = kernel->setArg(10, outz);
	checkErr(err, "Kernel::setArg(10)");
	err = set_float_arg(kernel, 11, min_step);
	checkErr(err, "Kernel::setArg(11)");
	err = kernel->setArg(12, outm);
	checkErr(err, "Kernel::setArg(12)");
	err = set_float_arg(kernel, 13, mr);
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
		cl::NDRange(pix_count),
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
			pix_count * 2,
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

#endif /* HAS_OPENCL */
