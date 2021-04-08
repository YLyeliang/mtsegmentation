# -*- coding: utf-8 -*- 
# @Time : 2021/4/8 1:51 下午 
# @Author : yl
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)


def build_engine(onnx_path, mode='fp32', dynamic_input=False, calib=None, shape=[1, 3, 224, 224]):
    """
    This is the function to create the TensorRT engine
    Args:
        onnx_path:
        max_batch_size:
        mode:
        calib:
        shape:
    Returns:
    """
    assert mode in ["fp32", "fp16", "int8"]
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network,
                                                                                                  TRT_LOGGER) as parser:
        builder.max_workspace_size = (256 << 20)
        builder.max_batch_size = 1
        if mode == "fp16":
            builder.fp16_mode = True
        elif mode == "int8":
            builder.int8_mode = True
            assert calib is not None
            builder.int8_calibrator = calib
        with onnx_path(onnx_path, "rb") as model:
            parser.parse(model.read())
        if dynamic_input:
            profile = builder.create_optimization_profile()
            profile.set_shape(network.get_input(0).name, (1, 3, 512, 512), (1, 3, 1600, 1600), (1, 3, 1024, 1024))
        engine = builder.build_cuda_engine(network)
        return engine


def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, "wb") as f:
        f.write(buf)


def load_engine(engine_file, TRT_LOGGER):
    with open(engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        return engine
