import torch
import torch.onnx
import onnx
import sys
import os
uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(uer_dir)
sys.path.append(uer_dir)
from adv_fine_tuning_cls import SpanBERT

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 自动初始化CUDA
import numpy as np
from adv_fine_tuning_cls import ETBERTDataset, prep_dataloader

batch_size = 32
val_x_path = r'D:\dataset_packet\ios_dataset_split\application_classification\val\x_bert.pickle'
val_y_path = r'D:\dataset_packet\ios_dataset_split\application_classification\val\y.npy'

mode = 'fp16'


class CalibDataLoader:
    def __init__(self, data, batch_size, calib_count):
        """Summary of __init__.
        
        Args:
            data (Any): Description.
            batch_size (Any): Description.
            calib_count (Any): Description.
        """
        self.data = data
        self.index = 0
        self.batch_size = batch_size
        self.calib_count = calib_count
        self.data1 = np.zeros((self.batch_size, 2, 256))
        self.data2 = np.zeros((self.batch_size, 1, 256))

    def reset(self):
        """Summary of reset.
        """
        self.index = 0

    def next_batch(self):
        """Summary of next_batch.
        
        Returns:
            Any: Description.
        """
        if self.index < self.calib_count:
            for i in range(self.batch_size):
                self.data1[i], self.data2[i], _ = self.data.__getitem__(i + self.index * self.batch_size)
            self.index += 1
            return self.data1, self.data2
        else:
            return np.array([])

    def __len__(self):
        """Summary of __len__.
        
        Returns:
            Any: Description.
        """
        return self.calib_count


class MyCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, dataloader, batch_size):
        """Summary of __init__.
        
        Args:
            dataloader (Any): Description.
            batch_size (Any): Description.
        """
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.cache_file = 'cachefile.txt'

    def get_batch_size(self):
        """Summary of get_batch_size.
        
        Returns:
            Any: Description.
        """
        return self.batch_size

    def get_batch(self, names):
        """Summary of get_batch.
        
        Args:
            names (Any): Description.
        Returns:
            Any: Description.
        """
        etbert_data, input3 = self.dataloader.next_batch()
        if not input3.size:
            return None
        input1 = etbert_data[0]  # 第一个输入
        input2 = etbert_data[1]  # 第二个输入

        return input1, input2, input3  # 返回多个输入

    def read_calibration_cache(self):
        # 如果校准表文件存在则直接从其中读取校准表
        """Summary of read_calibration_cache.
        
        Returns:
            Any: Description.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        # 如果进行了校准，则把校准表写入文件中以便下次使用
        """Summary of write_calibration_cache.
        
        Args:
            cache (Any): Description.
        """
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()

def gen_onnx():
    # 假设你已经有了一个训练好的 PyTorch 模型
    # model = torch.load(r'D:\project\ET-BERT\ET-BERT-main\packet_adv\model\ios.ckpt')  # 加载训练好的 PyTorch 模型
    """Summary of gen_onnx.
    """
    model = SpanBERT.load_from_checkpoint(r'D:\project\ET-BERT\ET-BERT-main\packet_adv\model\ios.ckpt')  # 加载训练好的 PyTorch 模型
    model.eval()  # 设置为评估模式
    model.to('cpu')

    batch = 8
# TensorRT生成引擎的函数
    # 生成一个虚拟输入，用于导出 ONNX 格式
    dummy_input1 = torch.randint(0, 60000, (batch, 256))  # , dtype=torch.long
    dummy_input2 = torch.randint(0, 2, (batch, 256))
    dummy_input3 = torch.randint(0, 60000, (batch, 256))
    dummy_input = (dummy_input1, dummy_input2, dummy_input3, 'test')

    # 转换为 ONNX 格式
    onnx_filename = r'D:\project\ET-BERT\ET-BERT-main\packet_adv\model\ios_batch8.onnx'
    torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3, 'test'), onnx_filename, input_names=["input1", "input2", "input3", "input4"],  output_names=["output1", "output2"], verbose=True)
    # torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3, 'test'), onnx_filename, input_names=["input1", "input2", "input3", "input4"],  output_names=["output1", "output2"], dynamic_axes={"input1": {0: "batch_size"}, "input2": {0: "batch_size"},"input3": {0: "batch_size"},"output1": {0: "batch_size"}, "output2": {0: "batch_size"}}, verbose=True)

    # 检查导出的 ONNX 模型
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)  # 验证模型是否合法
    print("ONNX model is valid.")


def gen_TensorRT():
    """Summary of gen_TensorRT.
    """
    int8_mode = True
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    onnx_model_path = r'D:\project\ET-BERT\ET-BERT-main\packet_adv\model\ios_batch8.onnx'
    onnx_model = onnx.load(onnx_model_path)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    onnx_parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_model_path, 'rb') as f:
        if not onnx_parser.parse(f.read()):
            print("Failed to parse ONNX model.")
            for error in range(onnx_parser.num_errors):
                print(onnx_parser.get_error(error))
        else:
            print("ONNX model parsed successfully.")

    config = builder.create_builder_config()

    # # **启用动态 Batch**
    # profile = builder.create_optimization_profile()
    #
    # # 设定动态 batch 范围 (1, 16, 32) 适用于你的模型
    # for i in range(network.num_inputs):
    #     input_tensor = network.get_input(i)
    #     shape = input_tensor.shape  # 原始输入形状，例如 (1, C, H, W)
    #
    #     if shape[0] == -1:  # 确保 batch 维度是动态的
    #         print(f"Setting dynamic batch for input {input_tensor.name}")
    #
    #     min_shape = (1,) + tuple(shape[1:])  # batch 最小 1
    #     opt_shape = (32,) + tuple(shape[1:])  # 最优 batch 16
    #     max_shape = (512,) + tuple(shape[1:])  # 最大 batch 32
    #     profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    #
    # config.add_optimization_profile(profile)

    if mode == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        dataset = ETBERTDataset(val_x_path, val_y_path, 'val')
        dataloader = CalibDataLoader(dataset, batch_size, 3000)
        calibrator = MyCalibrator(dataloader, batch_size=batch_size)
        config.int8_calibrator = calibrator
    elif mode == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_engine_with_config(network, config)

    with open(r"D:\project\ET-BERT\ET-BERT-main\packet_adv\model\ios_fp16_batch8.trt", "wb") as f:
        f.write(engine.serialize())

    print("TensorRT engine saved")


if __name__ == '__main__':
    # dataset = ETBERTDataset(val_x_path, val_y_path, 'val')
    # data = CalibDataLoader(dataset, batch_size, 256)
    gen_onnx()
    gen_TensorRT()

