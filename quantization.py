import argparse
import datetime
import random
import objsize
import time
from pathlib import Path
from crowd_datasets import SHHA
import torch
import torchvision.transforms as standard_transforms
import numpy as np
import torchvision as tv
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import cv2
import glob
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
import torch.onnx
import tensorflow as tf 
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantizationMode, quantize_dynamic, QuantType, quantize_static, QuantFormat, CalibrationMethod, CalibrationDataReader
from onnx2pytorch import ConvertModel


warnings.filterwarnings('ignore')
preprocess = tv.transforms.Compose([
    tv.transforms.Resize(256),
    tv.transforms.CenterCrop(224),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class ImageNetValDataset(torch.utils.data.Dataset):
  def __init__(self, img_dir, transform=None, target_transform=None):
      self.img_dir = img_dir
      self.img_paths = sorted(glob.glob(img_dir + "*/*.JPEG"), key=lambda x: int(x.split("_")[-1].split(".")[0]))
      self.transform = transform
      self.target_transform = target_transform

  def __len__(self):
      return len(self.img_paths)

  def __getitem__(self, idx):
      img_path = self.img_paths[idx]
      image = Image.open(img_path).convert('RGB')
      width, height = image.size
      new_width = width // 128 * 128
      new_height = height // 128 * 128
      #image = image.resize((new_width, new_height), Image.LANCZOS)
      #image = image.resize((896, 384), Image.LANCZOS)
      image = image.resize((256, 128), Image.LANCZOS)
        # pre-proccessing
      
      #image = torch.Tensor(image).unsqueeze(0)
      
      synset = img_path.split("/")[-2]
      label = synset
      if self.transform:
          image = self.transform(image)
      if self.target_transform:
          label = self.target_transform(label)
      return image, label

ds = ImageNetValDataset("./imagenet/val/", transform= transform)
print(ds)
offset = 20
calib_ds = torch.utils.data.Subset(ds, list(range(offset)))
print(len(calib_ds))



class QuntizationDataReader(CalibrationDataReader):
    def __init__(self, torch_ds, batch_size, input_name):

        self.torch_dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, shuffle=False)

        self.input_name = input_name
        self.datasize = len(self.torch_dl)
        print(self.datasize)

        self.enum_data = iter(self.torch_dl)
        print(self.enum_data)
    def to_numpy(self, pt_tensor):
        return pt_tensor.detach().cpu().numpy() if pt_tensor.requires_grad else pt_tensor.cpu().numpy()

    def get_next(self):
        batch = next(self.enum_data, None)
        
        
        if batch is not None:
          return {self.input_name: self.to_numpy(batch[0])}
        else:
          return None

    def rewind(self):
        self.enum_data = iter(self.torch_dl)




def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=1, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=1, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def main(args, debug=False):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    #device = torch.device('cuda')
    device = torch.device('cpu')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        print(checkpoint)
        model.load_state_dict(checkpoint)
    model.eval()
    
    # create the pre-processing transform
    

    # set your image path here
    img_path = "./vis/labeled/374323731_d8771f442d_z_crop0.jpg"
    # load the images
    img_raw = Image.open(img_path).convert('RGB')
    # round the size
    width, height = img_raw.size
    print("width :", width)
    print("height :", height)
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    print("new width :", new_width)
    print("new height :", new_height)
    #img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
    img_raw = img_raw.resize((256, 128), Image.LANCZOS)
    #img_raw = img_raw.resize((new_width, new_height), Image.LANCZOS)
    print(img_raw)
    # pre-proccessing
    img = transform(img_raw)

    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)
    
    
    #torch.onnx.export(model, samples, "P2Pgrape.onnx" , verbose=True, input_names = ['input'], dynamic_axes={"input": {0: "batch"}} )
    #ort.quantization.shape_inference.quant_pre_process("P2Ptotf.onnx", "P2Pquant.onnx", skip_symbolic_shape=False)
    
    #ort_sess = ort.InferenceSession( "P2Pgrape.onnx")
    """
    qdr = QuntizationDataReader(calib_ds, batch_size=2, input_name=ort_sess.get_inputs()[0].name)  
    
    quantize_static(
    "P2Pgrape.onnx",
    "P2Pgrapequant.onnx",
    calibration_data_reader= qdr,
    quant_format=QuantFormat.QDQ,
    op_types_to_quantize=None,
    per_channel=False,
    reduce_range=False,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    nodes_to_quantize=None,
    nodes_to_exclude=None,
    use_external_data_format=False,
    calibrate_method=CalibrationMethod.MinMax,
    extra_options=None,
)
    """
    ort_session2 = ort.InferenceSession("./P2Pgrapequant.onnx")
    
    
    
    #model_int8 = torch.quantization.quantize_dynamic(model,  {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
    # convert to eval mode
    #model.eval()
    
    f=print_size_of_model(model,"fp32")
    #q=print_size_of_model(quantized_model,"int8")
    #print("{0:.2f} times smaller".format(f/q))
    
    
    print(samples.shape)
    # run inference
    start = time.time()
    outputs = ort_session2.run(None, {"input": samples.numpy()})
    #outputs = model(samples)
    
    #print(torch.tensor(outputs).shape)
    #print(outputs['pred_logits'].shape)
    #print(outputs['pred_points'].shape)
    print("total time :", time.time()-start)
    #outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
    outputs_scores = torch.nn.functional.softmax(torch.tensor(outputs[0]), -1)[:, :, 1][0]
    
    #outputs_points = outputs['pred_points'][0]
    outputs_points = torch.tensor(outputs[1][0])
    
    
    threshold = 0.5
    # filter the predictions
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    #outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    #outputs_points = outputs['pred_points'][0]
    # draw the predictions
    size = 2
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    # save the visualized image
    cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(predict_cnt)), img_to_draw)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)