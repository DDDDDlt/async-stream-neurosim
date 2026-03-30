#from modules.quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN
import os
import torch.nn as nn
import shutil
from modules.quantization_cpu_np_infer import QConv2d,QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
import numpy as np
import torch
from utee import wage_quantizer
from utee import float_quantizer
from utee import async_stream


ASYNC_STREAM = False
ASYNC_STREAM_FREQUENCY = 1e4
ASYNC_STREAM_WINDOW = 32.0
ASYNC_STREAM_JITTER = 0.02
ASYNC_STREAM_ENERGY_PER_PULSE = 1.0

def Neural_Sim(self, input, output): 
    global model_n, FP, ASYNC_STREAM, ASYNC_STREAM_FREQUENCY, ASYNC_STREAM_WINDOW, ASYNC_STREAM_JITTER

    print("quantize layer ", self.name)
    input_file_name =  './layer_record_' + str(model_n) + '/input' + str(self.name) + '.csv'
    weight_file_name =  './layer_record_' + str(model_n) + '/weight' + str(self.name) + '.csv'
    metadata_file_name = './layer_record_' + str(model_n) + '/meta' + str(self.name) + '.txt'
    f = open('./layer_record_' + str(model_n) + '/trace_command.sh', "a")
    f.write(weight_file_name+' '+input_file_name+' '+metadata_file_name+' ')
    if FP:
        weight_q = float_quantizer.float_range_quantize(self.weight,self.wl_weight)
    else:
        weight_q = wage_quantizer.Q(self.weight,self.wl_weight)
    write_matrix_weight( weight_q.cpu().data.numpy(),weight_file_name)

    if hasattr(self, 'last_stream_stats') and self.last_stream_stats is not None:
        stats = self.last_stream_stats
    else:
        stats = async_stream.compute_stream_stats(
            input[0], ASYNC_STREAM_FREQUENCY, ASYNC_STREAM_WINDOW, ASYNC_STREAM_JITTER
        )
    async_stream.write_stream_metadata(metadata_file_name, stats)

    if len(self.weight.shape) > 2:
        k=self.weight.shape[-1]
        padding = self.padding
        stride = self.stride  
        write_matrix_activation_conv(stretch_input(input[0].cpu().data.numpy(),k,padding,stride),None,self.wl_input,input_file_name)
    else:
        write_matrix_activation_fc(input[0].cpu().data.numpy(),None ,self.wl_input, input_file_name)

def write_matrix_weight(input_matrix,filename):
    cout = input_matrix.shape[0]
    weight_matrix = input_matrix.reshape(cout,-1).transpose()
    np.savetxt(filename, weight_matrix, delimiter=",",fmt='%10.5f')


def write_matrix_activation_conv(input_matrix,fill_dimension,length,filename):
    filled_matrix_b = np.zeros([input_matrix.shape[2],input_matrix.shape[1]*length],dtype=str)
    filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length)
    for i,b in enumerate(filled_matrix_bin):
        filled_matrix_b[:,i::length] =  b.transpose()
    np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')


def write_matrix_activation_fc(input_matrix,fill_dimension,length,filename):

    filled_matrix_b = np.zeros([input_matrix.shape[1],length],dtype=str)
    filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length)
    for i,b in enumerate(filled_matrix_bin):
        filled_matrix_b[:,i] =  b
    np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')


def stretch_input(input_matrix,window_size = 5,padding=(0,0),stride=(1,1)):
    input_shape = input_matrix.shape
    output_shape_row = int((input_shape[2] + 2*padding[0] -window_size) / stride[0] + 1)
    output_shape_col = int((input_shape[3] + 2*padding[1] -window_size) / stride[1] + 1)
    item_num = int(output_shape_row * output_shape_col)
    output_matrix = np.zeros((input_shape[0],item_num,input_shape[1]*window_size*window_size))
    iter = 0
    if (padding[0] != 0):
        input_tmp = np.zeros((input_shape[0], input_shape[1], input_shape[2] + padding[0]*2, input_shape[3] + padding[1] *2))
        input_tmp[:, :, padding[0]: -padding[0], padding[1]: -padding[1]] = input_matrix
        input_matrix = input_tmp
    for i in range(output_shape_row):
        for j in range(output_shape_col):
            for b in range(input_shape[0]):
                output_matrix[b,iter,:] = input_matrix[b, :, i*stride[0]:i*stride[0]+window_size,j*stride[1]:j*stride[1]+window_size].reshape(input_shape[1]*window_size*window_size)
            iter += 1

    return output_matrix


def dec2bin(x,n):
    y = x.copy()
    out = []
    scale_list = []
    delta = 1.0/(2**(n-1))
    x_int = x/delta

    base = 2**(n-1)

    y[x_int>=0] = 0
    y[x_int< 0] = 1
    rest = x_int + base*y
    out.append(y.copy())
    scale_list.append(-base*delta)
    for i in range(n-1):
        base = base/2
        y[rest>=base] = 1
        y[rest<base]  = 0
        rest = rest - base * y
        out.append(y.copy())
        scale_list.append(base * delta)

    return out,scale_list

def bin2dec(x,n):
    bit = x.pop(0)
    base = 2**(n-1)
    delta = 1.0/(2**(n-1))
    y = -bit*base
    base = base/2
    for bit in x:
        y = y+base*bit
        base= base/2
    out = y*delta
    return out

def remove_hook_list(hook_handle_list):
    for handle in hook_handle_list:
        handle.remove()

def hardware_evaluation(model, wl_weight, wl_activation, subArray, parallelRead, model_name, mode,
                        async_stream_mode=False, stream_frequency=1e4, stream_window=32.0,
                        stream_jitter=0.02, stream_energy_per_pulse=1.0):
    global model_n, FP, ASYNC_STREAM, ASYNC_STREAM_FREQUENCY, ASYNC_STREAM_WINDOW, ASYNC_STREAM_JITTER, ASYNC_STREAM_ENERGY_PER_PULSE
    model_n = model_name
    FP = 1 if mode in ('FP', 'ASYNC') else 0
    ASYNC_STREAM = async_stream_mode
    ASYNC_STREAM_FREQUENCY = stream_frequency
    ASYNC_STREAM_WINDOW = stream_window
    ASYNC_STREAM_JITTER = stream_jitter
    ASYNC_STREAM_ENERGY_PER_PULSE = stream_energy_per_pulse
    if model_name == 'StreamCNN' and subArray > 32:
        subArray = 32
        parallelRead = min(parallelRead, subArray)
    
    hook_handle_list = []
    if not os.path.exists('./layer_record_'+str(model_name)):
        os.makedirs('./layer_record_'+str(model_name))
    if os.path.exists('./layer_record_'+str(model_name)+'/trace_command.sh'):
        os.remove('./layer_record_'+str(model_name)+'/trace_command.sh')
    f = open('./layer_record_'+str(model_name)+'/trace_command.sh', "w")
    f.write(
        './NeuroSIM/main ./NeuroSIM/NetWork_'+str(model_name)+'.csv '
        + str(wl_weight)+' '+str(wl_activation)+' '+str(subArray)+' '+str(parallelRead)+' '
        + str(1 if async_stream_mode else 0)+' '+str(stream_frequency)+' '+str(stream_window)+' '
        + str(stream_jitter)+' '+str(stream_energy_per_pulse)+' '
    )
    
    for i, layer in enumerate(model.modules()):
        if isinstance(layer, (FConv2d, QConv2d, nn.Conv2d)) or isinstance(layer, (FLinear, QLinear, nn.Linear)):
            hook_handle_list.append(layer.register_forward_hook(Neural_Sim))
    return hook_handle_list
