import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

#input_model_path = '/home/nvidia/tf-pose-estimation/models/graph/self_trained/'
input_model_path = './models/graph/self_trained/'
input_model_name = 'frozen-opt-model.pb'
graph_path = input_model_path + input_model_name

#save_model_path = '/home/nvidia/tf-pose-estimation/models/'
save_model_path = './models/trt_models/'
save_model_name = 'trt_tfopenpose_models_fp16.pb'

output_nodes = ["Openpose/concat_stage7"]
with tf.gfile.GFile(graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())


'''1
frozen_graph_def: frozen TensorFlow graphout
put_node_name:    list of strings with names of output nodes 
                  e.g. ["resnet_v1_50/predictions/Reshape_1"]
max_batch_size:   integer, size of input batch e.g. 16
max_workspace_size_bytes:   integer, maximum GPU memory size available for TensorRT
precision_mode:   string, allowed values FP32, FP16 or INT8
'''
trt_def = trt.create_inference_graph(
    graph_def,
    output_nodes,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 26,
    precision_mode="fp16",     #'FP32', 'FP16', 'INT8', 'fp32', 'fp16', 'int8'
    #minimum_segment_size=50,
    #is_dynamic_op=True,
    #maximum_cached_engines=int(1e3),
    #use_calibration=True,
)

# Save the frozen graph
with open(save_model_path + save_model_name, 'wb') as f:
    f.write(trt_def.SerializeToString())

print(f"File write  to {save_model_path + save_model_name}")

