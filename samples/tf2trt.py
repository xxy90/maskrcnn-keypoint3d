
import tensorrt as trt
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
class ModelData(object):
    MODEL_FILE = "graph_opt.uff"
    INPUT_NAME ="image"
    INPUT_SHAPE = ( 3,432, 368)
    OUTPUT_NAME = "Openpose/concat_stage7"


def build_engine(model_file):
    # For more information on TRT basics, refer to the introductory samples.
    engine_file_path='graph_opt.engine'
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size  = 1 << 30
        # Parse the Uff Network
        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output(ModelData.OUTPUT_NAME)
        parser.parse(model_file, network)
        # Build and return an engine.
        engine=builder.build_cuda_engine(network)
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        return engine

if __name__=='__main__':
    build_engine(model_file='graph_opt.uff')