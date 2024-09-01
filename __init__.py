from .nodes.renoise_sampler_node import ReNoiseSamplerNode
from .nodes.renoise_model_pred_node import ReNoiseModelSamplingPredNode


NODE_CLASS_MAPPINGS = {
    "ReNoiseSampler": ReNoiseSamplerNode,
    "ReNoiseModelSamplingPred": ReNoiseModelSamplingPredNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReNoiseSampler": "ReNoise Sampler",
    "ReNoiseModelSamplingPred": "ReNoise Model Pred"
}
