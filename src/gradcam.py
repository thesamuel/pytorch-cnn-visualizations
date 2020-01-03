from typing import List, Optional

import torch

from torch import nn
from torch.nn import functional as F

import matplotlib.cm as cm
import numpy as np

_TOP_K = 3


def _composite_image(gcam, raw_image):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    gcam = np.uint8(gcam)
    return gcam


def _generate(fmaps, grads, image_shape):
    # Compute grad weights
    weights = F.adaptive_avg_pool2d(grads, 1)

    gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
    gcam = F.relu(gcam)

    gcam = F.interpolate(
        gcam, image_shape, mode="bilinear", align_corners=False
    )

    B, C, H, W = gcam.shape
    gcam = gcam.view(B, -1)
    gcam -= gcam.min(dim=1, keepdim=True)[0]
    gcam /= gcam.max(dim=1, keepdim=True)[0]
    gcam = gcam.view(B, C, H, W)

    return gcam


class GradCAM(nn.Module):

    def _forward_helper(self, image):
        # Setup forward hook
        conv_output = None

        def save_layer_output(module: nn.Module, layer_input: torch.Tensor, layer_output: torch.Tensor):
            nonlocal conv_output
            conv_output = layer_output.detach()

        forward_handle = self.grad_cam_layer().register_forward_hook(save_layer_output)

        # Forward pass
        self.zero_grad()
        model_output = self.forward(image)

        forward_handle.remove()

        return model_output, conv_output

    def save_gradient(self, module, grad_input, grad_output):
        print("Backward hook triggered")
        self.foo = grad_output[0].detach()

    def _backward_helper(self, model_output: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        # Setup backward hook
        guided_gradients = None

        # def save_gradient(module, grad_input, grad_output):
        #     print("Backward hook triggered")
        #     nonlocal guided_gradients
        #     guided_gradients = grad_output[0].detach()

        backward_handle = self.grad_cam_layer().register_backward_hook(self.save_gradient)

        one_hot = model_output.new_zeros(model_output.size())
        one_hot.scatter_(1, ids, 1.0)

        model_output.backward(gradient=one_hot, retain_graph=True)

        # if guided_gradients is None:
        #     raise RuntimeError("Backward hook gave no gradients. See PyTorch issues related to register_backward_hook.")

        backward_handle.remove()

        return guided_gradients

    # TODO: rename image to images
    def generate_grad_cam(self,
                          images: torch.Tensor,
                          raw_images: list,
                          target_classes: Optional[torch.LongTensor] = None) -> List[List[np.array]]:
        self.eval()
        model_output, conv_output = self._forward_helper(images)

        if not target_classes:
            probs = F.softmax(model_output, dim=1)
            # TODO: use topk function
            top_probs, top_ids = probs.sort(dim=1, descending=True)
            target_classes = top_ids[:, :_TOP_K]

        results = [[]] * len(images)  # List with one empty list for each input image
        for target_class in target_classes.T:
            # Perform one backward operation for each target class and capture gradients
            guided_gradients = self._backward_helper(
                model_output,
                target_class.unsqueeze(dim=1)
            )

            image_shape = images.shape[2:]
            regions = _generate(conv_output, guided_gradients, image_shape)

            for i, region, raw_image in enumerate(zip(regions, raw_images)):
                result = _composite_image(region, raw_image)
                results[i].append(result)

        return results

    def grad_cam_layer(self) -> nn.Module:
        raise NotImplementedError("Return a layer in your model to target for Grad CAM.")
