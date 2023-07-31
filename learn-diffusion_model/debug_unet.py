import sys

sys.path.append('../../')
import torch
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    model = UNetModel(image_size=32,
                      in_channels=4,
                      out_channels=4,
                      model_channels=320,
                      attention_resolutions=[4, 2, 1],
                      num_res_blocks=2,
                      channel_mult=[1, 2, 4, 4],
                      num_heads=8,
                      use_spatial_transformer=True,
                      transformer_depth=1,
                      context_dim=768,
                      use_checkpoint=True,
                      legacy=False)

    writer = SummaryWriter('./tensorboard/debug')
    x = torch.randn(1, 4, 32, 32)
    timesteps = torch.tensor([1], dtype=torch.long)
    context = torch.randn(1, 77, 768)
    # writer.add_graph(model, [x, timesteps, context])
    # writer.close()

    y = model(x, timesteps=timesteps, context=context)

    # print(model)

    # model_scripted = torch.jit.trace(model, [x, timesteps, context])
    # model_scripted.eval()
    # model_out_path = './unet.pt'
    # model_scripted.save(model_out_path)