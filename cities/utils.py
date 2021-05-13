import torch
import numpy as np
from collections import deque
import os
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


class Image_logger(object):
    def __init__(self):
        self.day_images = deque(maxlen=50)
        self.night_images = deque(maxlen=50)


class LR_Decay(object):
    def __init__(self, epochs, offset=0, decay_epoch=100):
        self.epochs = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch)/(self.epochs - self.decay_epoch)


def save_models(model_list, outdir="saved_models"):

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    torch.save(model_list[0].state_dict(), os.path.join(outdir, "day_gen.pt"))
    torch.save(model_list[1].state_dict(), os.path.join(outdir, "day_disc.pt"))
    torch.save(model_list[2].state_dict(), os.path.join(outdir, "night_gen.pt"))
    torch.save(model_list[3].state_dict(), os.path.join(outdir, "night_disc.pt"))


def plot_images(model_list, samples, iter_count, image_outdir="gen_images"):

    if not os.path.exists(image_outdir):
        os.mkdir(image_outdir)

    day_gen = model_list[0]
    night_gen = model_list[2]
    day_real = samples[0]
    night_real = samples[1]

    day_fake = night_gen(day_real)
    night_fake = day_gen(night_real)
    day_recon = day_gen(night_fake)
    night_recon = night_gen(day_fake)

    fig = plt.figure(figsize=(8, 8))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, 3),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for i, (ax, im) in enumerate(zip(grid, [day_real, night_fake, day_recon, night_real, day_fake, night_recon])):
        ax.imshow(im.detach().cpu().reshape(128, 128, 3))
        ax.axis("off")

    plt.savefig(os.path.join(image_outdir, "grid_"+str(iter_count)))
