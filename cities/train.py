import argparse
import random
import itertools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_processing import ImagesDataset
from utils import Image_logger, save_models, plot_images, LR_Decay
from models import Generator, Discriminator


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", action="store_true")
    return parser.parse_args()


def main():

    args = parse_args()
    device = torch.device("cuda" if args.use_cuda else "cpu")
    night_loss = nn.MSELoss()
    day_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    lambda_ = 10    # cycle-consistency loss coefficient
    epochs = 200    # number of epochs in training
    save_freq = 2   # saves every save_freq epochs
    plot_freq = 50  # saves every plot_freq iterations

    day_generator = Generator().to(device)
    day_discriminator = Discriminator().to(device)
    night_generator = Generator().to(device)
    night_discriminator = Discriminator().to(device)

    model_list = [day_generator, day_discriminator, night_generator, night_discriminator]

    gen_optimizer = torch.optim.Adam(itertools.chain(day_generator.parameters(), night_generator.parameters()), lr=2e-4)
    disc_optimizer = torch.optim.Adam(itertools.chain(day_discriminator.parameters(), night_discriminator.parameters())
                                      , lr=2e-4)

    gen_scheduler = torch.optim.lr_scheduler.LambdaLR(gen_optimizer, lr_lambda=LR_Decay(epochs).step)
    disc_scheduler = torch.optim.lr_scheduler.LambdaLR(disc_optimizer, lr_lambda=LR_Decay(epochs).step)

    image_logger = Image_logger()

    img_dir = "images"

    dataset = ImagesDataset(img_dir)
    dataloader = DataLoader(dataset=dataset, batch_size=1, drop_last=True)

    total_iterations = 0
    for epoch in range(epochs):
        for i, samples in enumerate(dataloader):

            day_batch = samples[0].to(device)
            night_batch = samples[1].to(device)

            gen_optimizer.zero_grad()

            day_fake = day_generator(night_batch)
            night_fake = night_generator(day_batch)

            # store generated images in logger
            image_logger.day_images.append(day_fake.detach())
            image_logger.night_images.append(night_fake.detach())

            day_recon = day_generator(night_fake)
            night_recon = night_generator(day_fake)

            # identity loss
            day_idt_loss = lambda_ * .5 * l1_loss(day_recon, day_batch)
            night_idt_loss = lambda_ * .5 * l1_loss(night_recon, night_batch)

            # adversarial loss
            day_fake_pred = day_discriminator(day_fake)
            night_fake_pred = night_discriminator(night_fake)

            label = torch.autograd.Variable(torch.ones(day_fake_pred.size()))

            day_gen_loss = day_loss(night_fake_pred, label)
            night_gen_loss = night_loss(day_fake_pred, label)

            # cycle-consistency loss
            day_cyc_loss = lambda_ * l1_loss(day_recon, day_batch)
            night_cyc_loss = lambda_ * l1_loss(night_recon, night_batch)

            # overall loss
            gen_loss = day_gen_loss + night_gen_loss + day_cyc_loss + night_cyc_loss + day_idt_loss + night_idt_loss

            gen_loss.backward()
            gen_optimizer.step()

            # --------------------------------------- #
            ###          train discriminator        ###
            # --------------------------------------- #
            if total_iterations > 50:
                disc_optimizer.zero_grad()

                day_fake = random.choice(image_logger.day_images)
                day_fake = torch.autograd.Variable(day_fake)

                night_fake = random.choice(image_logger.night_images)
                night_fake = torch.autograd.Variable(night_fake)

                day_real_pred = day_discriminator(day_batch)
                day_fake_pred = day_discriminator(day_fake)

                night_real_pred = night_discriminator(night_batch)
                night_fake_pred = night_discriminator(night_fake)

                real_label = torch.autograd.Variable(torch.ones(day_real_pred.size())).to(device)
                fake_label = torch.autograd.Variable(torch.ones(night_fake_pred.size())).to(device)

                day_disc_real_loss = day_loss(day_real_pred, real_label)
                day_disc_fake_loss = day_loss(day_fake_pred, fake_label)
                night_disc_real_loss = night_loss(night_real_pred, real_label)
                night_disc_fake_loss = night_loss(night_fake_pred, fake_label)

                day_disc_loss = (day_disc_real_loss+day_disc_fake_loss)/2.
                night_disc_loss = (night_disc_real_loss+night_disc_fake_loss)/2.

                day_disc_loss.backward()
                night_disc_loss.backward()

                disc_optimizer.step()

                print("Epoch: {}/{}, Gen Loss: {:.3f}, Disc Loss: {:.3f}".format(epoch+1, epochs, gen_loss,
                                                                                 day_disc_loss+night_disc_loss))

            total_iterations += 1

            if total_iterations % plot_freq == 0:
                plot_images(model_list, samples, total_iterations)

        if epoch % save_freq == 0:
            save_models(model_list)

        dataset.shuffle_keys()
        print(total_iterations)

        gen_scheduler.step()
        disc_scheduler.step()


if __name__ == "__main__":
    main()