# System Utils
import os
import math
import time

# Torch
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

# Local modules
from model import encoder, decoder
from utils.loss import UVSVENetLoss
from utils.loader import CustomLoader

# Hyperparameters, modify these
LEARNING_RATE = 5e-6
EPOCHS = 2
GPU = 0
# Set this to 1 when training videos
# Use any value when training images
BATCH_SIZE = 1

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5*math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm") != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def main():
    encoder_lv1 = encoder.Encoder()
    encoder_lv2 = encoder.Encoder()
    encoder_lv3 = encoder.Encoder()

    decoder_lv1 = decoder.Decoder()
    decoder_lv2 = decoder.Decoder()
    decoder_lv3 = decoder.Decoder()

    encoder_lv1.apply(weight_init).cuda(GPU)
    encoder_lv2.apply(weight_init).cuda(GPU)
    encoder_lv3.apply(weight_init).cuda(GPU)

    decoder_lv1.apply(weight_init).cuda(GPU)
    decoder_lv2.apply(weight_init).cuda(GPU)
    decoder_lv3.apply(weight_init).cuda(GPU)

    encoder_lv1_optim = torch.optim.Adam(
        encoder_lv1.parameters(), 
        lr = LEARNING_RATE
    )
    encoder_lv1_scheduler = StepLR(
        encoder_lv1_optim, 
        step_size = 10, 
        gamma = 0.1
    )
    encoder_lv2_optim = torch.optim.Adam(
        encoder_lv2.parameters(), 
        lr = LEARNING_RATE
    )
    encoder_lv2_scheduler = StepLR(
        encoder_lv2_optim, 
        step_size = 10, 
        gamma = 0.1
    )
    encoder_lv3_optim = torch.optim.Adam(
        encoder_lv3.parameters(), 
        lr = LEARNING_RATE
    )
    encoder_lv3_scheduler = StepLR(
        encoder_lv3_optim, 
        step_size = 10, 
        gamma = 0.1
    )

    decoder_lv1_optim = torch.optim.Adam(
        decoder_lv1.parameters(), 
        lr = LEARNING_RATE
    )
    decoder_lv1_scheduler = StepLR(
        decoder_lv1_optim, 
        step_size = 10, 
        gamma = 0.1
    )
    decoder_lv2_optim = torch.optim.Adam(
        decoder_lv2.parameters(), 
        lr = LEARNING_RATE
    )
    decoder_lv2_scheduler = StepLR(
        decoder_lv2_optim, 
        step_size = 10, 
        gamma = 0.1
    )
    decoder_lv3_optim = torch.optim.Adam(
        decoder_lv3.parameters(), 
        lr = LEARNING_RATE
    )
    decoder_lv3_scheduler = StepLR(
        decoder_lv3_optim, 
        step_size = 10, 
        gamma = 0.1
    )

    if os.path.exists("./saved/encoder_lv1.pkl"):
        encoder_lv1.load_state_dict(
            torch.load("./saved/encoder_lv1.pkl")
        )
        print("Loaded Level 1 Encoder")

    if os.path.exists("./saved/encoder_lv2.pkl"):
        encoder_lv2.load_state_dict(
            torch.load("./saved/encoder_lv2.pkl")
        )
        print("Loaded Level 2 Encoder")

    if os.path.exists("./saved/encoder_lv3.pkl"):
        encoder_lv3.load_state_dict(
            torch.load("./saved/encoder_lv3.pkl")
        )
        print("Loaded Level 3 Encoder")

    if os.path.exists("./saved/decoder_lv1.pkl"):
        decoder_lv1.load_state_dict(
            torch.load("./saved/decoder_lv1.pkl")
        )
        print("Loaded Level 1 Decoder")

    if os.path.exists("./saved/decoder_lv2.pkl"):
        decoder_lv2.load_state_dict(
            torch.load("./saved/decoder_lv2.pkl")
        )
        print("Loaded Level 2 Decoder")

    if os.path.exists("./saved/decoder_lv3.pkl"):
        decoder_lv3.load_state_dict(
            torch.load("./saved/decoder_lv3.pkl")
        )
        print("Loaded Level 3 Decoder")

    if os.path.exists("./saved") == False:
        os.system("mkdir saved")
    
    main_start = time.time()

    for epoch in range(0, EPOCHS):
        encoder_lv1_scheduler.step(epoch)
        encoder_lv2_scheduler.step(epoch)
        encoder_lv3_scheduler.step(epoch)

        decoder_lv1_scheduler.step(epoch)
        decoder_lv2_scheduler.step(epoch)
        decoder_lv3_scheduler.step(epoch)

        print("Training...")

        dataset = CustomLoader(
            # Replace these with custom files
            distorted_list_file_path = "./dataset/uw_raw.txt",
            restored_list_file_path = "./dataset/uw_reference.txt",
            dataset_root = "",
        )
        dataloader = DataLoader(
            dataset, 
            batch_size = BATCH_SIZE, 
            # Shuffling would shuffle our frames, we don't need that happening
            shuffle = False
        )

        start = time.time()
        prev_img = None
        loss = UVSVENetLoss()
        loss_fn = loss.cuda(GPU)

        for iteration, images in enumerate(dataloader):
            image_name = images["name"]
            if image_name != prev_img:
                loss.out_prev = None

            clean_frame = Variable(images["restored_frame"] - 0.5).cuda(GPU)
            H = clean_frame.size(2)
            W = clean_frame.size(3)
            n_H = H
            n_W = W

            images_lv1 = Variable(images["distorted_frame"] - 0.5).cuda(GPU)

            images_lv2_1 = images_lv1[:, :, 0:int(H/2), :]
            images_lv2_2 = images_lv1[:, :, int(H/2):H, :]
            images_lv3_1 = images_lv2_1[:, :, :, 0:int(W/2)]
            images_lv3_2 = images_lv2_1[:, :, :, int(W/2):W]
            images_lv3_3 = images_lv2_2[:, :, :, 0:int(W/2)]
            images_lv3_4 = images_lv2_2[:, :, :, int(W/2):W]

            feature_lv3_1 = encoder_lv3(images_lv3_1)
            feature_lv3_2 = encoder_lv3(images_lv3_2)
            feature_lv3_3 = encoder_lv3(images_lv3_3)
            feature_lv3_4 = encoder_lv3(images_lv3_4)
            feature_lv3_top = torch.cat(
                (feature_lv3_1, feature_lv3_2), 3)
            feature_lv3_bot = torch.cat(
                (feature_lv3_3, feature_lv3_4), 3)
            feature_lv3 = torch.cat(
                (feature_lv3_top, feature_lv3_bot), 2)
            residual_lv3_top = decoder_lv3(feature_lv3_top)
            residual_lv3_bot = decoder_lv3(feature_lv3_bot)
            feature_lv2_1 = encoder_lv2(
                images_lv2_1 + residual_lv3_top[:images_lv2_1.shape[0], :images_lv2_1.shape[1], :images_lv2_1.shape[2], :images_lv2_1.shape[3]])
            feature_lv2_2 = encoder_lv2(
                images_lv2_2 + residual_lv3_bot[:images_lv2_2.shape[0], :images_lv2_2.shape[1], :images_lv2_2.shape[2], :images_lv2_2.shape[3]])
            feature_lv2_pre = torch.cat(
                (feature_lv2_1, feature_lv2_2), 2)
            feature_lv2 = feature_lv2_pre + \
                feature_lv3[:feature_lv2_pre.shape[0], :feature_lv2_pre.shape[1],
                            :feature_lv2_pre.shape[2], :feature_lv2_pre.shape[3]]
            residual_lv2 = decoder_lv2(feature_lv2)

            feature_lv1_pre = encoder_lv1(
                images_lv1 + residual_lv2[:images_lv1.shape[0], :images_lv1.shape[1], :images_lv1.shape[2], :images_lv1.shape[3]])
            feature_lv1 = feature_lv1_pre + \
                feature_lv2[:feature_lv1_pre.shape[0], :feature_lv1_pre.shape[1],
                            :feature_lv1_pre.shape[2], :feature_lv1_pre.shape[3]]
            restored_frame = decoder_lv1(feature_lv1)

            loss_lv1 = loss_fn(
                restored_frame[
                    :clean_frame.shape[0], 
                    :clean_frame.shape[1], 
                    :clean_frame.shape[2], 
                    :clean_frame.shape[3]
                ], 
                clean_frame
            )

            loss = loss_lv1

            encoder_lv1.zero_grad()
            encoder_lv2.zero_grad()
            encoder_lv3.zero_grad()

            decoder_lv1.zero_grad()
            decoder_lv2.zero_grad()
            decoder_lv3.zero_grad()

            loss.backward()

            encoder_lv1_optim.step()
            encoder_lv2_optim.step()
            encoder_lv3_optim.step()

            decoder_lv1_optim.step()
            decoder_lv2_optim.step()
            decoder_lv3_optim.step()

            prev_img = image_name

            if (iteration+1) % 10 == 0:
                current_time = time.time()
                time_taken = current_time - start
                eta = (
                    (current_time - main_start) / 
                    (
                        ((epoch) * len(dataloader)) + 
                        (iteration + 1)
                    )
                ) * (
                    (EPOCHS - epoch - 1) * len(dataloader) + 
                    (len(dataloader) - iteration - 1)
                )
                print(
                    "Epoch: ", epoch, 
                    "Iteration: ", iteration + 1,
                    "Loss: %.4f" % loss.item(), 
                    "Time: %.4f" % time_taken + "s",
                    "ETA: %.4f" % eta + "s"
                )
                start = time.time()

        torch.save(
            encoder_lv1.state_dict(), 
            "./saved/encoder_lv1.pkl"
        )
        torch.save(
            encoder_lv2.state_dict(), 
            "./saved/encoder_lv2.pkl"
        )
        torch.save(
            encoder_lv3.state_dict(), 
            "./saved/encoder_lv3.pkl"
        )

        torch.save(
            decoder_lv1.state_dict(), 
            "./saved/decoder_lv1.pkl"
        )
        torch.save(
            decoder_lv2.state_dict(), 
            "./saved/decoder_lv2.pkl"
        )
        torch.save(
            decoder_lv3.state_dict(), 
            "./saved/decoder_lv3.pkl"
        )


if __name__ == "__main__":
    main()
