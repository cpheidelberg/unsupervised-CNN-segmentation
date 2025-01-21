import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
import os

# Check if GPU is available
use_cuda = torch.cuda.is_available()

# Argument parser
parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation with Iterative Refinement')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
parser.add_argument('--input', type=str, required=True, help='Path to the input image')
parser.add_argument('--output', type=str, default='output.png', help='Path to save the output segmented image')
parser.add_argument('--nChannel', type=int, default=100, help='Number of channels used in the trained model')
parser.add_argument('--nConv', type=int, default=2, help='Number of convolutional layers in the trained model')
parser.add_argument('--iterations', type=int, default=3, help='Number of optimization iterations')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for optimization')
args = parser.parse_args()

# CNN model definition
class MyNet(nn.Module):
    def __init__(self, input_dim, nChannel, nConv):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for _ in range(nConv - 1):
            self.conv2.append(nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(nChannel))
        self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(len(self.conv2)):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

# Load the image
def load_image(image_path):
    im = cv2.imread(image_path)
    if im is None:
        raise ValueError(f"Failed to load image: {image_path}")
    data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
    if use_cuda:
        data = data.cuda()
    return data, im

# Optimization loop for refining segmentation
def optimize_segmentation(model, data, im):
    model.train()  # Set to train mode for optimization
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    loss_hpy = nn.L1Loss()
    loss_hpz = nn.L1Loss()

    HPy_target = torch.zeros(im.shape[0] - 1, im.shape[1], args.nChannel)
    HPz_target = torch.zeros(im.shape[0], im.shape[1] - 1, args.nChannel)
    if use_cuda:
        HPy_target = HPy_target.cuda()
        HPz_target = HPz_target.cuda()

    for i in range(args.iterations):
        optimizer.zero_grad()
        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)

        outputHP = output.view(im.shape[0], im.shape[1], args.nChannel)
        HPy = outputHP[1:, :, :] - outputHP[:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, :-1, :]

        lhpy = loss_hpy(HPy, HPy_target)
        lhpz = loss_hpz(HPz, HPz_target)

        _, target = torch.max(output, 1)

        # Compute the loss
        loss = loss_fn(output, target) + (lhpy + lhpz)
        loss.backward()
        optimizer.step()

        print(f"Iteration {i + 1}/{args.iterations}, Loss: {loss.item()}")

    return target.cpu().numpy()

# Load the trained model
def load_model(model_path, nChannel, nConv):
    model = MyNet(input_dim=3, nChannel=nChannel, nConv=nConv)
    if use_cuda:
        model.load_state_dict(torch.load(model_path))
        model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def main():
	# Load the model
	print(f"Loading model from {args.model_path}")
	model = load_model(args.model_path, args.nChannel, args.nConv)

	# Load the input image
	print(f"Processing image: {args.input}")
	data, im = load_image(args.input)

	# Optimize segmentation
	segmented_labels = optimize_segmentation(model, data, im)

	# Generate color-mapped output
	label_colours = np.array([
		(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 165, 0),
		(128, 0, 128), (0, 255, 255), (255, 192, 203), (128, 128, 0), (75, 0, 130),
		(255, 69, 0), (255, 20, 147), (0, 128, 128), (50, 205, 50), (138, 43, 226),
		(220, 20, 60), (240, 230, 140), (255, 105, 180), (154, 205, 50), (70, 130, 180),
		(244, 164, 96), (32, 178, 170), (255, 99, 71), (210, 105, 30), (127, 255, 212),
		(105, 105, 105), (192, 192, 192), (47, 79, 79), (255, 215, 0), (123, 104, 238),
		(173, 216, 230), (0, 191, 255), (184, 134, 11), (0, 255, 127), (255, 218, 185),
		(255, 160, 122), (46, 139, 87), (250, 128, 114), (255, 222, 173), (0, 206, 209),
		(233, 150, 122), (165, 42, 42), (255, 250, 205), (139, 69, 19), (153, 50, 204),
		(72, 209, 204), (147, 112, 219), (238, 130, 238), (255, 182, 193), (255, 0, 255),
		(0, 128, 0), (255, 248, 220), (250, 235, 215), (100, 149, 237), (255, 228, 196),
		(160, 82, 45), (127, 255, 0), (189, 183, 107), (0, 139, 139), (65, 105, 225),
		(139, 0, 0), (255, 140, 0), (85, 107, 47), (218, 165, 32), (95, 158, 160),
		(30, 144, 255), (72, 61, 139), (199, 21, 133), (178, 34, 34), (70, 130, 180),
		(255, 228, 225), (222, 184, 135), (128, 0, 0), (0, 255, 255), (173, 255, 47),
		(144, 238, 144), (0, 250, 154), (186, 85, 211), (250, 250, 210), (0, 0, 139),
		(255, 228, 181), (240, 128, 128), (0, 128, 128), (255, 255, 255), (240, 255, 240),
		(224, 255, 255), (245, 245, 220), (0, 0, 128), (240, 255, 255), (245, 255, 250),
		(47, 79, 79), (139, 0, 139), (176, 224, 230), (245, 245, 245), (255, 250, 240),
		(255, 255, 240), (255, 235, 205), (255, 239, 213), (253, 245, 230), (245, 222, 179),
	])

	segmented_image = np.array([label_colours[c % len(label_colours)] for c in segmented_labels])
	segmented_image = segmented_image.reshape(im.shape).astype(np.uint8)

	# Save the segmented output
	cv2.imwrite(args.output, segmented_image)
	print(f"Segmented output saved to {args.output}")

if __name__ == "__main__":
    main()
